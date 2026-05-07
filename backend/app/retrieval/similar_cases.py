"""Retrieve visually similar diagnosis cases from a ChromaDB case library.

Supports two embedding strategies:

1. **ResNet-18** (default when torch is available) — uses a pre-trained
   ResNet-18 with its classifier head removed, producing a 512-dim feature
   vector.  Robust and accurate for "looks similar" retrieval.
2. **Perceptual hashing** (fallback) — uses pHash via ``imagehash``.
   Lightweight, no GPU, works offline.  256-dim binary vector.

Swap ``_compute_image_embedding`` for CLIP when GPU budget allows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import chromadb
from PIL import Image

from app.schemas import SimilarCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — graceful fallback
# ---------------------------------------------------------------------------

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    import imagehash

    _IMAGEHASH_AVAILABLE = True
except ImportError:  # pragma: no cover
    imagehash = None  # type: ignore[assignment]
    _IMAGEHASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image embedders
# ---------------------------------------------------------------------------

class ImageEmbedder:
    """ResNet-18 feature extractor (512-dim vector, no classifier head)."""

    EMBED_DIM = 512

    def __init__(self) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch and torchvision are required for ImageEmbedder — "
                "pip install torch torchvision"
            )
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # Remove classifier
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def embed(self, image: Image.Image) -> list[float]:
        """Produce a 512-dim float vector from *image*."""
        with torch.no_grad():
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
            features = self.model(tensor).squeeze().tolist()
        return features  # 512-dim


class PHashEmbedder:
    """Lightweight perceptual-hash embedder (256-dim, no GPU needed)."""

    HASH_SIZE = 16  # 16×16 = 256-dim
    EMBED_DIM = 256

    def embed(self, image: Image.Image) -> list[float]:
        if not _IMAGEHASH_AVAILABLE:
            raise RuntimeError(
                "imagehash is required for pHash embedding — "
                "pip install imagehash>=4.3.0"
            )
        phash = imagehash.phash(image.convert("RGB"), hash_size=self.HASH_SIZE)
        return [float(b) for b in phash.hash.flatten()]


def _get_default_embedder() -> ImageEmbedder | PHashEmbedder:
    """Return ResNet-18 if torch available, else pHash fallback."""
    if _TORCH_AVAILABLE:
        return ImageEmbedder()
    return PHashEmbedder()


# ---------------------------------------------------------------------------
# Similar case retriever
# ---------------------------------------------------------------------------

class SimilarCaseRetriever:
    """Index & retrieve visually similar crop-disease cases."""

    def __init__(
        self,
        persist_dir: str = "data/chromadb/cases",
        embedder: ImageEmbedder | PHashEmbedder | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="case_library",
            metadata={"hnsw:space": "cosine"},
        )
        # Lazy-init: only create embedder when first needed
        self._embedder = embedder

    @property
    def embedder(self) -> ImageEmbedder | PHashEmbedder:
        if self._embedder is None:
            self._embedder = _get_default_embedder()
        return self._embedder

    # Expose the hash size constant for backward-compat with existing tests
    @property
    def HASH_SIZE(self) -> int:  # noqa: N802
        if isinstance(self.embedder, PHashEmbedder):
            return self.embedder.HASH_SIZE
        return 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_cases(self, dataset_file: str) -> int:
        """Index cases from a JSONL file (one JSON object per line).

        Expected schema per line::

            {"image_path": "...", "label": "...", "source": "..."}

        Returns the number of cases successfully indexed.
        """
        count = 0
        with open(dataset_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_path = record["image_path"]
                if not Path(image_path).is_file():
                    continue
                try:
                    self.index_single(
                        image_path=image_path,
                        label=record.get("label", ""),
                        source=record.get("source", ""),
                    )
                    count += 1
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to index %s, skipping", image_path)
        return count

    def index_single(
        self,
        image_path: str,
        label: str,
        source: str = "",
    ) -> str:
        """Index one image from a file path. Returns the document ID."""
        image = Image.open(image_path).convert("RGB")
        embedding = self.embedder.embed(image)
        doc_id = self._stable_id(image_path)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[label],
            metadatas=[
                {
                    "image_path": image_path,
                    "label": label,
                    "source": source,
                }
            ],
        )
        return doc_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_similar(
        self,
        image: Image.Image | str,
        top_k: int = 3,
    ) -> list[SimilarCase]:
        """Find the most similar indexed cases to *image*.

        *image* can be a ``PIL.Image`` or a file path string.
        Returns a list of ``SimilarCase`` objects.
        """
        if self.collection.count() == 0:
            return []

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        embedding = self.embedder.embed(image)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
        )

        formatted: list[SimilarCase] = []
        if not results["ids"] or not results["ids"][0]:
            return formatted

        for idx, _doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            similarity = 1.0 - distance
            meta = results["metadatas"][0][idx]
            formatted.append(
                SimilarCase(
                    image_path=meta.get("image_path", ""),
                    label=meta.get("label", ""),
                    similarity=round(similarity, 4),
                    source=meta.get("source", ""),
                )
            )
        return formatted

    # ------------------------------------------------------------------
    # Backward-compat: expose the raw embedding method
    # ------------------------------------------------------------------

    def _compute_image_embedding(self, image_path: str) -> list[float]:
        """Produce an embedding vector from an image file path."""
        image = Image.open(image_path).convert("RGB")
        return self.embedder.embed(image)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_id(image_path: str) -> str:
        return hashlib.sha256(image_path.encode()).hexdigest()[:16]
