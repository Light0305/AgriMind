"""Tests for app.retrieval — similar case image retrieval."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
from PIL import Image

from app.retrieval.similar_cases import (
    PHashEmbedder,
    SimilarCaseRetriever,
    _IMAGEHASH_AVAILABLE,
    _TORCH_AVAILABLE,
)
from app.schemas import SimilarCase


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_HAS_EMBEDDER = _TORCH_AVAILABLE or _IMAGEHASH_AVAILABLE
_skip_no_embedder = pytest.mark.skipif(
    not _HAS_EMBEDDER,
    reason="Neither torch nor imagehash available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(path, color=(255, 0, 0), size=(64, 64)):
    """Create a small solid-color PNG for testing."""
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def work_dir():
    d = tempfile.mkdtemp(prefix="test_retrieval_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def db_dir(work_dir):
    return os.path.join(work_dir, "case_db")


@pytest.fixture()
def retriever(db_dir):
    return SimilarCaseRetriever(persist_dir=db_dir)


@pytest.fixture()
def sample_images(work_dir):
    """Create three test images with different colors."""
    imgs = {
        "red": _create_test_image(os.path.join(work_dir, "red.png"), color=(255, 0, 0)),
        "red2": _create_test_image(os.path.join(work_dir, "red2.png"), color=(250, 5, 5)),
        "blue": _create_test_image(os.path.join(work_dir, "blue.png"), color=(0, 0, 255)),
    }
    return imgs


# ---------------------------------------------------------------------------
# ImageEmbedder / PHashEmbedder
# ---------------------------------------------------------------------------

class TestImageEmbedding:
    @_skip_no_embedder
    def test_returns_correct_dimensions(self, retriever, sample_images):
        emb = retriever._compute_image_embedding(sample_images["red"])
        if _TORCH_AVAILABLE:
            assert len(emb) == 512  # ResNet-18 feature dim
        else:
            assert len(emb) == 256  # pHash 16x16

    @_skip_no_embedder
    def test_all_values_are_float(self, retriever, sample_images):
        emb = retriever._compute_image_embedding(sample_images["red"])
        assert all(isinstance(v, float) for v in emb)

    @_skip_no_embedder
    def test_similar_images_have_close_embeddings(self, retriever, sample_images):
        emb_red = retriever._compute_image_embedding(sample_images["red"])
        emb_red2 = retriever._compute_image_embedding(sample_images["red2"])
        emb_blue = retriever._compute_image_embedding(sample_images["blue"])

        # L2 distance: similar images should be closer
        import math
        dist_similar = math.sqrt(sum((a - b) ** 2 for a, b in zip(emb_red, emb_red2)))
        dist_different = math.sqrt(sum((a - b) ** 2 for a, b in zip(emb_red, emb_blue)))
        assert dist_similar <= dist_different

    @_skip_no_embedder
    def test_embed_accepts_pil_image(self, retriever, sample_images):
        """Embedder.embed() should accept a PIL.Image directly."""
        img = Image.open(sample_images["red"]).convert("RGB")
        emb = retriever.embedder.embed(img)
        assert len(emb) > 0
        assert all(isinstance(v, float) for v in emb)


# ---------------------------------------------------------------------------
# PHashEmbedder standalone (if imagehash available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IMAGEHASH_AVAILABLE, reason="imagehash not installed")
class TestPHashEmbedder:
    def test_phash_produces_256_dim(self, sample_images):
        embedder = PHashEmbedder()
        img = Image.open(sample_images["red"]).convert("RGB")
        emb = embedder.embed(img)
        assert len(emb) == 256

    def test_phash_values_are_binary_floats(self, sample_images):
        embedder = PHashEmbedder()
        img = Image.open(sample_images["red"]).convert("RGB")
        emb = embedder.embed(img)
        assert all(v in (0.0, 1.0) for v in emb)


# ---------------------------------------------------------------------------
# Index and find_similar round-trip
# ---------------------------------------------------------------------------

@_skip_no_embedder
class TestIndexAndFindSimilar:
    def test_index_single_and_find(self, retriever, sample_images):
        retriever.index_single(
            image_path=sample_images["red"],
            label="番茄晚疫病",
            source="test",
        )
        results = retriever.find_similar(sample_images["red"], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SimilarCase)
        assert results[0].label == "番茄晚疫病"
        assert results[0].source == "test"
        assert results[0].similarity >= 0.99  # self-match

    def test_find_returns_multiple(self, retriever, sample_images):
        retriever.index_single(sample_images["red"], label="病害A")
        retriever.index_single(sample_images["red2"], label="病害B")
        retriever.index_single(sample_images["blue"], label="病害C")

        results = retriever.find_similar(sample_images["red"], top_k=3)
        assert len(results) == 3
        # First result should be the exact or nearest match
        assert results[0].label in ("病害A", "病害B")

    def test_find_similar_on_empty_collection(self, retriever, sample_images):
        results = retriever.find_similar(sample_images["red"], top_k=3)
        assert results == []

    def test_find_similar_accepts_pil_image(self, retriever, sample_images):
        """find_similar() should accept both file paths and PIL.Image."""
        retriever.index_single(sample_images["red"], label="病害D")
        # Pass a PIL.Image instead of a path
        img = Image.open(sample_images["red"]).convert("RGB")
        results = retriever.find_similar(img, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SimilarCase)

    def test_index_cases_from_jsonl(self, retriever, sample_images, work_dir):
        jsonl = os.path.join(work_dir, "dataset.jsonl")
        records = [
            {"image_path": sample_images["red"], "label": "苹果黑星病", "source": "ds1"},
            {"image_path": sample_images["blue"], "label": "葡萄霜霉病", "source": "ds2"},
            {"image_path": "/nonexistent/img.png", "label": "skip", "source": "bad"},
        ]
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records))

        count = retriever.index_cases(str(jsonl))
        assert count == 2  # third entry skipped (file doesn't exist)

        results = retriever.find_similar(sample_images["red"], top_k=2)
        assert len(results) == 2
        labels = {r.label for r in results}
        assert "苹果黑星病" in labels


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@_skip_no_embedder
class TestResultSchema:
    def test_result_is_similar_case(self, retriever, sample_images):
        retriever.index_single(sample_images["red"], label="test")
        results = retriever.find_similar(sample_images["red"], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SimilarCase)
        assert isinstance(r.similarity, float)
        assert isinstance(r.label, str)
        assert isinstance(r.image_path, str)
        assert isinstance(r.source, str)
