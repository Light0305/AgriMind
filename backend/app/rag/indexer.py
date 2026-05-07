"""Build and maintain the ChromaDB knowledge index for agricultural treatment data."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PDF support — gracefully degrade if not installed.
# ---------------------------------------------------------------------------

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Seed knowledge — 25 common crop disease entries so the system works out of
# the box without any external documents.
# ---------------------------------------------------------------------------

DEFAULT_KNOWLEDGE: list[dict] = [
    # ── 小麦 Wheat ──────────────────────────────────────────────
    {
        "text": (
            "小麦条锈病（Puccinia striiformis）防治：发病初期可用15%三唑酮可湿性粉剂"
            "1000-1500倍液喷雾，或25%丙环唑乳油2000倍液。注意在孕穗至抽穗期重点防治，"
            "间隔7-10天再喷一次。安全间隔期20天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "小麦赤霉病（Fusarium graminearum）防治：在小麦扬花初期用50%多菌灵可湿性"
            "粉剂500倍液喷雾，或40%戊唑·咪鲜胺水乳剂20-25毫升/亩。花期遇连阴雨需在"
            "首次用药5-7天后补喷一次。注意交替用药。安全间隔期14天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "小麦白粉病（Blumeria graminis）防治：发病初期用15%三唑酮可湿性粉剂"
            "1000倍液或20%三唑酮乳油1000-1500倍液叶面喷雾。病情严重时可用"
            "40%氟硅唑乳油8000-10000倍液，间隔10天再施一次。安全间隔期21天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "小麦纹枯病（Rhizoctonia cerealis）防治：返青拔节期病株率达15%时"
            "用5%井冈霉素水剂150-200毫升/亩对水喷施茎基部。也可用12.5%烯唑醇"
            "可湿性粉剂30-40克/亩喷雾。重点喷植株基部，间隔7-10天施药一次。"
        ),
        "source": "植保手册",
    },
    # ── 水稻 Rice ────────────────────────────────────────────────
    {
        "text": (
            "水稻稻瘟病（Magnaporthe oryzae）防治：叶瘟初期用75%三环唑可湿性粉剂"
            "1500-2000倍液喷雾；穗颈瘟在破口至齐穗期用40%稻瘟灵乳油600-800倍液"
            "或2%春雷霉素水剂500倍液喷施。连续阴雨天气需在齐穗后5-7天补喷。"
            "安全间隔期21天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "水稻纹枯病（Rhizoctonia solani）防治：分蘖末期至抽穗期病丛率达20%"
            "时用5%井冈霉素水剂100-150毫升/亩或24%噻呋酰胺悬浮剂15-20毫升/亩"
            "对水喷雾，重点喷中下部叶鞘。严重田块间隔7天再喷一次。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "水稻白叶枯病（Xanthomonas oryzae）防治：发病初期用20%噻菌铜悬浮剂"
            "100-150毫升/亩或72%农用硫酸链霉素可溶性粉剂2000倍液喷雾。台风暴雨"
            "过后应及时喷药预防。注意排水降湿以减轻病害。"
        ),
        "source": "植保手册",
    },
    # ── 玉米 Corn / Maize ────────────────────────────────────────
    {
        "text": (
            "玉米大斑病（Exserohilum turcicum）防治：在心叶末期至抽雄期发病初期"
            "用50%多菌灵可湿性粉剂500倍液或75%百菌清可湿性粉剂500倍液喷雾，"
            "重点喷中下部叶片。每7-10天喷一次，连续防治2-3次。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "玉米小斑病（Bipolaris maydis）防治：发病初期用25%丙环唑乳油1500倍液"
            "或70%甲基硫菌灵可湿性粉剂600倍液喷雾。注意增施钾肥提高抗病性，"
            "合理密植保证通风透光。安全间隔期14天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "玉米锈病（Puccinia sorghi）防治：发病初期用15%三唑酮可湿性粉剂"
            "800-1000倍液或25%丙环唑乳油1500倍液均匀喷雾。阴雨天气后及时巡查，"
            "发现中心病株立即喷药控制蔓延。"
        ),
        "source": "植保手册",
    },
    # ── 番茄 Tomato ──────────────────────────────────────────────
    {
        "text": (
            "番茄晚疫病（Phytophthora infestans）防治：发病前可喷施保护性杀菌剂"
            "如75%百菌清600倍液预防。发病后用68%精甲霜·锰锌500倍液或"
            "72%霜脲·锰锌600-800倍液喷雾，每7天一次，连续2-3次。避免在阴雨天喷药。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "番茄早疫病（Alternaria solani）防治：发病初期用75%百菌清可湿性粉剂"
            "500-600倍液或70%代森锰锌可湿性粉剂500倍液喷雾，每7-10天一次。"
            "高温高湿季节注意加强通风，及时摘除下部老叶病叶。安全间隔期15天。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "番茄灰霉病（Botrytis cinerea）防治：发病初期用50%腐霉利可湿性粉剂"
            "1000-1500倍液或40%嘧霉胺悬浮剂800-1000倍液喷雾。棚室栽培注意降低"
            "湿度和适当通风，蘸花药中可加入0.1%的50%腐霉利预防。"
        ),
        "source": "植保手册",
    },
    # ── 马铃薯 Potato ────────────────────────────────────────────
    {
        "text": (
            "马铃薯早疫病（Alternaria solani）防治：发病初期用75%百菌清可湿性粉剂"
            "500倍液或64%噁霜·锰锌可湿性粉剂400-500倍液喷雾。在现蕾开花期加强"
            "防治，每7-10天喷一次，连续3-4次。注意轮换用药防止抗药性。"
        ),
        "source": "植保手册",
    },
    {
        "text": (
            "马铃薯晚疫病（Phytophthora infestans）防治：发病前用保护性杀菌剂"
            "72%霜脲·锰锌600倍液预防。发病后用68%精甲霜·锰锌500倍液喷雾，"
            "7天一次。连阴雨天气注意排水降湿，及时拔除中心病株。安全间隔期15天。"
        ),
        "source": "植保手册",
    },
    # ── 苹果 Apple ───────────────────────────────────────────────
    {
        "text": (
            "苹果黑星病/苹果斑点落叶病（Venturia inaequalis）防治：萌芽前喷5°Bé"
            "石硫合剂清园；花后7-10天用10%苯醚甲环唑水分散粒剂3000-4000倍液"
            "或70%甲基硫菌灵可湿性粉剂800-1000倍液喷雾。雨季增加喷药频次。"
        ),
        "source": "果树病虫害防治手册",
    },
    {
        "text": (
            "苹果褐斑病（Marssonina coronaria）防治：谢花后至6月用80%代森锰锌"
            "可湿性粉剂600-800倍液或43%戊唑醇悬浮剂3000-4000倍液喷雾。秋季"
            "彻底清扫落叶、深翻土壤以减少菌源。安全间隔期28天。"
        ),
        "source": "果树病虫害防治手册",
    },
    # ── 葡萄 Grape ───────────────────────────────────────────────
    {
        "text": (
            "葡萄霜霉病（Plasmopara viticola）防治：发芽后新梢长到20-30厘米时"
            "开始喷施78%波尔多液可湿性粉剂500倍液预防。发病后用72%霜脲·锰锌"
            "600-800倍液或50%烯酰吗啉1500-2000倍液喷雾，每7-10天一次。"
        ),
        "source": "果树病虫害防治手册",
    },
    {
        "text": (
            "葡萄白粉病（Erysiphe necator）防治：展叶至开花前用25%乙嘧酚磺酸酯"
            "微乳剂800-1000倍液或15%三唑酮可湿性粉剂1000倍液喷雾，每10天一次。"
            "注意控制枝蔓密度，保持通风透光。安全间隔期14天。"
        ),
        "source": "果树病虫害防治手册",
    },
    # ── 柑橘 Citrus ──────────────────────────────────────────────
    {
        "text": (
            "柑橘溃疡病（Xanthomonas citri）防治：春梢萌发至展叶期用77%氢氧化铜"
            "可湿性粉剂600倍液或20%噻菌铜悬浮剂500倍液喷雾，每次新梢萌发后"
            "连喷2-3次。台风暴雨后24小时内必须喷药防护。严格检疫，清除病枝病叶。"
        ),
        "source": "果树病虫害防治手册",
    },
    {
        "text": (
            "柑橘炭疽病（Colletotrichum gloeosporioides）防治：春梢和秋梢抽发期"
            "用70%甲基硫菌灵可湿性粉剂800倍液或25%咪鲜胺乳油1000倍液喷雾。"
            "冬季清园剪除病枝枯枝并集中烧毁。增施有机肥和磷钾肥以增强树势。"
        ),
        "source": "果树病虫害防治手册",
    },
    # ── 黄瓜 Cucumber ────────────────────────────────────────────
    {
        "text": (
            "黄瓜霜霉病（Pseudoperonospora cubensis）防治：发病初期用72%霜脲·锰锌"
            "600-800倍液或52.5%噁酮·霜脲氰水分散粒剂1500倍液喷雾，每5-7天一次，"
            "连续3-4次。棚室栽培注意控温降湿，及时通风换气。安全间隔期7天。"
        ),
        "source": "蔬菜病虫害防治指南",
    },
    {
        "text": (
            "黄瓜白粉病（Podosphaera xanthii）防治：发病初期用15%三唑酮可湿性粉剂"
            "1000-1500倍液或2%农抗120水剂200倍液喷雾，每7-10天一次。棚室可采用"
            "硫磺熏蒸器夜间熏蒸，每亩每次用硫磺粉250克。"
        ),
        "source": "蔬菜病虫害防治指南",
    },
    # ── 辣椒 Pepper ──────────────────────────────────────────────
    {
        "text": (
            "辣椒疫病（Phytophthora capsici）防治：发病初期用72%霜脲·锰锌可湿性"
            "粉剂600倍液或50%烯酰吗啉水分散粒剂1500倍液灌根和喷雾并用。注意"
            "雨后排水，避免大水漫灌，与非茄科作物轮作3年以上。安全间隔期10天。"
        ),
        "source": "蔬菜病虫害防治指南",
    },
    # ── 草莓 Strawberry ──────────────────────────────────────────
    {
        "text": (
            "草莓灰霉病（Botrytis cinerea）防治：花期用50%腐霉利可湿性粉剂"
            "1000-1500倍液或40%嘧霉胺悬浮剂800倍液喷雾，重点喷花序和幼果。"
            "保持棚内通风降湿，地面铺设地膜防止果实接触土壤。安全间隔期7天。"
        ),
        "source": "蔬菜病虫害防治指南",
    },
]


class KnowledgeIndexer:
    """Indexes agricultural knowledge documents into ChromaDB.

    Supports plain text files, PDFs, and bulk directory indexing, as well as
    built-in seed knowledge for out-of-the-box operation.
    """

    def __init__(self, persist_dir: str = "data/chromadb/knowledge") -> None:
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="agri_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API — file-level indexing
    # ------------------------------------------------------------------

    def index_text_file(self, filepath: str, source: str | None = None) -> int:
        """Index a plain text file, splitting into overlapping chunks.

        Returns the number of chunks indexed.
        """
        path = Path(filepath)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._index_chunks(text, source=source or str(path))

    def index_pdf(self, filepath: str, source: str | None = None) -> int:
        """Index a PDF file (extract text per page, split into chunks).

        Requires ``pdfplumber`` — raises *RuntimeError* if not installed.
        Returns the number of chunks indexed.
        """
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is required for PDF indexing — pip install pdfplumber"
            )

        pages_text: list[str] = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        return self._index_chunks(full_text, source=source or filepath)

    def index_directory(self, dirpath: str, source: str | None = None) -> int:
        """Index all ``.txt``, ``.md``, and ``.pdf`` files in *dirpath* (recursive).

        Returns total number of chunks indexed.
        """
        count = 0
        for root, _dirs, files in os.walk(dirpath):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                file_source = source or fpath
                lower = fname.lower()
                try:
                    if lower.endswith((".txt", ".md")):
                        count += self.index_text_file(fpath, source=file_source)
                    elif lower.endswith(".pdf"):
                        count += self.index_pdf(fpath, source=file_source)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to index %s, skipping", fpath)
        return count

    # Backward-compat alias used by existing tests
    index_documents = index_directory

    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self.collection.count()
        return {
            "collection": self.collection.name,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
        }

    # ------------------------------------------------------------------
    # Public API — text-level indexing
    # ------------------------------------------------------------------

    def index_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        """Index a single text document. Returns number of chunks created."""
        return self._index_chunks(text, source=source, extra_meta=metadata)

    def seed_default_knowledge(self) -> int:
        """Insert the built-in seed knowledge entries.

        Skips entries that are already present (idempotent).
        Returns the number of newly-added entries.
        """
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for entry in DEFAULT_KNOWLEDGE:
            doc_id = self._stable_id(entry["text"])
            # Skip if already exists
            existing = self.collection.get(ids=[doc_id])
            if existing["ids"]:
                continue
            ids.append(doc_id)
            documents.append(entry["text"])
            metadatas.append({"source": entry["source"]})

        if ids:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int = 500, overlap: int = 100
    ) -> list[str]:
        """Split *text* into overlapping chunks (~500 chars, 100 overlap)."""
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _index_chunks(
        self,
        text: str,
        source: str,
        extra_meta: dict | None = None,
    ) -> int:
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        ids: list[str] = []
        metadatas: list[dict] = []
        for idx, chunk in enumerate(chunks):
            meta: dict = {"source": source, "chunk_index": idx}
            if extra_meta:
                meta.update(extra_meta)
            ids.append(self._stable_id(chunk))
            metadatas.append(meta)

        self.collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    @staticmethod
    def _stable_id(text: str) -> str:
        """Deterministic chunk ID from content hash — guarantees idempotency."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
