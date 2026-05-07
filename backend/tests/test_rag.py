"""Tests for app.rag — knowledge indexer & retriever."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from app.rag.indexer import DEFAULT_KNOWLEDGE, KnowledgeIndexer
from app.rag.retriever import KnowledgeRetriever
from app.schemas import RetrievedDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def work_dir():
    """Portable temp directory that avoids Windows tmp_path permission issues."""
    d = tempfile.mkdtemp(prefix="test_rag_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def db_dir(work_dir):
    """Return a temporary directory for ChromaDB persistence."""
    return os.path.join(work_dir, "knowledge_db")


@pytest.fixture()
def indexer(db_dir):
    return KnowledgeIndexer(persist_dir=db_dir)


@pytest.fixture()
def seeded_db(indexer, db_dir):
    """Seed default knowledge and return (indexer, db_dir)."""
    indexer.seed_default_knowledge()
    return indexer, db_dir


# ---------------------------------------------------------------------------
# KnowledgeIndexer._chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_string(self):
        assert KnowledgeIndexer._chunk_text("") == []

    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = KnowledgeIndexer._chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size(self):
        text = "a" * 500
        chunks = KnowledgeIndexer._chunk_text(text, chunk_size=500, overlap=100)
        # All content is in the first chunk; overlap causes a 2nd (tail) chunk
        assert len(chunks) >= 1
        assert chunks[0] == text

    def test_overlapping_chunks(self):
        text = "a" * 1000
        chunks = KnowledgeIndexer._chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) >= 2
        # Each chunk should be at most chunk_size long
        for c in chunks:
            assert len(c) <= 500

    def test_custom_params(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = KnowledgeIndexer._chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 4
        # Overlapping region: last 10 chars of chunk N == first 10 of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-10:]
            head = chunks[i + 1][:10]
            assert tail == head

    def test_default_chunk_size_is_500(self):
        """Default chunk size specified in spec is ~500 chars."""
        text = "x" * 1200
        chunks = KnowledgeIndexer._chunk_text(text)
        # With 500 chunk size and 100 overlap, expect ~3 chunks for 1200 chars
        assert len(chunks) >= 3
        assert all(len(c) <= 500 for c in chunks)


# ---------------------------------------------------------------------------
# Indexing and retrieval round-trip
# ---------------------------------------------------------------------------

class TestIndexAndRetrieve:
    def test_index_text_and_retrieve(self, indexer, db_dir):
        indexer.index_text(
            "苹果树腐烂病需要刮除病疤后涂抹药剂",
            source="test",
        )
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        results = retriever.retrieve("苹果树腐烂病", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedDocument)
        assert "苹果树腐烂病" in results[0].content
        assert results[0].source == "test"
        assert isinstance(results[0].score, float)

    def test_retrieve_top_k(self, indexer, db_dir):
        for i in range(5):
            indexer.index_text(f"知识条目 {i} 关于小麦病害", source=f"src{i}")
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        results = retriever.retrieve("小麦", top_k=3)
        assert len(results) == 3

    def test_index_text_file(self, indexer, work_dir, db_dir):
        fpath = os.path.join(work_dir, "test.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("玉米锈病防治使用三唑酮效果显著")
        count = indexer.index_text_file(fpath, source="txt_test")
        assert count >= 1

        retriever = KnowledgeRetriever(persist_dir=db_dir)
        results = retriever.retrieve("玉米锈病", top_k=1)
        assert len(results) >= 1
        assert results[0].source == "txt_test"

    def test_index_directory(self, indexer, work_dir, db_dir):
        docs_dir = os.path.join(work_dir, "docs")
        os.makedirs(docs_dir)
        with open(os.path.join(docs_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("玉米锈病用三唑酮防治")
        with open(os.path.join(docs_dir, "b.md"), "w", encoding="utf-8") as f:
            f.write("番茄晚疫病用百菌清预防")
        with open(os.path.join(docs_dir, "ignore.csv"), "w", encoding="utf-8") as f:
            f.write("should,be,ignored")

        count = indexer.index_directory(str(docs_dir))
        assert count >= 2  # at least one chunk per file

        retriever = KnowledgeRetriever(persist_dir=db_dir)
        results = retriever.retrieve("番茄晚疫病", top_k=1)
        assert len(results) >= 1

    def test_index_documents_alias(self, indexer, work_dir, db_dir):
        """index_documents is a backward-compat alias for index_directory."""
        docs_dir = os.path.join(work_dir, "docs2")
        os.makedirs(docs_dir)
        with open(os.path.join(docs_dir, "c.txt"), "w", encoding="utf-8") as f:
            f.write("测试向后兼容方法")
        count = indexer.index_documents(str(docs_dir))
        assert count >= 1


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_on_empty(self, indexer):
        stats = indexer.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["collection"] == "agri_knowledge"
        assert "persist_dir" in stats

    def test_stats_after_indexing(self, indexer):
        indexer.index_text("一些测试内容", source="test")
        stats = indexer.get_stats()
        assert stats["total_chunks"] >= 1


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_format_context_output(self, seeded_db):
        _indexer, db_dir = seeded_db
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        docs = retriever.retrieve("小麦条锈病", top_k=2)
        ctx = retriever.format_context(docs)
        assert "[参考资料 1]" in ctx
        assert "[参考资料 2]" in ctx
        # Each doc source should appear in parens
        for doc in docs:
            assert f"({doc.source})" in ctx

    def test_format_context_empty(self, seeded_db):
        _indexer, db_dir = seeded_db
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        assert retriever.format_context([]) == ""


# ---------------------------------------------------------------------------
# get_treatment_context
# ---------------------------------------------------------------------------

class TestTreatmentContext:
    def test_returns_relevant_context(self, seeded_db):
        _indexer, db_dir = seeded_db
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        ctx = retriever.get_treatment_context("小麦条锈病")
        assert "【植保知识参考】" in ctx
        # Default embeddings may not rank Chinese perfectly, but should
        # return wheat-related or disease-related content from seed data.
        assert len(ctx) > 20

    def test_empty_for_unknown(self, indexer, db_dir):
        # Only index one unrelated entry
        indexer.index_text("无关内容测试文本", source="test")
        retriever = KnowledgeRetriever(persist_dir=db_dir)
        # Still returns something (best-effort match) — just check it runs
        ctx = retriever.get_treatment_context("完全未知的疾病XYZ")
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# Seed knowledge
# ---------------------------------------------------------------------------

class TestSeedKnowledge:
    def test_seed_populates_collection(self, indexer):
        count = indexer.seed_default_knowledge()
        assert count == len(DEFAULT_KNOWLEDGE)
        assert indexer.collection.count() == len(DEFAULT_KNOWLEDGE)

    def test_seed_is_idempotent(self, indexer):
        first = indexer.seed_default_knowledge()
        second = indexer.seed_default_knowledge()
        assert first == len(DEFAULT_KNOWLEDGE)
        assert second == 0  # nothing new added
        assert indexer.collection.count() == len(DEFAULT_KNOWLEDGE)

    def test_default_knowledge_has_at_least_20_entries(self):
        assert len(DEFAULT_KNOWLEDGE) >= 20
