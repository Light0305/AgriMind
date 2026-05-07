"""Retrieve relevant agricultural knowledge from the ChromaDB index."""

from __future__ import annotations

import chromadb

from app.schemas import RetrievedDocument


class KnowledgeRetriever:
    """Retrieves relevant agricultural knowledge for diagnosis support."""

    def __init__(self, persist_dir: str = "data/chromadb/knowledge") -> None:
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("agri_knowledge")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Return the *top_k* most relevant knowledge chunks for *query*.

        Returns a list of ``RetrievedDocument`` instances with ``content``,
        ``source``, and ``score`` fields.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )

        formatted: list[RetrievedDocument] = []
        if not results["ids"] or not results["ids"][0]:
            return formatted

        for idx, _doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            # ChromaDB returns *distance*; convert to similarity for cosine.
            score = 1.0 - distance
            formatted.append(
                RetrievedDocument(
                    content=results["documents"][0][idx],
                    source=results["metadatas"][0][idx].get("source", ""),
                    score=round(score, 4),
                )
            )
        return formatted

    def format_context(self, docs: list[RetrievedDocument]) -> str:
        """Format retrieved docs into a context string for the LLM prompt."""
        if not docs:
            return ""
        return "\n\n".join(
            f"[参考资料 {i + 1}] ({doc.source})\n{doc.content}"
            for i, doc in enumerate(docs)
        )

    def get_treatment_context(self, diagnosis: str) -> str:
        """Build a knowledge context string for a given *diagnosis*.

        Fires multiple targeted queries and deduplicates results before
        assembling a reference block suitable for injection into agent prompts.
        """
        queries = [
            f"{diagnosis} 防治方法",
            f"{diagnosis} 推荐用药",
            f"{diagnosis} 发病规律",
        ]

        seen_texts: set[str] = set()
        unique_results: list[RetrievedDocument] = []
        for q in queries:
            for r in self.retrieve(q, top_k=2):
                if r.content not in seen_texts:
                    seen_texts.add(r.content)
                    unique_results.append(r)

        if not unique_results:
            return ""

        lines = ["【植保知识参考】"]
        for r in unique_results:
            lines.append(f"- {r.content}")
        return "\n".join(lines)
