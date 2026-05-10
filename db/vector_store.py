"""
db/vector_store.py
Thin wrapper around ChromaDB. All vector operations go through here
so the rest of the codebase never imports chromadb directly.
"""

from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

from config import settings


class VectorStore:
    """
    Wraps ChromaDB for semantic memory.

    Usage:
        vs = VectorStore()
        vs.upsert(entry)          # add or update an entry's embedding
        vs.search("cozy films")   # returns list of entry IDs + distances
        vs.delete(entry_id)       # remove from vector store
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(self, embedding_id: str, document: str, metadata: dict) -> None:
        """
        Add or update a document. Calling upsert with the same embedding_id
        overwrites the previous entry — safe to call after enrichment updates.
        """
        self._collection.upsert(
            ids=[embedding_id],
            documents=[document],
            metadatas=[metadata],
        )

    def delete(self, embedding_id: str) -> None:
        self._collection.delete(ids=[embedding_id])

    # ── Read ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts:
            [{"id": str, "document": str, "metadata": dict, "distance": float}]

        `where` lets you filter by metadata fields before the vector search, e.g.:
            where={"entry_type": "movie"}
            where={"event_date": {"$gte": "2024-01-01"}}
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        return self._collection.count()

    # ── Helpers ────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        return {
            "collection": settings.chroma_collection_name,
            "total_documents": self.count(),
            "persist_dir": settings.chroma_persist_dir,
        }


# Singleton — import this everywhere instead of instantiating directly
vector_store = VectorStore()
