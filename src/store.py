from __future__ import annotations

import re
from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        prefer_chroma: bool = True,
        persist_directory: str = ".chroma",
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._client = None
        self._next_index = 0
        self._backend_name = "in-memory"

        if not prefer_chroma:
            return

        try:
            import chromadb

            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
            self._backend_name = "ChromaDB"
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._client = None
            self._backend_name = "in-memory"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)

        record = {
            "id": f"{doc.id}:{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": None,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        stop_words = {
            "a", "an", "and", "are", "did", "does", "for", "how", "in", "is", "of",
            "on", "or", "the", "to", "was", "what", "when", "which", "why",
        }

        query_embedding = self._embedding_fn(query)
        query_tokens = {
            token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if token not in stop_words
        }

        scored = []
        for record in records:
            embedding = record.get("embedding")
            if embedding is None:
                continue

            semantic_score = compute_similarity(query_embedding, embedding)
            content_tokens = {
                token for token in re.findall(r"[a-zA-Z0-9]+", record["content"].lower()) if token not in stop_words
            }
            lexical_overlap = len(query_tokens & content_tokens)
            lexical_score = (lexical_overlap / len(query_tokens)) if query_tokens else 0.0
            total_score = float(semantic_score + lexical_score)

            scored.append(
                {
                    "id": record["doc_id"],
                    "content": record["content"],
                    "metadata": dict(record.get("metadata") or {}),
                    "score": total_score,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _query_chroma_candidate_ids(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[str]:
        if top_k <= 0 or not self._use_chroma or self._collection is None:
            return []

        try:
            query_embedding = self._embedding_fn(query)
            n_results = min(max(top_k * 3, top_k), len(self._store)) if self._store else top_k
            if n_results <= 0:
                return []

            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }
            if metadata_filter:
                query_kwargs["where"] = metadata_filter

            result = self._collection.query(**query_kwargs)
            ids = result.get("ids", [[]])
            return list(ids[0]) if ids else []
        except Exception:
            return []

    def _search_records_by_ids(self, query: str, candidate_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []

        id_set = set(candidate_ids)
        candidate_records = [record for record in self._store if record["id"] in id_set]
        return self._search_records(query, candidate_records, top_k)

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        chroma_ids: list[str] = []
        chroma_docs: list[str] = []
        chroma_embeddings: list[list[float]] = []
        chroma_metadatas: list[dict[str, Any]] = []

        for doc in docs:
            record = self._make_record(doc)
            record["embedding"] = self._embedding_fn(doc.content)
            self._store.append(record)

            if self._use_chroma and self._collection is not None:
                chroma_ids.append(record["id"])
                chroma_docs.append(record["content"])
                chroma_embeddings.append(record["embedding"])
                chroma_metadatas.append(record["metadata"])

        if chroma_ids and self._collection is not None:
            try:
                if hasattr(self._collection, "upsert"):
                    self._collection.upsert(
                        ids=chroma_ids,
                        documents=chroma_docs,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                    )
                else:
                    self._collection.add(
                        ids=chroma_ids,
                        documents=chroma_docs,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                    )
            except Exception:
                self._use_chroma = False
                self._backend_name = "in-memory"

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        If ChromaDB is enabled, use it to fetch vector candidates first, then
        re-rank those candidates with the local scoring logic for consistency.
        """
        chroma_candidate_ids = self._query_chroma_candidate_ids(query, top_k)
        if chroma_candidate_ids:
            reranked = self._search_records_by_ids(query, chroma_candidate_ids, top_k)
            if reranked:
                return reranked

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First try the vector backend with a metadata filter, then fall back to
        in-memory filtered search if needed.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        chroma_candidate_ids = self._query_chroma_candidate_ids(query, top_k, metadata_filter=metadata_filter)
        if chroma_candidate_ids:
            reranked = self._search_records_by_ids(query, chroma_candidate_ids, top_k)
            if reranked:
                return reranked

        filtered_records = [
            record
            for record in self._store
            if all((record.get("metadata") or {}).get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        ids_to_delete = [
            record["id"]
            for record in self._store
            if record.get("doc_id") == doc_id or (record.get("metadata") or {}).get("doc_id") == doc_id
        ]
        if not ids_to_delete:
            return False

        self._store = [record for record in self._store if record["id"] not in ids_to_delete]

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=ids_to_delete)
            except Exception:
                pass

        return True
