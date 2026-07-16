"""ChromaDB vector store — embeddings only.

Chronological listing and system-of-record fields live in SQLite
(``sqlite_store.SQLiteStore``). This module must not be treated as SoR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from ..models.episode import Episode


def _serialize_concepts(concepts: List[str]) -> str:
    return "|".join(c.strip() for c in concepts if c and c.strip())


def _deserialize_concepts(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [p for p in str(raw).split("|") if p]


class ChromaMemoryStore:
    """Local persistent vector index for episode embeddings."""

    def __init__(self, persist_directory: str = "./grmc_data/chroma"):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="grmc_episodes",
            metadata={"hnsw:space": "cosine"},
        )

    def add_episode(self, episode: Episode, embedding: List[float]) -> str:
        """Index an episode embedding (idempotent replace if id exists)."""
        meta = {
            "timestamp": episode.timestamp.isoformat(),
            "source": episode.source,
            "conversation_id": episode.conversation_id or "",
            "importance_score": float(episode.importance_score),
            "extracted_concepts": _serialize_concepts(episode.extracted_concepts),
        }
        # upsert-like: delete then add for re-ingest safety
        try:
            existing = self.collection.get(ids=[episode.episode_id])
            if existing and existing.get("ids"):
                self.collection.delete(ids=[episode.episode_id])
        except Exception:
            pass

        self.collection.add(
            ids=[episode.episode_id],
            embeddings=[embedding],
            documents=[episode.content_summary],
            metadatas=[meta],
        )
        return episode.episode_id

    def query(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic search for relevant episode ids + summaries."""
        if self.count() == 0:
            return []

        n = min(top_k, self.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        episodes: List[Dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return episodes

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] or {}
            episodes.append(
                {
                    "episode_id": results["ids"][0][i],
                    "summary": results["documents"][0][i],
                    "metadata": meta,
                    "distance": results["distances"][0][i],
                    "extracted_concepts": _deserialize_concepts(
                        meta.get("extracted_concepts")
                    ),
                }
            )
        return episodes

    def count(self) -> int:
        return self.collection.count()

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(
            ids=[episode_id],
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return None
        meta = (results["metadatas"] or [None])[0] or {}
        doc = (results["documents"] or [None])[0] or ""
        return {
            "episode_id": results["ids"][0],
            "summary": doc,
            "metadata": meta,
            "extracted_concepts": _deserialize_concepts(meta.get("extracted_concepts")),
        }
