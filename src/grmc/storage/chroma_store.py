"""ChromaDB-backed episode store (Phase 0 / early Phase 1).

Honest limitations of this layer:
- Chroma is optimized for vector search, not chronological listing.
- ``list_recent`` / ``get_all_episodes`` fetch candidates then sort client-side
  by the ``timestamp`` metadata field we write on ingest.
- Metadata values must be scalar (str/int/float/bool); lists are serialized.
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
    """Local persistent store for episode embeddings + metadata."""

    def __init__(self, persist_directory: str = "./grmc_data"):
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
        """Add an episode with its embedding."""
        self.collection.add(
            ids=[episode.episode_id],
            embeddings=[embedding],
            documents=[episode.content_summary],
            metadatas=[
                {
                    "timestamp": episode.timestamp.isoformat(),
                    "source": episode.source,
                    "conversation_id": episode.conversation_id or "",
                    "importance_score": float(episode.importance_score),
                    "extracted_concepts": _serialize_concepts(episode.extracted_concepts),
                }
            ],
        )
        return episode.episode_id

    def query(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic search for relevant episodes."""
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

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Return all stored episodes (unsorted). Used as base for list_recent."""
        total = self.count()
        if total == 0:
            return []

        results = self.collection.get(include=["documents", "metadatas"])
        episodes: List[Dict[str, Any]] = []
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        for i, episode_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            doc = documents[i] if i < len(documents) else ""
            episodes.append(
                {
                    "episode_id": episode_id,
                    "summary": doc or "",
                    "metadata": meta,
                    "extracted_concepts": _deserialize_concepts(
                        meta.get("extracted_concepts")
                    ),
                }
            )
        return episodes

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return episodes sorted by timestamp descending (client-side).

        Phase 0 honesty: Chroma has no native chronological index. We load
        stored metadata and sort by the ISO timestamp written at ingest.
        Episodes without a parseable timestamp sort last.
        """
        episodes = self.get_all_episodes()
        if not episodes:
            return []

        def sort_key(ep: Dict[str, Any]) -> str:
            meta = ep.get("metadata") or {}
            ts = meta.get("timestamp") or ""
            return str(ts)

        episodes.sort(key=sort_key, reverse=True)
        return episodes[: max(0, limit)]

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single episode by id, or None if missing."""
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
