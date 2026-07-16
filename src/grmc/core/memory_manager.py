"""Phase 0/1 Memory Manager — ingestion, retrieval, and reflection entrypoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models.episode import Episode
from ..models.reflection_report import ReflectionReport
from ..storage.chroma_store import ChromaMemoryStore
from .embedder import Embedder, create_embedder


class MemoryManager:
    """Handles episode ingestion, semantic retrieval, and reflection access."""

    def __init__(
        self,
        store: ChromaMemoryStore,
        embedder: Optional[Embedder] = None,
        embedder_prefer: str = "auto",
    ):
        self.store = store
        self.embedder: Embedder = embedder or create_embedder(prefer=embedder_prefer)
        self._reflection_engine = None

    @property
    def reflection_engine(self):
        """Lazy-init to keep import light when only ingesting."""
        if self._reflection_engine is None:
            from ..reflection.reflection_engine import ReflectionEngine

            self._reflection_engine = ReflectionEngine(self)
        return self._reflection_engine

    def ingest_episode(self, episode: Episode) -> str:
        """Create embedding and store the episode."""
        if not episode.embedding:
            text = episode.content_summary or episode.raw_content or ""
            episode.embedding = self.embedder.encode(text)

        return self.store.add_episode(episode, episode.embedding)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embed query and retrieve relevant episodes."""
        query_embedding = self.embedder.encode(query)
        return self.store.query(query_embedding, top_k=top_k)

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent episodes (client-side timestamp sort)."""
        return self.store.list_recent(limit=limit)

    def reflect(
        self,
        recent_limit: int = 30,
        topic: Optional[str] = None,
        persist: bool = True,
    ) -> ReflectionReport:
        """Run a conservative, report-only reflection pass."""
        return self.reflection_engine.reflect(
            recent_limit=recent_limit,
            topic=topic,
            persist=persist,
        )
