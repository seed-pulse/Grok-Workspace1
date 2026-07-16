"""Memory Manager — dual backend: SQLite (SoR) + Chroma (vectors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.episode import Episode
from ..models.reflection_report import ReflectionReport
from ..storage.chroma_store import ChromaMemoryStore
from ..storage.sqlite_store import SQLiteStore
from .approval_queue import ApprovalQueue
from .embedder import Embedder, create_embedder


class MemoryManager:
    """Ingestion, retrieval, reflection, and approval entrypoints."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        vector_store: ChromaMemoryStore,
        embedder: Optional[Embedder] = None,
        embedder_prefer: str = "auto",
        data_dir: Optional[str | Path] = None,
    ):
        self.sqlite = sqlite
        self.vector_store = vector_store
        # Back-compat alias used by ReflectionEngine (vector store path/dir)
        self.store = vector_store
        self.embedder: Embedder = embedder or create_embedder(prefer=embedder_prefer)
        self.data_dir = Path(data_dir) if data_dir else Path(vector_store.persist_dir)
        self.approval = ApprovalQueue(sqlite)
        self._reflection_engine = None

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path = "./grmc_data",
        embedder_prefer: str = "auto",
        embedder: Optional[Embedder] = None,
    ) -> "MemoryManager":
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        sqlite = SQLiteStore(data_path / "grmc.db")
        # Chroma lives under data_dir/chroma to keep files tidy
        vector = ChromaMemoryStore(persist_directory=str(data_path / "chroma"))
        return cls(
            sqlite=sqlite,
            vector_store=vector,
            embedder=embedder,
            embedder_prefer=embedder_prefer,
            data_dir=data_path,
        )

    @property
    def reflection_engine(self):
        if self._reflection_engine is None:
            from ..reflection.reflection_engine import ReflectionEngine

            report_dir = self.data_dir / "reflections"
            self._reflection_engine = ReflectionEngine(
                self, report_dir=str(report_dir)
            )
        return self._reflection_engine

    def ingest_episode(self, episode: Episode) -> str:
        """Write episode to SQLite (SoR) and embedding to Chroma (search)."""
        if not episode.embedding:
            text = episode.content_summary or episode.raw_content or ""
            episode.embedding = self.embedder.encode(text)

        self.sqlite.add_episode(episode)
        self.vector_store.add_episode(episode, episode.embedding)
        return episode.episode_id

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search via Chroma; enrich from SQLite when possible."""
        query_embedding = self.embedder.encode(query)
        hits = self.vector_store.query(query_embedding, top_k=top_k)
        enriched: List[Dict[str, Any]] = []
        for hit in hits:
            row = self.sqlite.get_episode(hit["episode_id"])
            if row:
                row = dict(row)
                row["distance"] = hit.get("distance")
                enriched.append(row)
            else:
                enriched.append(hit)
        return enriched

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Indexed chronological list from SQLite."""
        return self.sqlite.list_recent(limit=limit)

    def count_episodes(self) -> int:
        return self.sqlite.count_episodes()

    def reflect(
        self,
        recent_limit: int = 30,
        topic: Optional[str] = None,
        persist: bool = True,
        enqueue_proposals: bool = True,
    ) -> ReflectionReport:
        """Report-only reflection; optionally enqueue approval proposals.

        Graph is never written here. Proposals are pending suggestions only.
        """
        report = self.reflection_engine.reflect(
            recent_limit=recent_limit,
            topic=topic,
            persist=persist,
        )
        # Persist structured report in SQLite for history / provenance
        self.sqlite.save_reflection_report(report)

        if enqueue_proposals and report.concept_candidates:
            created = self.approval.enqueue_from_report(report)
            report.metadata["proposals_enqueued"] = len(created)
            report.metadata["proposal_ids"] = [p.proposal_id for p in created]
            if created:
                report.notes.append(
                    f"Enqueued {len(created)} pending proposal(s) for human review "
                    f"(`grmc propose`). Graph still unchanged."
                )
                report.suggested_actions.insert(
                    0,
                    "Review pending proposals with `grmc propose`, then "
                    "`grmc approve <id>` or `grmc reject <id>`.",
                )
        return report
