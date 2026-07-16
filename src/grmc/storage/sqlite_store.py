"""SQLite system of record for episodes, reflection history, proposals, graph.

ChromaDB remains vector-search only. Chronology, provenance, and approval
state live here with proper indexes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..models.episode import Episode
from ..models.graph_node import GraphNode
from ..models.proposal import Proposal
from ..models.reflection_report import ReflectionReport

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    conversation_id TEXT,
    source TEXT NOT NULL DEFAULT 'unknown',
    content_summary TEXT NOT NULL,
    raw_content TEXT,
    extracted_concepts TEXT NOT NULL DEFAULT '[]',
    importance_score REAL NOT NULL DEFAULT 0.5,
    linked_graph_nodes TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
    ON episodes(timestamp DESC);

CREATE TABLE IF NOT EXISTS reflection_reports (
    report_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    mode TEXT,
    topic TEXT,
    episodes_analyzed INTEGER NOT NULL DEFAULT 0,
    mutates_memory INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reflection_timestamp
    ON reflection_reports(timestamp DESC);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.35,
    source TEXT NOT NULL DEFAULT 'heuristic',
    report_id TEXT,
    supporting_episode_ids TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL DEFAULT '{}',
    reviewed_at TEXT,
    resulting_node_id TEXT,
    review_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status_created
    ON proposals(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_proposals_label
    ON proposals(label);

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    supporting_episodes TEXT NOT NULL DEFAULT '[]',
    contradicting_episodes TEXT NOT NULL DEFAULT '[]',
    last_reflected TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_label
    ON graph_nodes(label);
"""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _dt_iso(value: datetime | str | None) -> str:
    if value is None:
        return datetime.utcnow().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class SQLiteStore:
    """Local SQLite database for GRMC system-of-record data."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, timestamp, conversation_id, source,
                    content_summary, raw_content, extracted_concepts,
                    importance_score, linked_graph_nodes, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    _dt_iso(episode.timestamp),
                    episode.conversation_id,
                    episode.source,
                    episode.content_summary,
                    episode.raw_content,
                    _json_dumps(episode.extracted_concepts),
                    float(episode.importance_score),
                    _json_dumps(episode.linked_graph_nodes),
                    _json_dumps(episode.metadata),
                    datetime.utcnow().isoformat(),
                ),
            )
        return episode.episode_id

    def count_episodes(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
            return int(row["n"]) if row else 0

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return episodes ordered by timestamp DESC (indexed)."""
        limit = max(0, int(limit))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._episode_row_to_dict(r) for r in rows]

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        return self._episode_row_to_dict(row) if row else None

    def _episode_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        concepts = _json_loads(row["extracted_concepts"], [])
        meta = _json_loads(row["metadata"], {})
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta.setdefault("timestamp", row["timestamp"])
        meta.setdefault("source", row["source"])
        meta.setdefault("conversation_id", row["conversation_id"] or "")
        meta.setdefault("importance_score", row["importance_score"])
        return {
            "episode_id": row["episode_id"],
            "summary": row["content_summary"],
            "content_summary": row["content_summary"],
            "raw_content": row["raw_content"],
            "metadata": meta,
            "extracted_concepts": concepts if isinstance(concepts, list) else [],
            "linked_graph_nodes": _json_loads(row["linked_graph_nodes"], []),
            "timestamp": row["timestamp"],
            "source": row["source"],
        }

    # ------------------------------------------------------------------
    # Reflection history
    # ------------------------------------------------------------------

    def save_reflection_report(self, report: ReflectionReport) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reflection_reports (
                    report_id, timestamp, mode, topic, episodes_analyzed,
                    mutates_memory, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    _dt_iso(report.timestamp),
                    report.mode,
                    report.topic,
                    int(report.episodes_analyzed),
                    1 if report.mutates_memory else 0,
                    report.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )
        return report.report_id

    def list_reflection_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT report_id, timestamp, mode, topic, episodes_analyzed,
                       mutates_memory
                FROM reflection_reports
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(0, limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_reflection(self) -> Optional[Dict[str, Any]]:
        items = self.list_reflection_reports(limit=1)
        return items[0] if items else None

    # ------------------------------------------------------------------
    # Proposals (approval queue)
    # ------------------------------------------------------------------

    def add_proposal(self, proposal: Proposal) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO proposals (
                    proposal_id, created_at, status, kind, label, confidence,
                    source, report_id, supporting_episode_ids, payload_json,
                    reviewed_at, resulting_node_id, review_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    _dt_iso(proposal.created_at),
                    proposal.status,
                    proposal.kind,
                    proposal.label,
                    float(proposal.confidence),
                    proposal.source,
                    proposal.report_id,
                    _json_dumps(proposal.supporting_episode_ids),
                    _json_dumps(proposal.payload),
                    _dt_iso(proposal.reviewed_at) if proposal.reviewed_at else None,
                    proposal.resulting_node_id,
                    proposal.review_note,
                ),
            )
        return proposal.proposal_id

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def list_proposals(
        self,
        status: Optional[str] = "pending",
        limit: int = 50,
    ) -> List[Proposal]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM proposals
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, max(0, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM proposals
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (max(0, limit),),
                ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def update_proposal(self, proposal: Proposal) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE proposals SET
                    status = ?,
                    reviewed_at = ?,
                    resulting_node_id = ?,
                    review_note = ?,
                    confidence = ?,
                    payload_json = ?
                WHERE proposal_id = ?
                """,
                (
                    proposal.status,
                    _dt_iso(proposal.reviewed_at) if proposal.reviewed_at else None,
                    proposal.resulting_node_id,
                    proposal.review_note,
                    float(proposal.confidence),
                    _json_dumps(proposal.payload),
                    proposal.proposal_id,
                ),
            )

    def count_proposals(self, status: Optional[str] = None) -> int:
        with self._conn() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM proposals WHERE status = ?",
                    (status,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM proposals").fetchone()
            return int(row["n"]) if row else 0

    def find_pending_by_label(self, label: str) -> Optional[Proposal]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM proposals
                WHERE status = 'pending' AND lower(label) = lower(?)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (label,),
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def _row_to_proposal(self, row: sqlite3.Row) -> Proposal:
        return Proposal(
            proposal_id=row["proposal_id"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.utcnow(),
            status=row["status"],
            kind=row["kind"],
            label=row["label"],
            confidence=float(row["confidence"]),
            source=row["source"] or "heuristic",
            report_id=row["report_id"],
            supporting_episode_ids=_json_loads(row["supporting_episode_ids"], []),
            payload=_json_loads(row["payload_json"], {}),
            reviewed_at=datetime.fromisoformat(row["reviewed_at"])
            if row["reviewed_at"]
            else None,
            resulting_node_id=row["resulting_node_id"],
            review_note=row["review_note"],
        )

    # ------------------------------------------------------------------
    # Graph nodes (written only via approval)
    # ------------------------------------------------------------------

    def add_graph_node(self, node: GraphNode) -> str:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO graph_nodes (
                    node_id, type, label, confidence, supporting_episodes,
                    contradicting_episodes, last_reflected, version, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.node_id,
                    node.type,
                    node.label,
                    float(node.confidence),
                    _json_dumps(node.supporting_episodes),
                    _json_dumps(node.contradicting_episodes),
                    _dt_iso(node.last_reflected) if node.last_reflected else None,
                    int(node.version),
                    _json_dumps(node.metadata),
                    now,
                    now,
                ),
            )
        return node.node_id

    def get_graph_node(self, node_id: str) -> Optional[GraphNode]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM graph_nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        return self._row_to_node(row) if row else None

    def find_graph_node_by_label(
        self, label: str, node_type: str = "concept"
    ) -> Optional[GraphNode]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM graph_nodes
                WHERE lower(label) = lower(?) AND type = ?
                LIMIT 1
                """,
                (label, node_type),
            ).fetchone()
        return self._row_to_node(row) if row else None

    def list_graph_nodes(self, limit: int = 50) -> List[GraphNode]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM graph_nodes
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(0, limit),),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_graph_nodes(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()
            return int(row["n"]) if row else 0

    def update_graph_node(self, node: GraphNode) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE graph_nodes SET
                    type = ?,
                    label = ?,
                    confidence = ?,
                    supporting_episodes = ?,
                    contradicting_episodes = ?,
                    last_reflected = ?,
                    version = ?,
                    metadata = ?,
                    updated_at = ?
                WHERE node_id = ?
                """,
                (
                    node.type,
                    node.label,
                    float(node.confidence),
                    _json_dumps(node.supporting_episodes),
                    _json_dumps(node.contradicting_episodes),
                    _dt_iso(node.last_reflected) if node.last_reflected else None,
                    int(node.version),
                    _json_dumps(node.metadata),
                    datetime.utcnow().isoformat(),
                    node.node_id,
                ),
            )

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            node_id=row["node_id"],
            type=row["type"],  # type: ignore[arg-type]
            label=row["label"],
            confidence=float(row["confidence"]),
            supporting_episodes=_json_loads(row["supporting_episodes"], []),
            contradicting_episodes=_json_loads(row["contradicting_episodes"], []),
            last_reflected=datetime.fromisoformat(row["last_reflected"])
            if row["last_reflected"]
            else None,
            version=int(row["version"]),
            metadata=_json_loads(row["metadata"], {}),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "db_path": str(self.db_path.resolve()),
            "episodes": self.count_episodes(),
            "proposals_pending": self.count_proposals("pending"),
            "proposals_total": self.count_proposals(),
            "graph_nodes": self.count_graph_nodes(),
            "reflections": len(self.list_reflection_reports(limit=100000)),
        }
