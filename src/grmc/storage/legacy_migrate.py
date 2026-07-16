"""Migrate pre-0.3 Chroma-root episode stores into SQLite SoR.

Safe, additive: never deletes Chroma data; skips ids already in SQLite.
Does not create graph nodes or edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.episode import Episode
from .chroma_store import ChromaMemoryStore, _deserialize_concepts
from .sqlite_store import SQLiteStore


@dataclass
class MigrateResult:
    source: str
    scanned: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "scanned": self.scanned,
            "inserted": self.inserted,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
        }


def _chroma_dump(store: ChromaMemoryStore) -> List[Dict[str, Any]]:
    total = store.count()
    if total == 0:
        return []
    results = store.collection.get(include=["documents", "metadatas"])
    out: List[Dict[str, Any]] = []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for i, eid in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        doc = docs[i] if i < len(docs) else ""
        out.append(
            {
                "episode_id": eid,
                "summary": doc or "",
                "metadata": meta or {},
                "extracted_concepts": _deserialize_concepts(
                    (meta or {}).get("extracted_concepts")
                ),
            }
        )
    return out


def migrate_chroma_to_sqlite(
    data_dir: str | Path,
    *,
    legacy_chroma_path: Optional[str | Path] = None,
) -> MigrateResult:
    """Import episodes from a Chroma collection into SQLite.

    Default legacy path: ``{data_dir}`` itself (pre-0.3 layout where Chroma
    lived at the data root). New layout uses ``{data_dir}/chroma``.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    sqlite = SQLiteStore(data_path / "grmc.db")

    src = Path(legacy_chroma_path) if legacy_chroma_path else data_path
    result = MigrateResult(source=str(src.resolve()))

    # Detect whether this looks like a chroma persist dir
    if not src.exists():
        result.errors.append(f"Path does not exist: {src}")
        return result

    try:
        chroma = ChromaMemoryStore(persist_directory=str(src))
    except Exception as exc:
        result.errors.append(f"Failed to open Chroma at {src}: {exc}")
        return result

    try:
        rows = _chroma_dump(chroma)
    except Exception as exc:
        result.errors.append(f"Failed to read Chroma collection: {exc}")
        return result

    result.scanned = len(rows)
    for row in rows:
        eid = row["episode_id"]
        if sqlite.get_episode(eid):
            result.skipped_existing += 1
            continue
        meta = row.get("metadata") or {}
        ts_raw = meta.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.utcnow()
        except ValueError:
            ts = datetime.utcnow()
        try:
            imp = float(meta.get("importance_score", 0.5))
        except (TypeError, ValueError):
            imp = 0.5
        ep = Episode(
            episode_id=eid,
            timestamp=ts,
            conversation_id=meta.get("conversation_id") or None,
            source=str(meta.get("source") or "legacy-chroma"),
            content_summary=row.get("summary") or "",
            raw_content=row.get("summary") or "",
            extracted_concepts=row.get("extracted_concepts") or [],
            importance_score=max(0.0, min(1.0, imp)),
            metadata={"migrated_from": str(src), **{k: v for k, v in meta.items()}},
        )
        try:
            sqlite.add_episode(ep)
            result.inserted += 1
        except Exception as exc:
            result.errors.append(f"{eid}: {exc}")

    return result
