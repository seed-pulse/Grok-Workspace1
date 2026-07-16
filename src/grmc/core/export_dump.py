"""Human-readable memory dump (read-only overview)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ..storage.sqlite_store import SQLiteStore


def build_dump(sqlite: SQLiteStore, *, recent_episodes: int = 20) -> Dict[str, Any]:
    stats = sqlite.stats()
    episodes = sqlite.list_recent(limit=recent_episodes)
    nodes = sqlite.list_graph_nodes(limit=100)
    edges = sqlite.list_graph_edges(limit=100)
    pending = sqlite.list_proposals(status="pending", limit=50)
    reflections = sqlite.list_reflection_reports(limit=10)

    label_by_id = {n.node_id: n.label for n in nodes}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "version_note": "GRMC dump — read-only overview; graph writes require approve",
        "stats": stats,
        "recent_episodes": [
            {
                "episode_id": e["episode_id"],
                "timestamp": e.get("timestamp"),
                "source": e.get("source"),
                "summary": (e.get("summary") or "")[:240],
                "concepts": e.get("extracted_concepts") or [],
            }
            for e in episodes
        ],
        "graph_nodes": [
            {
                "node_id": n.node_id,
                "label": n.label,
                "type": n.type,
                "confidence": n.confidence,
                "supporting_episodes": n.supporting_episodes,
                "provenance_link_count": len(sqlite.list_links_for_node(n.node_id)),
            }
            for n in nodes
        ],
        "graph_edges": [
            {
                "edge_id": e.edge_id,
                "type": e.edge_type,
                "source": e.source_node_id,
                "target": e.target_node_id,
                "source_label": label_by_id.get(e.source_node_id, e.source_node_id),
                "target_label": label_by_id.get(e.target_node_id, e.target_node_id),
                "confidence": e.confidence,
            }
            for e in edges
        ],
        "pending_proposals": [
            {
                "proposal_id": p.proposal_id,
                "kind": p.kind,
                "label": p.label,
                "confidence": p.confidence,
                "source": p.source,
            }
            for p in pending
        ],
        "recent_reflections": reflections,
    }


def dump_markdown(sqlite: SQLiteStore, *, recent_episodes: int = 15) -> str:
    data = build_dump(sqlite, recent_episodes=recent_episodes)
    st = data["stats"]
    lines: List[str] = []
    lines.append("# GRMC Memory Dump")
    lines.append("")
    lines.append(f"Generated: `{data['generated_at']}`")
    lines.append("")
    lines.append(
        "> Read-only snapshot. Reflection never writes the graph; "
        "only `grmc approve` creates nodes/edges."
    )
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Episodes | {st.get('episodes', 0)} |")
    lines.append(f"| Graph nodes | {st.get('graph_nodes', 0)} |")
    lines.append(f"| Graph edges | {st.get('graph_edges', 0)} |")
    lines.append(f"| Provenance links | {st.get('episode_node_links', 0)} |")
    lines.append(f"| Pending proposals | {st.get('proposals_pending', 0)} |")
    lines.append(f"| Reflection reports | {st.get('reflections', 0)} |")
    lines.append(f"| Schema | v{st.get('schema_version', '?')} |")
    lines.append("")

    lines.append("## Recent episodes")
    lines.append("")
    if not data["recent_episodes"]:
        lines.append("_None — try `grmc ingest -t \"...\"`._")
        lines.append("")
    for e in data["recent_episodes"]:
        concepts = ", ".join(e.get("concepts") or []) or "—"
        lines.append(
            f"- `{e['episode_id']}` · {str(e.get('timestamp') or '')[:19]} · "
            f"{e.get('source')}"
        )
        lines.append(f"  - {e['summary']}")
        lines.append(f"  - concepts: {concepts}")
    if data["recent_episodes"]:
        lines.append("")

    lines.append("## Graph nodes")
    lines.append("")
    if not data["graph_nodes"]:
        lines.append("_None — `grmc reflect` then `grmc approve` on a concept proposal._")
        lines.append("")
    for n in data["graph_nodes"]:
        lines.append(
            f"- **{n['label']}** (`{n['node_id']}`) — "
            f"{n['type']}, conf {n['confidence']:.2f}, "
            f"{len(n['supporting_episodes'])} support eps, "
            f"{n.get('provenance_link_count', 0)} provenance links"
        )
    if data["graph_nodes"]:
        lines.append("")

    lines.append("## Graph edges")
    lines.append("")
    if not data["graph_edges"]:
        lines.append("_None — `grmc edges propose` then `approve`._")
        lines.append("")
    for e in data["graph_edges"]:
        lines.append(
            f"- **{e['source_label']}** `--[{e['type']}]-->` **{e['target_label']}** "
            f"(conf {e['confidence']:.2f}, `{e['edge_id']}`)"
        )
    if data["graph_edges"]:
        lines.append("")

    lines.append("## Pending proposals")
    lines.append("")
    if not data["pending_proposals"]:
        lines.append("_Queue empty._")
        lines.append("")
    else:
        for p in data["pending_proposals"]:
            lines.append(
                f"- `{p['proposal_id']}` · {p['kind']} · {p['label']} · "
                f"conf {p['confidence']:.2f}"
            )
        lines.append("")
        lines.append("Review: `grmc propose` → `approve` / `reject`.")
        lines.append("")

    if data.get("recent_reflections"):
        lines.append("## Recent reflections")
        lines.append("")
        for r in data["recent_reflections"][:5]:
            lines.append(
                f"- `{r.get('report_id')}` · {str(r.get('timestamp') or '')[:19]} · "
                f"{r.get('mode')} · {r.get('episodes_analyzed')} eps · "
                f"mutates={bool(r.get('mutates_memory'))}"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Docs: `docs/QUICKSTART.md` · `docs/DESIGN_PRINCIPLES.md` · `docs/HANDOVER.md`"
    )
    lines.append("")
    return "\n".join(lines)
