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

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "stats": stats,
        "recent_episodes": [
            {
                "episode_id": e["episode_id"],
                "timestamp": e.get("timestamp"),
                "source": e.get("source"),
                "summary": (e.get("summary") or "")[:200],
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
            }
            for n in nodes
        ],
        "graph_edges": [
            {
                "edge_id": e.edge_id,
                "type": e.edge_type,
                "source": e.source_node_id,
                "target": e.target_node_id,
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
            }
            for p in pending
        ],
        "recent_reflections": reflections,
    }


def dump_markdown(sqlite: SQLiteStore, *, recent_episodes: int = 15) -> str:
    data = build_dump(sqlite, recent_episodes=recent_episodes)
    lines: List[str] = []
    lines.append("# GRMC Memory Dump")
    lines.append("")
    lines.append(f"_Generated: {data['generated_at']}_")
    lines.append("")
    st = data["stats"]
    lines.append("## Stats")
    lines.append(
        f"- episodes: **{st.get('episodes')}** · nodes: **{st.get('graph_nodes')}** · "
        f"edges: **{st.get('graph_edges')}** · provenance links: **{st.get('episode_node_links')}**"
    )
    lines.append(
        f"- pending proposals: **{st.get('proposals_pending')}** · "
        f"reflections: **{st.get('reflections')}**"
    )
    lines.append("")
    lines.append("## Recent episodes")
    for e in data["recent_episodes"]:
        lines.append(
            f"- `{e['episode_id']}` ({e.get('timestamp', '')[:19]}) "
            f"— {e['summary']}"
        )
    lines.append("")
    lines.append("## Graph nodes")
    if not data["graph_nodes"]:
        lines.append("_None yet — approve concept proposals first._")
    for n in data["graph_nodes"]:
        lines.append(
            f"- `{n['node_id']}` **{n['label']}** "
            f"(type={n['type']}, conf={n['confidence']:.2f}, "
            f"eps={len(n['supporting_episodes'])})"
        )
    lines.append("")
    lines.append("## Graph edges")
    if not data["graph_edges"]:
        lines.append("_None yet — approve edge proposals first._")
    for e in data["graph_edges"]:
        lines.append(
            f"- `{e['edge_id']}` {e['source']} -[{e['type']}]-> {e['target']} "
            f"(conf={e['confidence']:.2f})"
        )
    lines.append("")
    lines.append("## Pending proposals")
    if not data["pending_proposals"]:
        lines.append("_Queue empty._")
    for p in data["pending_proposals"]:
        lines.append(
            f"- `{p['proposal_id']}` [{p['kind']}] {p['label']} "
            f"(conf={p['confidence']:.2f})"
        )
    lines.append("")
    lines.append("---")
    lines.append("_Dump is read-only. Graph writes require `grmc approve`._")
    lines.append("")
    return "\n".join(lines)
