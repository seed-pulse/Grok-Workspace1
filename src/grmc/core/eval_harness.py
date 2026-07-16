"""Minimal evaluation harness — catch over-confidence and provenance gaps.

Read-only against SQLite. Never mutates the graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..storage.sqlite_store import SQLiteStore

# Align with approval caps
NODE_CONF_SOFT_MAX = 0.55
EDGE_CONF_SOFT_MAX = 0.45


@dataclass
class EvalReport:
    ok: bool
    score: float  # 0..1 higher is healthier
    checks: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_eval(
    sqlite: SQLiteStore,
    *,
    fixture_mode: bool = False,
) -> EvalReport:
    nodes = sqlite.list_graph_nodes(limit=10000)
    edges = sqlite.list_graph_edges(limit=10000)
    reflections = sqlite.list_reflection_reports(limit=200)
    pending = sqlite.count_proposals("pending")
    stats = sqlite.stats()

    checks: List[Dict[str, Any]] = []
    recs: List[str] = []

    # 1) Provenance coverage for nodes
    with_links = 0
    for n in nodes:
        links = sqlite.list_links_for_node(n.node_id)
        if links or n.supporting_episodes:
            with_links += 1
    coverage = (with_links / len(nodes)) if nodes else 1.0
    checks.append(
        {
            "name": "provenance_coverage",
            "ok": coverage >= 0.8 or len(nodes) == 0,
            "value": round(coverage, 3),
            "detail": f"{with_links}/{len(nodes)} nodes have episode grounding",
        }
    )
    if nodes and coverage < 0.8:
        recs.append(
            "Some nodes lack provenance links — prefer approve paths that "
            "include supporting_episode_ids."
        )

    # 2) Over-confident nodes
    hot_nodes = [n for n in nodes if n.confidence > NODE_CONF_SOFT_MAX + 1e-9]
    checks.append(
        {
            "name": "node_confidence_cap",
            "ok": len(hot_nodes) == 0,
            "value": len(hot_nodes),
            "detail": f"nodes with conf > {NODE_CONF_SOFT_MAX}: {len(hot_nodes)}",
        }
    )
    if hot_nodes:
        recs.append(
            f"Found {len(hot_nodes)} node(s) above soft max {NODE_CONF_SOFT_MAX} "
            "— review manually; system defaults cap new approvals."
        )

    # 3) Over-confident edges
    hot_edges = [e for e in edges if e.confidence > EDGE_CONF_SOFT_MAX + 1e-9]
    checks.append(
        {
            "name": "edge_confidence_cap",
            "ok": len(hot_edges) == 0,
            "value": len(hot_edges),
            "detail": f"edges with conf > {EDGE_CONF_SOFT_MAX}: {len(hot_edges)}",
        }
    )

    # 4) Reflection reports must not claim mutation
    bad_refl = [r for r in reflections if r.get("mutates_memory")]
    checks.append(
        {
            "name": "reflection_non_mutating",
            "ok": len(bad_refl) == 0,
            "value": len(bad_refl),
            "detail": "reflection rows with mutates_memory=1",
        }
    )
    if bad_refl:
        recs.append("Unexpected mutates_memory=true in reflection history.")

    # 5) Pending queue backlog signal (informational)
    checks.append(
        {
            "name": "pending_queue",
            "ok": True,
            "value": pending,
            "detail": f"{pending} pending proposal(s) awaiting human review",
        }
    )
    if pending > 50:
        recs.append("Large pending queue — run `grmc propose` and approve/reject.")

    # 6) Isolated nodes (no edges) — soft warning only
    isolated = 0
    if edges and nodes:
        touched = set()
        for e in edges:
            touched.add(e.source_node_id)
            touched.add(e.target_node_id)
        isolated = sum(1 for n in nodes if n.node_id not in touched)
    checks.append(
        {
            "name": "isolated_nodes",
            "ok": True,
            "value": isolated,
            "detail": f"{isolated} node(s) with no incident edges (informational)",
        }
    )

    passed = sum(1 for c in checks if c["ok"])
    # Weight only hard checks for score
    hard = [c for c in checks if c["name"] != "pending_queue" and c["name"] != "isolated_nodes"]
    hard_pass = sum(1 for c in hard if c["ok"])
    score = hard_pass / len(hard) if hard else 1.0
    # Blend in provenance coverage gently
    score = round(0.7 * score + 0.3 * coverage, 3)
    ok = all(c["ok"] for c in hard)

    # 7) Soft edges should not exceed soft cap (defense in depth)
    softish = [
        e
        for e in edges
        if (e.metadata or {}).get("approved_from_proposal")
        and e.confidence > EDGE_CONF_SOFT_MAX + 1e-9
    ]
    checks.append(
        {
            "name": "edge_metadata_sanity",
            "ok": True,
            "value": len(softish),
            "detail": "edges with conf above soft max (informational recount)",
        }
    )

    # Fixture mode: empty DB is a special "blank slate" pass with note
    if fixture_mode and stats.get("episodes", 0) == 0:
        recs.append("Fixture mode: empty database — baseline only.")

    if not recs and ok:
        recs.append("Health looks fine under current conservative thresholds.")

    return EvalReport(
        ok=ok,
        score=score,
        checks=checks,
        stats={
            **stats,
            "nodes_sampled": len(nodes),
            "edges_sampled": len(edges),
            "reflections_sampled": len(reflections),
            "fixture_mode": fixture_mode,
        },
        recommendations=recs,
    )
