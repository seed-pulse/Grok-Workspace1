"""Read-only graph neighborhood queries (no writes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..models.graph_edge import GraphEdge
from ..models.graph_node import GraphNode
from ..storage.sqlite_store import SQLiteStore


@dataclass
class NeighborHop:
    depth: int
    via_edge_id: str
    via_edge_type: str
    direction: str  # out | in
    node: GraphNode
    edge_confidence: float


@dataclass
class NeighborhoodResult:
    root: GraphNode
    hops: List[NeighborHop] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": {
                "node_id": self.root.node_id,
                "label": self.root.label,
                "type": self.root.type,
                "confidence": self.root.confidence,
            },
            "neighbors": [
                {
                    "depth": h.depth,
                    "direction": h.direction,
                    "edge_id": h.via_edge_id,
                    "edge_type": h.via_edge_type,
                    "edge_confidence": h.edge_confidence,
                    "node_id": h.node.node_id,
                    "label": h.node.label,
                    "confidence": h.node.confidence,
                }
                for h in self.hops
            ],
            "provenance": self.provenance,
        }


def neighborhood(
    sqlite: SQLiteStore,
    node_id: str,
    *,
    depth: int = 1,
    edge_type: Optional[str] = None,
    include_provenance: bool = True,
    limit: int = 100,
) -> NeighborhoodResult:
    """BFS neighborhood up to depth 1 or 2."""
    depth = 1 if depth < 1 else (2 if depth > 2 else depth)
    root = sqlite.get_graph_node(node_id)
    if root is None:
        raise KeyError(f"Unknown node: {node_id}")

    hops: List[NeighborHop] = []
    visited: Set[str] = {node_id}
    frontier = [node_id]

    for d in range(1, depth + 1):
        next_frontier: List[str] = []
        for nid in frontier:
            edges = sqlite.list_graph_edges(node_id=nid, edge_type=edge_type, limit=500)
            for e in edges:
                if edge_type and e.edge_type != edge_type:
                    continue
                if e.source_node_id == nid:
                    other_id = e.target_node_id
                    direction = "out"
                else:
                    other_id = e.source_node_id
                    direction = "in"
                if other_id in visited:
                    continue
                other = sqlite.get_graph_node(other_id)
                if other is None:
                    continue
                visited.add(other_id)
                next_frontier.append(other_id)
                hops.append(
                    NeighborHop(
                        depth=d,
                        via_edge_id=e.edge_id,
                        via_edge_type=e.edge_type,
                        direction=direction,
                        node=other,
                        edge_confidence=e.confidence,
                    )
                )
                if len(hops) >= limit:
                    break
            if len(hops) >= limit:
                break
        frontier = next_frontier
        if not frontier or len(hops) >= limit:
            break

    provenance: List[Dict[str, Any]] = []
    if include_provenance:
        for link in sqlite.list_links_for_node(node_id):
            ep = sqlite.get_episode(link.episode_id)
            provenance.append(
                {
                    "link_id": link.link_id,
                    "episode_id": link.episode_id,
                    "relation": link.relation,
                    "confidence": link.confidence,
                    "proposal_id": link.proposal_id,
                    "summary": (ep or {}).get("summary", "")[:120],
                }
            )

    return NeighborhoodResult(root=root, hops=hops, provenance=provenance)
