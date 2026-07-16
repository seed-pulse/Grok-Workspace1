"""Read-only graph neighborhood and path queries (no writes)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


@dataclass
class PathStep:
    from_node_id: str
    to_node_id: str
    edge_id: str
    edge_type: str
    direction: str
    edge_confidence: float
    from_label: str = ""
    to_label: str = ""


@dataclass
class GraphPath:
    node_ids: List[str]
    steps: List[PathStep]
    length: int

    def describe(self) -> str:
        if not self.steps:
            return self.node_ids[0] if self.node_ids else ""
        parts = [self.steps[0].from_label or self.steps[0].from_node_id]
        for s in self.steps:
            arrow = f" -[{s.edge_type}/{s.direction}]-> "
            parts.append(arrow + (s.to_label or s.to_node_id))
        return "".join(parts)


@dataclass
class PathQueryResult:
    source: GraphNode
    target: GraphNode
    found: bool
    paths: List[GraphPath] = field(default_factory=list)
    max_depth: int = 3
    provenance: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": {"node_id": self.source.node_id, "label": self.source.label},
            "target": {"node_id": self.target.node_id, "label": self.target.label},
            "found": self.found,
            "max_depth": self.max_depth,
            "paths": [
                {
                    "length": p.length,
                    "node_ids": p.node_ids,
                    "description": p.describe(),
                    "steps": [
                        {
                            "from": s.from_node_id,
                            "to": s.to_node_id,
                            "edge_id": s.edge_id,
                            "edge_type": s.edge_type,
                            "direction": s.direction,
                            "confidence": s.edge_confidence,
                        }
                        for s in p.steps
                    ],
                }
                for p in self.paths
            ],
            "provenance": self.provenance,
        }


def _node_provenance(sqlite: SQLiteStore, node_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for link in sqlite.list_links_for_node(node_id):
        ep = sqlite.get_episode(link.episode_id)
        out.append(
            {
                "link_id": link.link_id,
                "episode_id": link.episode_id,
                "relation": link.relation,
                "confidence": link.confidence,
                "proposal_id": link.proposal_id,
                "summary": (ep or {}).get("summary", "")[:120],
            }
        )
    return out


def _adjacent(
    sqlite: SQLiteStore,
    node_id: str,
    edge_type: Optional[str],
) -> List[Tuple[str, GraphEdge, str]]:
    """Return list of (other_id, edge, direction)."""
    result: List[Tuple[str, GraphEdge, str]] = []
    for e in sqlite.list_graph_edges(node_id=node_id, edge_type=edge_type, limit=500):
        if edge_type and e.edge_type != edge_type:
            continue
        if e.source_node_id == node_id:
            result.append((e.target_node_id, e, "out"))
        else:
            result.append((e.source_node_id, e, "in"))
    return result


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
            for other_id, e, direction in _adjacent(sqlite, nid, edge_type):
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

    provenance = _node_provenance(sqlite, node_id) if include_provenance else []
    return NeighborhoodResult(root=root, hops=hops, provenance=provenance)


def find_paths(
    sqlite: SQLiteStore,
    source_id: str,
    target_id: str,
    *,
    max_depth: int = 3,
    edge_type: Optional[str] = None,
    max_paths: int = 5,
    include_provenance: bool = True,
) -> PathQueryResult:
    """Find shortest path(s) between two nodes (undirected over edges), max_depth hops.

    Returns up to ``max_paths`` paths of the minimal length found within depth.
    If source==target, returns a trivial path of length 0.
    """
    max_depth = 1 if max_depth < 1 else (3 if max_depth > 3 else max_depth)
    max_paths = max(1, min(max_paths, 20))

    source = sqlite.get_graph_node(source_id)
    target = sqlite.get_graph_node(target_id)
    if source is None:
        raise KeyError(f"Unknown source node: {source_id}")
    if target is None:
        raise KeyError(f"Unknown target node: {target_id}")

    label_cache: Dict[str, str] = {
        source_id: source.label,
        target_id: target.label,
    }

    def label_of(nid: str) -> str:
        if nid not in label_cache:
            n = sqlite.get_graph_node(nid)
            label_cache[nid] = n.label if n else nid
        return label_cache[nid]

    if source_id == target_id:
        path = GraphPath(node_ids=[source_id], steps=[], length=0)
        prov = {}
        if include_provenance:
            prov[source_id] = _node_provenance(sqlite, source_id)
        return PathQueryResult(
            source=source,
            target=target,
            found=True,
            paths=[path],
            max_depth=max_depth,
            provenance=prov,
        )

    # BFS over paths: state = (current_node, path_steps as list of (from,to,edge,dir))
    # parent map for first shortest only + collect all shortest
    queue: deque[Tuple[str, List[Tuple[str, str, GraphEdge, str]]]] = deque()
    queue.append((source_id, []))
    # visited_at_depth: node -> min depth reached (allow revisit only at same shortest?)
    # For multiple shortest paths: track visited with depth, only expand if depth <= best
    best_len: Optional[int] = None
    found_paths: List[List[Tuple[str, str, GraphEdge, str]]] = []
    # Prevent exponential blow-up: don't re-expand node at worse/equal depth for first hop
    # For k-shortest of equal length: allow multiple arrivals at same node only at best_len
    seen_depth: Dict[str, int] = {source_id: 0}

    while queue:
        current, steps = queue.popleft()
        depth = len(steps)
        if best_len is not None and depth >= best_len:
            continue
        if depth >= max_depth:
            continue
        for other_id, edge, direction in _adjacent(sqlite, current, edge_type):
            # avoid cycles within this path
            path_nodes = {source_id} | {s[1] for s in steps}
            if other_id in path_nodes:
                continue
            new_steps = steps + [(current, other_id, edge, direction)]
            new_depth = len(new_steps)
            if other_id == target_id:
                if best_len is None or new_depth < best_len:
                    best_len = new_depth
                    found_paths = [new_steps]
                elif new_depth == best_len and len(found_paths) < max_paths:
                    found_paths.append(new_steps)
                continue
            prev = seen_depth.get(other_id)
            # Expand if first visit or same depth as best known (for alternate routes)
            if prev is not None and new_depth > prev:
                continue
            if prev is None or new_depth <= prev:
                seen_depth[other_id] = new_depth
                if best_len is None or new_depth < best_len:
                    queue.append((other_id, new_steps))

    paths: List[GraphPath] = []
    for step_list in found_paths[:max_paths]:
        node_ids = [source_id] + [s[1] for s in step_list]
        path_steps: List[PathStep] = []
        for frm, to, edge, direction in step_list:
            path_steps.append(
                PathStep(
                    from_node_id=frm,
                    to_node_id=to,
                    edge_id=edge.edge_id,
                    edge_type=edge.edge_type,
                    direction=direction,
                    edge_confidence=edge.confidence,
                    from_label=label_of(frm),
                    to_label=label_of(to),
                )
            )
        paths.append(
            GraphPath(node_ids=node_ids, steps=path_steps, length=len(path_steps))
        )

    provenance: Dict[str, List[Dict[str, Any]]] = {}
    if include_provenance and paths:
        nodes_on_paths: Set[str] = set()
        for p in paths:
            nodes_on_paths.update(p.node_ids)
        for nid in nodes_on_paths:
            provenance[nid] = _node_provenance(sqlite, nid)

    return PathQueryResult(
        source=source,
        target=target,
        found=bool(paths),
        paths=paths,
        max_depth=max_depth,
        provenance=provenance,
    )
