"""Human-in-the-loop approval gate for nodes *and* edges.

Reflection *thinks* and may enqueue concept proposals.
Edge proposals are manual (or future soft-suggestions) — never auto-strong.
Only ``approve()`` writes GraphNodes / GraphEdges / provenance links.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from ..models.graph_edge import (
    EDGE_TYPES,
    EpisodeNodeLink,
    GraphEdge,
    ProvenanceRelation,
)
from ..models.graph_node import GraphNode, NodeType
from ..models.proposal import Proposal
from ..models.reflection_report import ConceptCandidate, ReflectionReport
from ..storage.sqlite_store import SQLiteStore

# Hard cap on confidence for first-time human approvals (truth-seeking).
DEFAULT_APPROVE_CONFIDENCE_CAP = 0.55
DEFAULT_EDGE_CONFIDENCE_CAP = 0.45


class ApprovalQueue:
    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    # ------------------------------------------------------------------
    # Concept proposals
    # ------------------------------------------------------------------

    def enqueue_from_report(
        self,
        report: ReflectionReport,
        *,
        max_candidates: int = 25,
        skip_duplicates: bool = True,
        conservative_filter: bool = True,
    ) -> List[Proposal]:
        """Turn concept candidates into pending proposals. Does not touch graph."""
        created: List[Proposal] = []
        for candidate in report.concept_candidates[:max_candidates]:
            if conservative_filter and not self._is_enqueue_worthy(candidate):
                continue
            prop = self.enqueue_candidate(
                candidate,
                report_id=report.report_id,
                skip_if_pending_duplicate=skip_duplicates,
            )
            if prop is not None:
                created.append(prop)
        return created

    @staticmethod
    def _is_enqueue_worthy(candidate: ConceptCandidate) -> bool:
        if candidate.source == "episode_field":
            return True
        if candidate.frequency >= 2 and candidate.confidence >= 0.28:
            return True
        if (
            candidate.frequency >= 1
            and candidate.confidence >= 0.35
            and len(candidate.label) >= 8
        ):
            return True
        return False

    def enqueue_candidate(
        self,
        candidate: ConceptCandidate,
        *,
        report_id: Optional[str] = None,
        skip_if_pending_duplicate: bool = True,
    ) -> Optional[Proposal]:
        label = (candidate.label or "").strip()
        if not label:
            return None
        if skip_if_pending_duplicate:
            existing = self.sqlite.find_pending_by_label(label)
            if existing and not existing.is_edge:
                return None
            if self.sqlite.find_graph_node_by_label(label, "concept"):
                return None

        proposal = Proposal(
            kind="concept_candidate",
            label=label,
            confidence=float(candidate.confidence),
            source=candidate.source,
            report_id=report_id,
            supporting_episode_ids=list(candidate.supporting_episode_ids),
            payload={
                "frequency": candidate.frequency,
                "source": candidate.source,
                "label": label,
            },
            status="pending",
        )
        self.sqlite.add_proposal(proposal)
        return proposal

    def enqueue_many_labels(
        self,
        labels: Sequence[str],
        *,
        confidence: float = 0.35,
        source: str = "manual",
    ) -> List[Proposal]:
        out: List[Proposal] = []
        for label in labels:
            label = label.strip()
            if not label:
                continue
            candidate = ConceptCandidate(
                label=label,
                frequency=1,
                confidence=confidence,
                source=source,
            )
            prop = self.enqueue_candidate(candidate, skip_if_pending_duplicate=True)
            if prop:
                out.append(prop)
        return out

    # ------------------------------------------------------------------
    # Edge proposals (always human-gated; never auto-strong)
    # ------------------------------------------------------------------

    def enqueue_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str = "related_to",
        *,
        confidence: float = 0.35,
        supporting_episode_ids: Optional[Sequence[str]] = None,
        report_id: Optional[str] = None,
        note: Optional[str] = None,
        source: str = "manual",
    ) -> Proposal:
        """Queue a node→node edge for approval. Does not write the edge yet."""
        if edge_type not in EDGE_TYPES:
            raise ValueError(
                f"Unknown edge_type {edge_type!r}. Allowed: {', '.join(EDGE_TYPES)}"
            )
        if source_node_id == target_node_id:
            raise ValueError("source and target must differ")

        src = self.sqlite.get_graph_node(source_node_id)
        tgt = self.sqlite.get_graph_node(target_node_id)
        if src is None:
            raise KeyError(f"Unknown source node: {source_node_id}")
        if tgt is None:
            raise KeyError(f"Unknown target node: {target_node_id}")

        existing_edge = self.sqlite.find_edge(source_node_id, target_node_id, edge_type)
        if existing_edge:
            raise ValueError(
                f"Edge already exists: {existing_edge.edge_id} "
                f"({src.label} -[{edge_type}]-> {tgt.label})"
            )

        pending = self.sqlite.find_pending_edge_proposal(
            source_node_id, target_node_id, edge_type
        )
        if pending:
            raise ValueError(f"Pending edge proposal already exists: {pending.proposal_id}")

        # Conservative default confidence for edges
        conf = min(float(confidence), DEFAULT_EDGE_CONFIDENCE_CAP)
        label = f"{src.label} -[{edge_type}]-> {tgt.label}"
        proposal = Proposal(
            kind="edge",
            label=label,
            confidence=conf,
            source=source,
            report_id=report_id,
            supporting_episode_ids=list(supporting_episode_ids or []),
            payload={
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "edge_type": edge_type,
                "source_label": src.label,
                "target_label": tgt.label,
                "note": note,
            },
            status="pending",
        )
        self.sqlite.add_proposal(proposal)
        return proposal

    # ------------------------------------------------------------------
    # List / get / reject
    # ------------------------------------------------------------------

    def list(
        self,
        status: Optional[str] = "pending",
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Proposal]:
        items = self.sqlite.list_proposals(status=status, limit=limit)
        if kind:
            items = [p for p in items if p.kind == kind]
        return items

    def get(self, proposal_id: str) -> Optional[Proposal]:
        return self.sqlite.get_proposal(proposal_id)

    def reject(
        self,
        proposal_id: str,
        *,
        note: Optional[str] = None,
    ) -> Proposal:
        proposal = self.sqlite.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is {proposal.status!r}, expected 'pending'"
            )
        proposal.status = "rejected"
        proposal.reviewed_at = datetime.utcnow()
        proposal.review_note = note
        self.sqlite.update_proposal(proposal)
        return proposal

    # ------------------------------------------------------------------
    # Approve (only graph write path)
    # ------------------------------------------------------------------

    def approve(
        self,
        proposal_id: str,
        *,
        node_type: NodeType = "concept",
        confidence_cap: Optional[float] = None,
        note: Optional[str] = None,
        merge_if_exists: bool = True,
        also_link_related: bool = False,
        related_to_node_id: Optional[str] = None,
        related_edge_type: str = "related_to",
    ) -> Dict[str, Any]:
        """Approve a pending proposal.

        Returns a dict:
          {"kind": "node"|"edge", "node": GraphNode?, "edge": GraphEdge?,
           "provenance_links": [...], "related_edge_proposal": Proposal? }

        If ``also_link_related`` and ``related_to_node_id`` are set on a *node*
        approval, a *pending* edge proposal is enqueued (not auto-written).
        """
        proposal = self.sqlite.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is {proposal.status!r}, expected 'pending'"
            )

        if proposal.is_edge:
            edge = self._approve_edge(
                proposal,
                confidence_cap=confidence_cap
                if confidence_cap is not None
                else DEFAULT_EDGE_CONFIDENCE_CAP,
                note=note,
            )
            return {
                "kind": "edge",
                "edge": edge,
                "node": None,
                "provenance_links": [],
                "related_edge_proposal": None,
            }

        node, links = self._approve_node(
            proposal,
            node_type=node_type,
            confidence_cap=confidence_cap
            if confidence_cap is not None
            else DEFAULT_APPROVE_CONFIDENCE_CAP,
            note=note,
            merge_if_exists=merge_if_exists,
        )

        related_prop: Optional[Proposal] = None
        if also_link_related and related_to_node_id:
            try:
                related_prop = self.enqueue_edge(
                    node.node_id,
                    related_to_node_id,
                    related_edge_type,
                    confidence=min(node.confidence, DEFAULT_EDGE_CONFIDENCE_CAP),
                    supporting_episode_ids=proposal.supporting_episode_ids,
                    report_id=proposal.report_id,
                    note=f"Suggested alongside approve of {proposal.proposal_id}",
                    source="approve-suggest",
                )
            except (KeyError, ValueError):
                related_prop = None

        return {
            "kind": "node",
            "node": node,
            "edge": None,
            "provenance_links": links,
            "related_edge_proposal": related_prop,
        }

    def _approve_node(
        self,
        proposal: Proposal,
        *,
        node_type: NodeType,
        confidence_cap: float,
        note: Optional[str],
        merge_if_exists: bool,
    ) -> tuple[GraphNode, List[EpisodeNodeLink]]:
        capped = min(float(proposal.confidence), float(confidence_cap))
        existing = self.sqlite.find_graph_node_by_label(proposal.label, node_type)

        if existing and merge_if_exists:
            supports = list(
                dict.fromkeys(
                    list(existing.supporting_episodes)
                    + list(proposal.supporting_episode_ids)
                )
            )
            existing.supporting_episodes = supports
            existing.confidence = min(max(existing.confidence, capped), confidence_cap)
            existing.version += 1
            existing.last_reflected = datetime.utcnow()
            meta = dict(existing.metadata or {})
            meta["last_approved_proposal"] = proposal.proposal_id
            if note:
                meta["last_review_note"] = note
            existing.metadata = meta
            self.sqlite.update_graph_node(existing)
            node = existing
        else:
            node = GraphNode(
                type=node_type,
                label=proposal.label,
                confidence=capped,
                supporting_episodes=list(proposal.supporting_episode_ids),
                last_reflected=datetime.utcnow(),
                version=1,
                metadata={
                    "approved_from_proposal": proposal.proposal_id,
                    "report_id": proposal.report_id,
                    "source": proposal.source,
                    "review_note": note,
                    "confidence_capped_at": confidence_cap,
                },
            )
            self.sqlite.add_graph_node(node)

        links = self._write_provenance(
            node=node,
            proposal=proposal,
            relation="supports",
            confidence=capped,
            note=note,
        )

        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        proposal.resulting_node_id = node.node_id
        proposal.review_note = note
        self.sqlite.update_proposal(proposal)
        return node, links

    def _approve_edge(
        self,
        proposal: Proposal,
        *,
        confidence_cap: float,
        note: Optional[str],
    ) -> GraphEdge:
        payload = proposal.payload or {}
        source_id = payload.get("source_node_id")
        target_id = payload.get("target_node_id")
        edge_type = payload.get("edge_type", "related_to")
        if not source_id or not target_id:
            raise ValueError("Edge proposal missing source_node_id / target_node_id")
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Invalid edge_type in proposal: {edge_type}")

        src = self.sqlite.get_graph_node(source_id)
        tgt = self.sqlite.get_graph_node(target_id)
        if src is None or tgt is None:
            raise ValueError("Both nodes must exist before approving an edge")

        capped = min(float(proposal.confidence), float(confidence_cap))
        existing = self.sqlite.find_edge(source_id, target_id, edge_type)
        if existing:
            # Merge provenance episodes; do not raise confidence above cap
            eps = list(
                dict.fromkeys(
                    list(existing.supporting_episode_ids)
                    + list(proposal.supporting_episode_ids)
                )
            )
            existing.supporting_episode_ids = eps
            existing.confidence = min(max(existing.confidence, capped), confidence_cap)
            existing.proposal_id = proposal.proposal_id
            meta = dict(existing.metadata or {})
            meta["last_approved_proposal"] = proposal.proposal_id
            if note:
                meta["last_review_note"] = note
            existing.metadata = meta
            self.sqlite.update_graph_edge(existing)
            edge = existing
        else:
            edge = GraphEdge(
                source_node_id=source_id,
                target_node_id=target_id,
                edge_type=edge_type,  # type: ignore[arg-type]
                confidence=capped,
                proposal_id=proposal.proposal_id,
                report_id=proposal.report_id,
                supporting_episode_ids=list(proposal.supporting_episode_ids),
                metadata={
                    "approved_from_proposal": proposal.proposal_id,
                    "review_note": note,
                    "source_label": payload.get("source_label") or src.label,
                    "target_label": payload.get("target_label") or tgt.label,
                    "confidence_capped_at": confidence_cap,
                },
            )
            self.sqlite.add_graph_edge(edge)

        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        proposal.resulting_node_id = edge.edge_id  # stores resulting id for edges too
        proposal.review_note = note
        payload = dict(proposal.payload)
        payload["resulting_edge_id"] = edge.edge_id
        proposal.payload = payload
        self.sqlite.update_proposal(proposal)
        return edge

    def _write_provenance(
        self,
        *,
        node: GraphNode,
        proposal: Proposal,
        relation: ProvenanceRelation,
        confidence: float,
        note: Optional[str],
    ) -> List[EpisodeNodeLink]:
        """Record explicit episode→node links for approved concept."""
        links: List[EpisodeNodeLink] = []
        for episode_id in proposal.supporting_episode_ids:
            if not episode_id:
                continue
            link = EpisodeNodeLink(
                episode_id=episode_id,
                node_id=node.node_id,
                relation=relation,
                proposal_id=proposal.proposal_id,
                report_id=proposal.report_id,
                confidence=confidence,
                note=note or f"approved from {proposal.proposal_id}",
            )
            self.sqlite.add_episode_node_link(link)
            links.append(link)
        return links
