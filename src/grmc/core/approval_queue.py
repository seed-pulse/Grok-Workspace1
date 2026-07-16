"""Human-in-the-loop approval gate.

Reflection *thinks* and may enqueue proposals.
Only ``approve()`` writes GraphNodes — never automatic.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from ..models.graph_node import GraphNode, NodeType
from ..models.proposal import Proposal
from ..models.reflection_report import ConceptCandidate, ReflectionReport
from ..storage.sqlite_store import SQLiteStore

# Hard cap on confidence for first-time human approvals (truth-seeking).
DEFAULT_APPROVE_CONFIDENCE_CAP = 0.55


class ApprovalQueue:
    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def enqueue_from_report(
        self,
        report: ReflectionReport,
        *,
        max_candidates: int = 25,
        skip_duplicates: bool = True,
        conservative_filter: bool = True,
    ) -> List[Proposal]:
        """Turn concept candidates into pending proposals. Does not touch graph.

        When ``conservative_filter`` is True (default), only enqueue stronger
        signals: explicit episode fields, or heuristics with freq>=2 and
        conf>=0.28 — reduces stopword-like noise in the human queue.
        """
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
        # Longer multi-word / technical labels may still be useful once
        if candidate.frequency >= 1 and candidate.confidence >= 0.35 and len(candidate.label) >= 8:
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
            if existing:
                return None
            # Also skip if already an approved graph node with same label
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

    def list(
        self,
        status: Optional[str] = "pending",
        limit: int = 50,
    ) -> List[Proposal]:
        return self.sqlite.list_proposals(status=status, limit=limit)

    def get(self, proposal_id: str) -> Optional[Proposal]:
        return self.sqlite.get_proposal(proposal_id)

    def approve(
        self,
        proposal_id: str,
        *,
        node_type: NodeType = "concept",
        confidence_cap: float = DEFAULT_APPROVE_CONFIDENCE_CAP,
        note: Optional[str] = None,
        merge_if_exists: bool = True,
    ) -> GraphNode:
        """Promote a pending proposal to a GraphNode. First real graph write path."""
        proposal = self.sqlite.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is {proposal.status!r}, expected 'pending'"
            )

        capped = min(float(proposal.confidence), float(confidence_cap))
        existing = self.sqlite.find_graph_node_by_label(proposal.label, node_type)

        if existing and merge_if_exists:
            # Conservative merge: keep max conf still under cap, union supports
            supports = list(
                dict.fromkeys(
                    list(existing.supporting_episodes)
                    + list(proposal.supporting_episode_ids)
                )
            )
            existing.supporting_episodes = supports
            existing.confidence = min(
                max(existing.confidence, capped), confidence_cap
            )
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

        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        proposal.resulting_node_id = node.node_id
        proposal.review_note = note
        self.sqlite.update_proposal(proposal)
        return node

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

    def enqueue_many_labels(
        self,
        labels: Sequence[str],
        *,
        confidence: float = 0.35,
        source: str = "manual",
    ) -> List[Proposal]:
        """Manual proposals (still pending until approve)."""
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
