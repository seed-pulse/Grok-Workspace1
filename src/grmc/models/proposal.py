"""Approval-queue proposal: a *suggested* graph write, never applied until approve."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ProposalStatus = Literal["pending", "approved", "rejected"]
ProposalKind = Literal["concept_candidate", "manual"]


class Proposal(BaseModel):
    """Human-gated suggestion produced by reflection (or manual entry)."""

    model_config = ConfigDict()

    proposal_id: str = Field(default_factory=lambda: f"prop_{uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProposalStatus = "pending"
    kind: ProposalKind = "concept_candidate"
    label: str
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    source: str = "heuristic"
    report_id: Optional[str] = None
    supporting_episode_ids: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    reviewed_at: Optional[datetime] = None
    resulting_node_id: Optional[str] = None
    review_note: Optional[str] = None
