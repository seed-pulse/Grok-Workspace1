"""Graph edge + episode↔node provenance models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# Conservative, small vocabulary — extend carefully.
EdgeType = Literal[
    "supports",
    "contradicts",
    "related_to",
    "implies",
    "derived_from",
    "part_of",
]

EDGE_TYPES: tuple[str, ...] = (
    "supports",
    "contradicts",
    "related_to",
    "implies",
    "derived_from",
    "part_of",
)

# Episode → node provenance relations
ProvenanceRelation = Literal["supports", "contradicts", "mentioned_in"]

PROVENANCE_RELATIONS: tuple[str, ...] = (
    "supports",
    "contradicts",
    "mentioned_in",
)


class GraphEdge(BaseModel):
    """Directed relation between two graph nodes (written only via approval)."""

    model_config = ConfigDict()

    edge_id: str = Field(default_factory=lambda: f"edge_{uuid4().hex[:12]}")
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType = "related_to"
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    proposal_id: Optional[str] = None
    report_id: Optional[str] = None
    supporting_episode_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def describe(self, src_label: str = "", tgt_label: str = "") -> str:
        a = src_label or self.source_node_id
        b = tgt_label or self.target_node_id
        return f"{a} -[{self.edge_type}]-> {b}"


class EpisodeNodeLink(BaseModel):
    """Explicit provenance: which episode grounds which node."""

    model_config = ConfigDict()

    link_id: str = Field(default_factory=lambda: f"enl_{uuid4().hex[:12]}")
    episode_id: str
    node_id: str
    relation: ProvenanceRelation = "supports"
    proposal_id: Optional[str] = None
    report_id: Optional[str] = None
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
