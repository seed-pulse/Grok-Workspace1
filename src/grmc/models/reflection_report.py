"""Structured output of a reflection pass.

Reflection is intentionally non-mutating: this report is a proposal for a human
(or a future approval gate), not an automatic knowledge-graph write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConceptCandidate(BaseModel):
    """A concept noticed during reflection — not yet a graph node."""

    label: str
    frequency: int = 1
    supporting_episode_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Conservative default; heuristics are weak signals only.",
    )
    source: str = "heuristic"  # heuristic | episode_field | topic_query


class ContradictionFlag(BaseModel):
    """A possible tension between two episodes. Always treated as uncertain."""

    episode_id_a: str
    episode_id_b: str
    summary_a: str
    summary_b: str
    reason: str
    confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Low by design — heuristics over-flag less often than under-flag.",
    )
    requires_human_review: bool = True
    method: str = "negation_overlap"  # negation_overlap | embedding_polarity | corpus_poles
    similarity: Optional[float] = None


class EdgeSuggestion(BaseModel):
    """Soft node↔node edge idea from reflection — never auto-written."""

    source_label: str = ""
    target_label: str = ""
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    edge_type: str = "related_to"  # supports | contradicts | related_to (basic set)
    confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    reason: str = ""
    supporting_episode_ids: List[str] = Field(default_factory=list)
    requires_human_review: bool = True


class ReflectionReport(BaseModel):
    """Full report from one reflection cycle. Never mutates memory by itself."""

    report_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: Literal["recent", "topic", "all"] = "recent"
    topic: Optional[str] = None

    episodes_analyzed: int = 0
    episode_ids: List[str] = Field(default_factory=list)

    concept_candidates: List[ConceptCandidate] = Field(default_factory=list)
    concept_frequencies: Dict[str, int] = Field(default_factory=dict)
    potential_contradictions: List[ContradictionFlag] = Field(default_factory=list)
    edge_suggestions: List[EdgeSuggestion] = Field(default_factory=list)

    potential_issues: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    confidence_level: Literal["conservative", "moderate", "aggressive"] = "conservative"
    mutates_memory: bool = False
    engine_version: str = "0.1.0"

    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict()
