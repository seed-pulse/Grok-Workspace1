from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Episode(BaseModel):
    """A single unit of memory (conversation turn, event, insight, etc.)."""

    model_config = ConfigDict()

    episode_id: str = Field(default_factory=lambda: f"ep_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: Optional[str] = None
    source: str = "unknown"  # e.g. "cli", "grok-conversation", "user-note"
    content_summary: str
    raw_content: Optional[str] = None
    extracted_concepts: List[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    linked_graph_nodes: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
