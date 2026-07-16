from datetime import datetime
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


NodeType = Literal["concept", "belief", "fact", "user_model", "self_model"]


class GraphNode(BaseModel):
    """A node in the semantic knowledge graph."""

    model_config = ConfigDict()

    node_id: str = Field(default_factory=lambda: f"node_{uuid4().hex[:10]}")
    type: NodeType = "concept"
    label: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    supporting_episodes: List[str] = Field(default_factory=list)
    contradicting_episodes: List[str] = Field(default_factory=list)
    last_reflected: Optional[datetime] = None
    version: int = 1
    metadata: dict = Field(default_factory=dict)
