"""Bridge message protocol.

Design goal: let *web Grok* (grok.com) and *CLI Grok* (this repo / GRMC)
collaborate without fragile browser login automation.

Human is the courier when needed; Git/files are the shared bus.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

Party = Literal["web-grok", "cli-grok", "human", "system"]
MessageStatus = Literal["open", "delivered", "acked", "superseded"]


class BridgeMessage(BaseModel):
    model_config = ConfigDict()

    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender: Party
    recipient: Party
    body: str
    status: MessageStatus = "open"
    in_reply_to: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def one_line_preview(self, width: int = 80) -> str:
        text = " ".join(self.body.strip().split())
        if len(text) > width:
            return text[: width - 1] + "…"
        return text


class BridgeMeta(BaseModel):
    model_config = ConfigDict()

    channel_id: str = Field(default_factory=lambda: f"ch_{uuid4().hex[:10]}")
    title: str = "GRMC Dual-Grok Bridge"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    protocol_version: str = "1.0"
    notes: str = (
        "Primary channel between web-grok and cli-grok. "
        "Browser login automation is intentionally out of scope."
    )
