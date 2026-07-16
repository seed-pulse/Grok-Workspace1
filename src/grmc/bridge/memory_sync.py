"""Import bridge messages into GRMC episodic memory (read-path only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.memory_manager import MemoryManager
from ..models.episode import Episode
from .channel import BridgeChannel
from .protocol import BridgeMessage


def message_to_episode(message: BridgeMessage) -> Episode:
    concepts = [
        "bridge",
        f"from:{message.sender}",
        f"to:{message.recipient}",
        *message.tags,
    ]
    summary = (
        f"[bridge {message.sender}→{message.recipient}] {message.one_line_preview(160)}"
    )
    return Episode(
        content_summary=summary,
        raw_content=message.body,
        source=f"bridge:{message.sender}",
        conversation_id=message.id,
        extracted_concepts=concepts,
        importance_score=0.65,
        metadata={
            "bridge_message_id": message.id,
            "bridge_status": message.status,
            "in_reply_to": message.in_reply_to,
            "timestamp": message.timestamp.isoformat(),
        },
    )


def sync_channel_to_memory(
    channel: BridgeChannel,
    manager: MemoryManager,
    *,
    only_new: bool = True,
    already_synced: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Ingest bridge messages as episodes.

    Returns counts; never mutates the knowledge graph.
    """
    synced = set(already_synced or [])
    # Prefer reading a sidecar of synced ids if present
    sidecar = channel.root / ".synced_message_ids"
    if only_new and sidecar.exists():
        synced |= {
            line.strip()
            for line in sidecar.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    ingested: List[str] = []
    skipped: List[str] = []
    for msg in channel.list_messages():
        if only_new and msg.id in synced:
            skipped.append(msg.id)
            continue
        ep = message_to_episode(msg)
        eid = manager.ingest_episode(ep)
        ingested.append(eid)
        synced.add(msg.id)

    sidecar.write_text("\n".join(sorted(synced)) + "\n", encoding="utf-8")
    return {
        "ingested_episodes": ingested,
        "skipped_message_ids": skipped,
        "total_messages": len(channel.list_messages()),
    }
