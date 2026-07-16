"""Markdown + JSONL dual representation of a bridge channel.

Human-friendly: ``active_channel.md`` (easy to paste into / from grok.com)
Machine-friendly: ``channel.jsonl`` (stable append log, source of truth)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .protocol import BridgeMessage, BridgeMeta, MessageStatus, Party

DEFAULT_BRIDGE_DIR = Path("bridge")
JSONL_NAME = "channel.jsonl"
MD_NAME = "active_channel.md"


class BridgeChannel:
    def __init__(self, root: Path | str = DEFAULT_BRIDGE_DIR):
        self.root = Path(root)
        self.jsonl_path = self.root / JSONL_NAME
        self.md_path = self.root / MD_NAME
        self.meta_path = self.root / "channel_meta.json"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init(self, title: str = "GRMC Dual-Grok Bridge") -> BridgeMeta:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            return self.load_meta()

        meta = BridgeMeta(title=title)
        self.meta_path.write_text(
            meta.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if not self.jsonl_path.exists():
            self.jsonl_path.write_text("", encoding="utf-8")
        self._rewrite_markdown(self.list_messages(), meta)
        return meta

    def load_meta(self) -> BridgeMeta:
        if not self.meta_path.exists():
            return self.init()
        return BridgeMeta.model_validate_json(
            self.meta_path.read_text(encoding="utf-8")
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def list_messages(self) -> List[BridgeMessage]:
        if not self.jsonl_path.exists():
            return []
        messages: List[BridgeMessage] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            messages.append(BridgeMessage.model_validate_json(line))
        return messages

    def append(self, message: BridgeMessage, rewrite_md: bool = True) -> BridgeMessage:
        self.init()  # ensure paths exist
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(message.model_dump_json() + "\n")
        if rewrite_md:
            self._rewrite_markdown(self.list_messages(), self.load_meta())
        return message

    def post(
        self,
        body: str,
        sender: Party,
        recipient: Party,
        *,
        in_reply_to: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        status: MessageStatus = "open",
    ) -> BridgeMessage:
        msg = BridgeMessage(
            body=body.strip(),
            sender=sender,
            recipient=recipient,
            in_reply_to=in_reply_to,
            tags=list(tags or []),
            status=status,
        )
        return self.append(msg)

    def set_status(self, message_id: str, status: MessageStatus) -> BridgeMessage:
        messages = self.list_messages()
        found: Optional[BridgeMessage] = None
        updated: List[BridgeMessage] = []
        for m in messages:
            if m.id == message_id:
                data = m.model_dump()
                data["status"] = status
                found = BridgeMessage.model_validate(data)
                updated.append(found)
            else:
                updated.append(m)
        if found is None:
            raise KeyError(f"Unknown message id: {message_id}")
        self._rewrite_jsonl(updated)
        self._rewrite_markdown(updated, self.load_meta())
        return found

    def inbox(
        self,
        recipient: Party = "cli-grok",
        statuses: Iterable[MessageStatus] = ("open", "delivered"),
    ) -> List[BridgeMessage]:
        wanted = set(statuses)
        return [
            m
            for m in self.list_messages()
            if m.recipient == recipient and m.status in wanted
        ]

    def last_from(self, sender: Party) -> Optional[BridgeMessage]:
        msgs = [m for m in self.list_messages() if m.sender == sender]
        return msgs[-1] if msgs else None

    def paste_block(self, message: BridgeMessage) -> str:
        """Format a message for easy copy into grok.com."""
        ts = message.timestamp.isoformat()
        reply = f"\nin_reply_to: {message.in_reply_to}" if message.in_reply_to else ""
        return (
            f"--- GRMC Bridge Message ---\n"
            f"id: {message.id}\n"
            f"from: {message.sender}\n"
            f"to: {message.recipient}\n"
            f"timestamp: {ts}\n"
            f"status: {message.status}{reply}\n"
            f"---\n\n"
            f"{message.body.strip()}\n"
        )

    def import_paste(self, text: str, default_sender: Party = "web-grok") -> BridgeMessage:
        """Parse a pasted block (or free text) from the other Grok / human."""
        text = text.strip()
        if not text:
            raise ValueError("Empty paste")

        header: dict = {}
        body = text
        # Structured paste
        if text.startswith("---") or text.startswith("id:"):
            # Split header/body on a lone --- line after headers, or first blank line after kv
            m = re.match(
                r"(?:---[^\n]*\n)?(?P<header>(?:[a-z_]+:\s*.+\n)+)(?:---\s*\n)?(?P<body>.*)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                for line in m.group("header").splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        header[k.strip().lower()] = v.strip()
                body = (m.group("body") or "").strip() or text

        sender = _as_party(header.get("from"), default_sender)
        recipient = _as_party(header.get("to"), _counterparty(sender))
        msg = BridgeMessage(
            id=header.get("id") or f"msg_{datetime.utcnow().strftime('%H%M%S')}_{len(body)%997:03d}",
            sender=sender,
            recipient=recipient,
            body=body,
            in_reply_to=header.get("in_reply_to") or header.get("in-reply-to"),
            status="delivered",
            tags=["imported-paste"],
            metadata={"import_source": "paste"},
        )
        # Avoid id collision on re-import: if id exists, mint new
        existing_ids = {m.id for m in self.list_messages()}
        if msg.id in existing_ids:
            msg = BridgeMessage(
                **{
                    **msg.model_dump(),
                    "id": f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                }
            )
        return self.append(msg)

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def _rewrite_jsonl(self, messages: Sequence[BridgeMessage]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("w", encoding="utf-8") as fh:
            for m in messages:
                fh.write(m.model_dump_json() + "\n")

    def _rewrite_markdown(
        self, messages: Sequence[BridgeMessage], meta: BridgeMeta
    ) -> None:
        lines = [
            f"# {meta.title}",
            "",
            f"> channel_id: `{meta.channel_id}` · protocol `{meta.protocol_version}`",
            ">",
            f"> {meta.notes}",
            "",
            "## How to use (human courier)",
            "",
            "1. **web-grok → cli-grok**: paste the other Grok's message here, or run "
            "`grmc bridge receive --file note.md`",
            "2. **cli-grok → web-grok**: run `grmc bridge reply -t \"...\"` then "
            "`grmc bridge paste` and paste into grok.com",
            "3. Optional: `grmc bridge sync-memory` to store the channel into GRMC",
            "",
            "---",
            "",
        ]
        if not messages:
            lines.append("_No messages yet._")
            lines.append("")
        for m in messages:
            ts = m.timestamp.isoformat()
            lines.append(f"## [{ts}] {m.sender} → {m.recipient}")
            lines.append("")
            lines.append(f"- **id**: `{m.id}`")
            lines.append(f"- **status**: `{m.status}`")
            if m.in_reply_to:
                lines.append(f"- **in_reply_to**: `{m.in_reply_to}`")
            if m.tags:
                lines.append(f"- **tags**: {', '.join(m.tags)}")
            lines.append("")
            lines.append(m.body.strip())
            lines.append("")
            lines.append("---")
            lines.append("")
        self.md_path.write_text("\n".join(lines), encoding="utf-8")


def _as_party(value: Optional[str], default: Party) -> Party:
    if not value:
        return default
    v = value.strip().lower()
    aliases = {
        "web": "web-grok",
        "web-grok": "web-grok",
        "grok.com": "web-grok",
        "cli": "cli-grok",
        "cli-grok": "cli-grok",
        "local": "cli-grok",
        "human": "human",
        "user": "human",
        "system": "system",
    }
    party = aliases.get(v, default)
    return party  # type: ignore[return-value]


def _counterparty(sender: Party) -> Party:
    if sender == "web-grok":
        return "cli-grok"
    if sender == "cli-grok":
        return "web-grok"
    return "cli-grok"
