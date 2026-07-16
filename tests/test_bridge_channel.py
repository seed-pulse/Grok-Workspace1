from pathlib import Path

from grmc.bridge.channel import BridgeChannel
from grmc.bridge.fetch import _html_to_text, _looks_auth_walled


def test_channel_roundtrip(tmp_path: Path):
    ch = BridgeChannel(tmp_path / "bridge")
    meta = ch.init()
    assert meta.channel_id

    inbound = ch.import_paste(
        "Long-term memory should stay conservative and report-only.",
        default_sender="web-grok",
    )
    assert inbound.sender == "web-grok"
    assert inbound.recipient == "cli-grok"

    reply = ch.post(
        body="Agreed. Bridge channel is the reliable courier.",
        sender="cli-grok",
        recipient="web-grok",
        in_reply_to=inbound.id,
    )
    assert reply.in_reply_to == inbound.id

    inbox = ch.inbox("cli-grok")
    assert any(m.id == inbound.id for m in inbox)

    block = ch.paste_block(reply)
    assert reply.id in block
    assert "cli-grok" in block

    ch.set_status(inbound.id, "acked")
    assert ch.inbox("cli-grok") == [] or all(
        m.id != inbound.id for m in ch.inbox("cli-grok")
    )

    # markdown mirror exists
    assert ch.md_path.exists()
    assert ch.jsonl_path.exists()


def test_structured_paste(tmp_path: Path):
    ch = BridgeChannel(tmp_path / "bridge")
    ch.init()
    raw = """--- GRMC Bridge Message ---
id: msg_test123
from: web-grok
to: cli-grok
timestamp: 2026-07-16T00:00:00
status: open
---

Hello from web side.
"""
    msg = ch.import_paste(raw)
    assert msg.sender == "web-grok"
    assert "Hello from web side" in msg.body


def test_html_to_text_strips_scripts():
    title, text = _html_to_text(
        "<html><head><title>Hi</title><script>evil()</script></head>"
        "<body><p>Hello <b>world</b></p></body></html>"
    )
    assert title == "Hi"
    assert "Hello" in text and "world" in text
    assert "evil" not in text


def test_auth_wall_hint_for_grok():
    note = _looks_auth_walled("https://grok.com/c/abc", "Grok", "Grok")
    assert note is not None
