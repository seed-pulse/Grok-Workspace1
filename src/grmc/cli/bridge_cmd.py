"""CLI subcommands for the dual-Grok bridge channel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..bridge.channel import BridgeChannel
from ..bridge.fetch import fetch_url
from ..bridge.memory_sync import sync_channel_to_memory
from ..bridge.protocol import Party
from ..core.memory_manager import MemoryManager

bridge_app = typer.Typer(
    help=(
        "Dual-Grok file bridge (human courier). "
        "Not a grok.com login bot — see docs/BRIDGE.md."
    ),
    no_args_is_help=True,
)
console = Console()

DEFAULT_BRIDGE = "bridge"
DEFAULT_DATA = "./grmc_data"


def _channel(root: str) -> BridgeChannel:
    return BridgeChannel(root)


@bridge_app.command("init")
def bridge_init(
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Create bridge/ channel files if missing."""
    ch = _channel(bridge_dir)
    meta = ch.init()
    console.print(f"[green]✓[/green] Bridge ready: {ch.root.resolve()}")
    console.print(f"  channel_id: {meta.channel_id}")
    console.print(f"  markdown:   {ch.md_path}")
    console.print(f"  log:        {ch.jsonl_path}")


@bridge_app.command("status")
def bridge_status(
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Show channel meta + open inbox for cli-grok."""
    ch = _channel(bridge_dir)
    meta = ch.init()
    msgs = ch.list_messages()
    inbox = ch.inbox("cli-grok")

    console.print(
        Panel.fit(
            f"[bold]{meta.title}[/bold]\n"
            f"channel_id={meta.channel_id}\n"
            f"messages={len(msgs)}  open_for_cli={len(inbox)}\n"
            f"dir={ch.root.resolve()}",
            title="Bridge status",
            border_style="cyan",
        )
    )
    if not msgs:
        console.print("[dim]No messages. Receive from web-grok or post a reply.[/dim]")
        return

    table = Table(title="Recent messages")
    table.add_column("id", style="cyan")
    table.add_column("from→to")
    table.add_column("status")
    table.add_column("preview")
    for m in msgs[-10:]:
        table.add_row(
            m.id,
            f"{m.sender}→{m.recipient}",
            m.status,
            m.one_line_preview(50),
        )
    console.print(table)


@bridge_app.command("receive")
def bridge_receive(
    text: Optional[str] = typer.Option(
        None, "--text", "-t", help="Message body from web-grok / human"
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read paste from file"
    ),
    sender: str = typer.Option("web-grok", "--from", help="Sender party"),
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Import a message from web-grok (paste or file) into the channel."""
    if text is None and file is None:
        console.print("[red]Provide --text or --file[/red]")
        raise typer.Exit(1)
    raw = text if text is not None else file.read_text(encoding="utf-8")  # type: ignore[union-attr]
    ch = _channel(bridge_dir)
    msg = ch.import_paste(raw, default_sender=sender)  # type: ignore[arg-type]
    console.print(f"[green]✓[/green] Received {msg.id} ({msg.sender}→{msg.recipient})")
    console.print(Panel(msg.body, title="Body", border_style="green"))


@bridge_app.command("reply")
def bridge_reply(
    text: str = typer.Option(..., "--text", "-t", help="Reply body from cli-grok"),
    to: str = typer.Option("web-grok", "--to", help="Recipient"),
    reply_to: Optional[str] = typer.Option(
        None, "--reply-to", help="Message id being answered"
    ),
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Post a cli-grok → web-grok message (then use `bridge paste`)."""
    ch = _channel(bridge_dir)
    # Default reply-to: latest open inbox item
    if reply_to is None:
        inbox = ch.inbox("cli-grok")
        if inbox:
            reply_to = inbox[-1].id
    msg = ch.post(
        body=text,
        sender="cli-grok",
        recipient=to,  # type: ignore[arg-type]
        in_reply_to=reply_to,
        tags=["cli-reply"],
        status="open",
    )
    console.print(f"[green]✓[/green] Posted {msg.id} (reply_to={reply_to})")
    console.print("[dim]Next: grmc bridge paste   # copy into grok.com[/dim]")


@bridge_app.command("paste")
def bridge_paste(
    message_id: Optional[str] = typer.Option(
        None, "--id", help="Message id (default: last cli-grok outbound)"
    ),
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
    mark_delivered: bool = typer.Option(
        True, "--mark-delivered/--no-mark", help="Mark status delivered after print"
    ),
):
    """Print a copy-ready block for pasting into grok.com."""
    ch = _channel(bridge_dir)
    msg = None
    if message_id:
        for m in ch.list_messages():
            if m.id == message_id:
                msg = m
                break
        if msg is None:
            console.print(f"[red]Unknown id: {message_id}[/red]")
            raise typer.Exit(1)
    else:
        msg = ch.last_from("cli-grok")
        if msg is None:
            console.print("[yellow]No cli-grok messages yet. Use: grmc bridge reply[/yellow]")
            raise typer.Exit(1)

    block = ch.paste_block(msg)
    console.print(block)
    if mark_delivered and msg.status == "open":
        ch.set_status(msg.id, "delivered")
        console.print(f"[dim]status → delivered ({msg.id})[/dim]")


@bridge_app.command("inbox")
def bridge_inbox(
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
    party: str = typer.Option("cli-grok", "--as", help="Inbox owner"),
):
    """List open messages for a party (default: cli-grok)."""
    ch = _channel(bridge_dir)
    items = ch.inbox(party)  # type: ignore[arg-type]
    if not items:
        console.print(f"[dim]Inbox empty for {party}[/dim]")
        return
    for m in items:
        console.print(
            Panel(
                m.body,
                title=f"{m.id} · {m.sender}→{m.recipient} · {m.status}",
                border_style="yellow",
            )
        )


@bridge_app.command("ack")
def bridge_ack(
    message_id: str = typer.Argument(..., help="Message id to acknowledge"),
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Mark a message as acked (human/cli finished handling it)."""
    ch = _channel(bridge_dir)
    try:
        msg = ch.set_status(message_id, "acked")
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] {msg.id} → acked")


@bridge_app.command("sync-memory")
def bridge_sync_memory(
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
    embedder: str = typer.Option("auto", "--embedder"),
    all_messages: bool = typer.Option(
        False, "--all", help="Re-ingest even if previously synced"
    ),
):
    """Ingest bridge messages into GRMC episodic memory (no graph mutation)."""
    ch = _channel(bridge_dir)
    ch.init()
    manager = MemoryManager.from_data_dir(data_dir, embedder_prefer=embedder)
    result = sync_channel_to_memory(ch, manager, only_new=not all_messages)
    console.print(
        f"[green]✓[/green] ingested={len(result['ingested_episodes'])} "
        f"skipped={len(result['skipped_message_ids'])} "
        f"total_messages={result['total_messages']}"
    )


@bridge_app.command("fetch")
def bridge_fetch(
    url: str = typer.Argument(..., help="Public URL to fetch"),
    backend: str = typer.Option(
        "auto", "--backend", help="auto | httpx | playwright"
    ),
    save: Optional[Path] = typer.Option(
        None, "--save", "-o", help="Write extracted text to file"
    ),
    to_bridge: bool = typer.Option(
        False,
        "--to-bridge",
        help="Also post a system→cli-grok summary message into the channel",
    ),
    bridge_dir: str = typer.Option(DEFAULT_BRIDGE, "--dir", "-d"),
):
    """Fetch a public page (not for grok.com private chats)."""
    result = fetch_url(url, backend=backend)
    style = "green" if result.ok else "yellow"
    console.print(
        Panel.fit(
            f"backend={result.backend}  ok={result.ok}  status={result.status_code}\n"
            f"title={result.title or '(none)'}\n"
            f"error={result.error or '-'}",
            title="Fetch result",
            border_style=style,
        )
    )
    for note in result.notes:
        console.print(f"[yellow]• {note}[/yellow]")
    if result.text:
        console.print(Panel(result.preview(800), title="Text preview", border_style="dim"))
    if save and result.text:
        save.write_text(result.text, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {save}")
    if to_bridge:
        ch = _channel(bridge_dir)
        body = (
            f"Fetched URL: {result.url}\n"
            f"ok={result.ok} backend={result.backend} title={result.title}\n\n"
            f"{result.preview(2000)}"
        )
        if result.notes:
            body += "\n\nNotes:\n" + "\n".join(f"- {n}" for n in result.notes)
        msg = ch.post(
            body=body,
            sender="system",
            recipient="cli-grok",
            tags=["fetch", result.backend],
            status="delivered",
        )
        console.print(f"[green]✓[/green] posted to bridge as {msg.id}")
