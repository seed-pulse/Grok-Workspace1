"""Operational CLIs: eval, migrate, llm audit, export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.eval_harness import run_eval
from ..core.export_dump import build_dump, dump_markdown
from ..llm.audit import LLMAuditLog
from ..storage.legacy_migrate import migrate_chroma_to_sqlite
from ..storage.sqlite_store import SQLiteStore

console = Console()

DEFAULT_DATA = "./grmc_data"

ops_app = typer.Typer(
    help=(
        "Operations: eval health, export/dump overview, LLM audit log, "
        "legacy migrate (additive). No silent graph writes."
    ),
    no_args_is_help=True,
)


@ops_app.command("eval")
def eval_cmd(
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
    fixture: bool = typer.Option(
        False, "--fixture", help="Note empty/baseline fixture mode"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", "-o", help="Write full eval JSON"
    ),
):
    """Run conservative health checks (over-confidence, provenance, etc.)."""
    db = SQLiteStore(Path(data_dir) / "grmc.db")
    report = run_eval(db, fixture_mode=fixture)

    style = "green" if report.ok else "yellow"
    console.print(
        Panel.fit(
            f"ok={report.ok}  score={report.score:.3f}\n"
            f"episodes={report.stats.get('episodes')}  "
            f"nodes={report.stats.get('graph_nodes')}  "
            f"edges={report.stats.get('graph_edges')}  "
            f"links={report.stats.get('episode_node_links')}",
            title="GRMC Eval",
            border_style=style,
        )
    )

    table = Table(title="Checks")
    table.add_column("Name")
    table.add_column("OK")
    table.add_column("Value")
    table.add_column("Detail")
    for c in report.checks:
        table.add_row(
            c["name"],
            "✓" if c["ok"] else "!",
            str(c.get("value")),
            str(c.get("detail", ""))[:60],
        )
    console.print(table)

    if report.recommendations:
        console.print("[bold]Recommendations[/bold]")
        for r in report.recommendations:
            console.print(f"  • {r}")

    if json_out:
        json_out.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] wrote {json_out}")

    if not report.ok:
        raise typer.Exit(2)


@ops_app.command("migrate-legacy")
def migrate_legacy_cmd(
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
    from_path: Optional[str] = typer.Option(
        None,
        "--from",
        help="Chroma persist dir (default: data_dir root, pre-0.3 layout)",
    ),
):
    """Copy episodes from legacy Chroma into SQLite (additive, no graph writes)."""
    result = migrate_chroma_to_sqlite(data_dir, legacy_chroma_path=from_path)
    console.print(
        Panel.fit(
            f"source={result.source}\n"
            f"scanned={result.scanned}  inserted={result.inserted}  "
            f"skipped_existing={result.skipped_existing}\n"
            f"errors={len(result.errors)}",
            title="Legacy migrate",
            border_style="cyan",
        )
    )
    for err in result.errors[:10]:
        console.print(f"[yellow]• {err}[/yellow]")
    if result.inserted:
        console.print(
            "[dim]Note: vectors were not re-copied. Re-ingest or re-embed if "
            "you need Chroma search for migrated rows.[/dim]"
        )


@ops_app.command("llm-log")
def llm_log_cmd(
    limit: int = typer.Option(20, "--limit", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Show recent LLM audit log (empty if LLM never enabled)."""
    audit = LLMAuditLog(data_dir)
    summary = audit.summary()
    console.print(
        Panel.fit(
            f"path={summary['path']}\n"
            f"calls={summary['calls']}  success={summary['success']}  "
            f"failed={summary['failed']}\n"
            f"total_tokens_est={summary['total_tokens_est']}\n"
            f"by_purpose={summary['by_purpose']}",
            title="LLM audit summary",
            border_style="cyan",
        )
    )
    records = audit.list_recent(limit=limit)
    if not records:
        console.print(
            "[dim]No LLM calls logged. LLM is default-off; enable with "
            "GRMC_LLM=1 or `grmc reflect --llm`.[/dim]"
        )
        return
    table = Table(title=f"Recent LLM calls (up to {limit})")
    table.add_column("ID", style="cyan")
    table.add_column("Time")
    table.add_column("Purpose")
    table.add_column("Model")
    table.add_column("OK")
    table.add_column("Tok", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("Err")
    for r in records:
        table.add_row(
            r.call_id,
            (r.timestamp or "")[:19],
            r.purpose,
            (r.model or "")[:16],
            "✓" if r.success else "!",
            str(r.total_tokens_est),
            str(r.latency_ms),
            (r.error or "")[:24],
        )
    console.print(table)


@ops_app.command("export")
def export_cmd(
    format: str = typer.Option(
        "md", "--format", "-f", help="md | json"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write to file (default: stdout)"
    ),
    recent: int = typer.Option(15, "--recent", "-n", help="Recent episodes to include"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Dump a human-readable overview of memory (read-only; no graph writes)."""
    db = SQLiteStore(Path(data_dir) / "grmc.db")
    fmt = (format or "md").lower()
    if fmt == "json":
        text = json.dumps(
            build_dump(db, recent_episodes=recent),
            indent=2,
            ensure_ascii=False,
        )
    else:
        text = dump_markdown(db, recent_episodes=recent)

    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {output}")
    else:
        console.print(text)


@ops_app.command("dump")
def dump_cmd(
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    recent: int = typer.Option(15, "--recent", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Alias for `ops export --format md`."""
    db = SQLiteStore(Path(data_dir) / "grmc.db")
    text = dump_markdown(db, recent_episodes=recent)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {output}")
    else:
        console.print(text)
