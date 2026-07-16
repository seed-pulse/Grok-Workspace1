"""Operational CLIs: eval harness + legacy migrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.eval_harness import run_eval
from ..storage.legacy_migrate import migrate_chroma_to_sqlite
from ..storage.sqlite_store import SQLiteStore

ops_help = "Eval + migration utilities (read-mostly; migrate is additive)."
console = Console()

DEFAULT_DATA = "./grmc_data"

ops_app = typer.Typer(help=ops_help, no_args_is_help=True)


@ops_app.command("eval")
def eval_cmd(
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
    json_out: Optional[Path] = typer.Option(
        None, "--json", "-o", help="Write full eval JSON"
    ),
):
    """Run conservative health checks (over-confidence, provenance, etc.)."""
    db = SQLiteStore(Path(data_dir) / "grmc.db")
    report = run_eval(db)

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
