"""CLI for graph edges and provenance (human-gated writes)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.memory_manager import MemoryManager
from ..models.graph_edge import EDGE_TYPES
from ..storage.sqlite_store import SQLiteStore

edges_app = typer.Typer(
    help=(
        "Graph edges (node↔node). "
        "`propose` only queues; `grmc approve` is the write gate."
    ),
    no_args_is_help=True,
)
console = Console()

DEFAULT_DATA = "./grmc_data"


def _manager(data_dir: str) -> MemoryManager:
    return MemoryManager.from_data_dir(data_dir, embedder_prefer="hashing")


def _db(data_dir: str) -> SQLiteStore:
    from pathlib import Path

    return SQLiteStore(Path(data_dir) / "grmc.db")


@edges_app.command("list")
def edges_list(
    node_id: Optional[str] = typer.Option(
        None, "--node", help="Filter edges touching this node id"
    ),
    edge_type: Optional[str] = typer.Option(
        None, "--type", help=f"Filter type: {', '.join(EDGE_TYPES)}"
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """List approved graph edges."""
    db = _db(data_dir)
    edges = db.list_graph_edges(node_id=node_id, edge_type=edge_type, limit=limit)
    if not edges:
        console.print("[dim]No edges yet. Propose one: grmc edges propose ...[/dim]")
        return

    table = Table(title="Graph edges")
    table.add_column("Edge ID", style="cyan")
    table.add_column("Type")
    table.add_column("Source")
    table.add_column("Target")
    table.add_column("Conf", justify="right")
    table.add_column("Eps", justify="right")
    for e in edges:
        src = db.get_graph_node(e.source_node_id)
        tgt = db.get_graph_node(e.target_node_id)
        table.add_row(
            e.edge_id,
            e.edge_type,
            (src.label if src else e.source_node_id)[:28],
            (tgt.label if tgt else e.target_node_id)[:28],
            f"{e.confidence:.2f}",
            str(len(e.supporting_episode_ids)),
        )
    console.print(table)


@edges_app.command("propose")
def edges_propose(
    source: str = typer.Option(..., "--from", help="Source node id"),
    target: str = typer.Option(..., "--to", help="Target node id"),
    edge_type: str = typer.Option(
        "related_to",
        "--type",
        help=f"Edge type: {', '.join(EDGE_TYPES)}",
    ),
    confidence: float = typer.Option(
        0.35, "--conf", help="Suggested confidence (capped on approve)"
    ),
    episodes: Optional[str] = typer.Option(
        None,
        "--episodes",
        "-e",
        help="Comma-separated episode ids as provenance for this edge",
    ),
    note: Optional[str] = typer.Option(None, "--note"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Enqueue a pending edge proposal (does NOT write the edge yet)."""
    manager = _manager(data_dir)
    eps = [p.strip() for p in episodes.split(",")] if episodes else []
    try:
        prop = manager.approval.enqueue_edge(
            source,
            target,
            edge_type,
            confidence=confidence,
            supporting_episode_ids=eps,
            note=note,
            source="manual",
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]{prop.proposal_id}[/bold]\n"
            f"{prop.label}\n"
            f"confidence={prop.confidence:.2f}  status=pending\n"
            f"Next: grmc approve {prop.proposal_id}",
            title="Edge proposal enqueued (no graph write yet)",
            border_style="yellow",
        )
    )


@edges_app.command("types")
def edges_types():
    """Show allowed edge types."""
    for t in EDGE_TYPES:
        console.print(f"  • {t}")
