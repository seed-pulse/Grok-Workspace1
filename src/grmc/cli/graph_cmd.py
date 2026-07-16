"""Read-only graph navigation CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.graph_query import find_paths, neighborhood
from ..storage.sqlite_store import SQLiteStore

graph_app = typer.Typer(
    help="Read-only graph queries (no writes).",
    no_args_is_help=True,
)
console = Console()
DEFAULT_DATA = "./grmc_data"


def _db(data_dir: str) -> SQLiteStore:
    return SQLiteStore(Path(data_dir) / "grmc.db")


@graph_app.command("neighbors")
def neighbors_cmd(
    node_id: str = typer.Argument(..., help="Root node id"),
    depth: int = typer.Option(
        1, "--depth", "-d", min=1, max=2, help="Hop depth (1 or 2)"
    ),
    edge_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Filter edge type: supports | contradicts | related_to | ...",
    ),
    no_provenance: bool = typer.Option(
        False, "--no-provenance", help="Skip root provenance listing"
    ),
    json_out: Optional[Path] = typer.Option(None, "--json", "-o"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Show neighborhood of a node (depth 1–2) and optional provenance."""
    db = _db(data_dir)
    try:
        result = neighborhood(
            db,
            node_id,
            depth=depth,
            edge_type=edge_type,
            include_provenance=not no_provenance,
        )
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]{result.root.node_id}[/bold]  {result.root.label}\n"
            f"type={result.root.type}  conf={result.root.confidence:.2f}\n"
            f"neighbors={len(result.hops)}  depth<={depth}"
            + (f"  filter={edge_type}" if edge_type else ""),
            title="Graph neighborhood (read-only)",
            border_style="cyan",
        )
    )

    if result.hops:
        table = Table(title="Neighbors")
        table.add_column("Depth", justify="right")
        table.add_column("Dir")
        table.add_column("Edge type")
        table.add_column("EConf", justify="right")
        table.add_column("Node ID", style="cyan")
        table.add_column("Label")
        table.add_column("NConf", justify="right")
        for h in result.hops:
            table.add_row(
                str(h.depth),
                h.direction,
                h.via_edge_type,
                f"{h.edge_confidence:.2f}",
                h.node.node_id,
                h.node.label[:36],
                f"{h.node.confidence:.2f}",
            )
        console.print(table)
    else:
        console.print("[dim]No neighbors at this depth/filter.[/dim]")

    if result.provenance:
        ptable = Table(title="Root provenance (episode → node)")
        ptable.add_column("Episode")
        ptable.add_column("Rel")
        ptable.add_column("Conf", justify="right")
        ptable.add_column("Summary")
        for p in result.provenance:
            ptable.add_row(
                p["episode_id"],
                p["relation"],
                f"{p['confidence']:.2f}",
                (p.get("summary") or "")[:50],
            )
        console.print(ptable)

    if json_out:
        json_out.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] wrote {json_out}")


@graph_app.command("path")
def path_cmd(
    node_a: str = typer.Argument(..., help="Source node id"),
    node_b: str = typer.Argument(..., help="Target node id"),
    max_depth: int = typer.Option(
        3, "--depth", "-d", min=1, max=3, help="Max hop length (1–3)"
    ),
    edge_type: Optional[str] = typer.Option(
        None, "--type", help="Filter edge type along the path"
    ),
    max_paths: int = typer.Option(
        3, "--max-paths", "-k", min=1, max=10, help="Max shortest paths to show"
    ),
    no_provenance: bool = typer.Option(
        False, "--no-provenance", help="Skip per-node provenance on path"
    ),
    json_out: Optional[Path] = typer.Option(None, "--json", "-o"),
    data_dir: str = typer.Option(DEFAULT_DATA, "--data-dir"),
):
    """Find shortest path(s) between two nodes (max 3 hops). Read-only."""
    db = _db(data_dir)
    try:
        result = find_paths(
            db,
            node_a,
            node_b,
            max_depth=max_depth,
            edge_type=edge_type,
            max_paths=max_paths,
            include_provenance=not no_provenance,
        )
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]{result.source.label}[/bold] ({result.source.node_id})\n"
            f"    →  [bold]{result.target.label}[/bold] ({result.target.node_id})\n"
            f"found={result.found}  max_depth={result.max_depth}  "
            f"paths={len(result.paths)}"
            + (f"  filter={edge_type}" if edge_type else ""),
            title="Graph path (read-only)",
            border_style="green" if result.found else "yellow",
        )
    )

    if not result.found:
        console.print("[yellow]No path within depth limit.[/yellow]")
    else:
        for i, path in enumerate(result.paths, 1):
            console.print(f"\n[bold]Path {i}[/bold] (length={path.length})")
            console.print(f"  {path.describe()}")
            if path.steps:
                table = Table(show_header=True, title=f"Steps (path {i})")
                table.add_column("#", justify="right")
                table.add_column("From")
                table.add_column("Type")
                table.add_column("Dir")
                table.add_column("To")
                table.add_column("EConf", justify="right")
                for j, s in enumerate(path.steps, 1):
                    table.add_row(
                        str(j),
                        (s.from_label or s.from_node_id)[:24],
                        s.edge_type,
                        s.direction,
                        (s.to_label or s.to_node_id)[:24],
                        f"{s.edge_confidence:.2f}",
                    )
                console.print(table)

    if result.provenance:
        console.print("\n[bold]Provenance on path nodes[/bold]")
        for nid, links in result.provenance.items():
            node = db.get_graph_node(nid)
            label = node.label if node else nid
            if not links:
                console.print(f"  [dim]{label}: (no episode links)[/dim]")
                continue
            for link in links[:3]:
                console.print(
                    f"  {label}: {link['episode_id']} [{link['relation']}] "
                    f"conf={link['confidence']:.2f} — {(link.get('summary') or '')[:40]}"
                )

    if json_out:
        json_out.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] wrote {json_out}")
