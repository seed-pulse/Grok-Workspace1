"""GRMC CLI — memory, reflect, propose/approve, bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.memory_manager import MemoryManager
from ..models.episode import Episode
from ..storage.sqlite_store import SQLiteStore
from .bridge_cmd import bridge_app
from .edges_cmd import edges_app
from .graph_cmd import graph_app
from .ops_cmd import ops_app

app = typer.Typer(
    help="GRMC - Grok Reflective Memory Core (think via reflect, write via approve)",
    no_args_is_help=True,
)
app.add_typer(bridge_app, name="bridge")
app.add_typer(edges_app, name="edges")
app.add_typer(graph_app, name="graph")
app.add_typer(ops_app, name="ops")
console = Console()

DEFAULT_DATA_DIR = "./grmc_data"


def _manager(
    data_dir: str = DEFAULT_DATA_DIR,
    embedder: str = "auto",
) -> MemoryManager:
    return MemoryManager.from_data_dir(data_dir, embedder_prefer=embedder)


def _sqlite(data_dir: str = DEFAULT_DATA_DIR) -> SQLiteStore:
    return SQLiteStore(Path(data_dir) / "grmc.db")


@app.command()
def ingest(
    text: str = typer.Option(..., "--text", "-t", help="Content to ingest as memory"),
    source: str = typer.Option("cli", "--source", "-s", help="Source of this memory"),
    conversation_id: Optional[str] = typer.Option(None, "--conv", help="Conversation ID"),
    concepts: Optional[str] = typer.Option(
        None,
        "--concepts",
        "-c",
        help="Comma-separated concept labels to attach (optional)",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="Embedding backend: auto | sentence-transformers | hashing",
    ),
):
    """Ingest a new episode (SQLite SoR + Chroma vector)."""
    manager = _manager(data_dir, embedder=embedder)
    extracted = [p.strip() for p in concepts.split(",")] if concepts else []

    episode = Episode(
        content_summary=text,
        raw_content=text,
        source=source,
        conversation_id=conversation_id,
        importance_score=0.7,
        extracted_concepts=extracted,
    )

    episode_id = manager.ingest_episode(episode)
    console.print(f"[green]✓[/green] Ingested episode: {episode_id}")
    if extracted:
        console.print(f"  concepts: {extracted}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="What to search for in memory"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="Embedding backend: auto | sentence-transformers | hashing",
    ),
):
    """Retrieve relevant memories using semantic search (Chroma)."""
    manager = _manager(data_dir, embedder=embedder)
    results = manager.retrieve(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant memories found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} Memories for: {query}")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Summary", style="white")
    table.add_column("Score", style="green")

    for r in results:
        summary = r.get("summary") or r.get("content_summary") or ""
        if len(summary) > 80:
            summary = summary[:80] + "..."
        dist = r.get("distance")
        score = 1 - dist if dist is not None else 0.0
        table.add_row(r["episode_id"], summary, f"{score:.3f}")

    console.print(table)


@app.command("list")
def list_cmd(
    limit: int = typer.Option(10, "--limit", "-n", help="Max episodes to show"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
):
    """List recent episodes from SQLite (timestamp index)."""
    episodes = _sqlite(data_dir).list_recent(limit=limit)

    if not episodes:
        console.print("[yellow]No episodes stored yet.[/yellow]")
        return

    table = Table(title=f"Recent episodes (SQLite, up to {limit})")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Source", style="magenta")
    table.add_column("Summary", style="white")

    for ep in episodes:
        summary = ep.get("summary") or ""
        if len(summary) > 70:
            summary = summary[:70] + "..."
        table.add_row(
            ep.get("episode_id", ""),
            str(ep.get("timestamp") or (ep.get("metadata") or {}).get("timestamp", "")),
            str(ep.get("source") or (ep.get("metadata") or {}).get("source", "")),
            summary,
        )

    console.print(table)
    console.print("[dim]Source of truth: SQLite episodes table (indexed by timestamp).[/dim]")


@app.command()
def status(
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
):
    """Show SQLite stats, pending proposals, and last reflection."""
    db = _sqlite(data_dir)
    stats = db.stats()
    console.print("[bold]GRMC Memory Status[/bold]")
    console.print(f"Storage: {Path(data_dir).resolve()}")
    console.print(f"SQLite:  {stats['db_path']}")
    console.print(f"Episodes: [cyan]{stats['episodes']}[/cyan]")
    console.print(
        f"Proposals: pending=[yellow]{stats['proposals_pending']}[/yellow] "
        f"total={stats['proposals_total']}"
    )
    console.print(
        f"Graph nodes: [cyan]{stats['graph_nodes']}[/cyan]  "
        f"edges: [cyan]{stats.get('graph_edges', 0)}[/cyan]  "
        f"provenance links: [cyan]{stats.get('episode_node_links', 0)}[/cyan]"
    )
    console.print(
        f"Reflection reports (SQLite): {stats['reflections']}  "
        f"schema_v={stats.get('schema_version', '?')}"
    )

    try:
        from ..core.embedder import create_embedder

        emb = create_embedder("hashing")  # avoid broken torch on status
        console.print(f"Embedder probe (hashing): [cyan]{emb.name}[/cyan]")
    except Exception as exc:  # pragma: no cover
        console.print(f"[yellow]Embedder probe failed: {exc}[/yellow]")

    latest = db.latest_reflection()
    if latest:
        console.print("[bold]Last reflection (SQLite)[/bold]")
        console.print(f"  report_id: {latest.get('report_id')}")
        console.print(f"  timestamp: {latest.get('timestamp')}")
        console.print(f"  mode: {latest.get('mode')}  episodes: {latest.get('episodes_analyzed')}")
        console.print(
            f"  mutates_memory: {bool(latest.get('mutates_memory'))} "
            "[dim](should always be false)[/dim]"
        )
    else:
        file_latest = Path(data_dir) / "reflections" / "latest.json"
        if file_latest.exists():
            try:
                info = json.loads(file_latest.read_text(encoding="utf-8"))
                console.print("[bold]Last reflection (file pointer)[/bold]")
                console.print(f"  report_id: {info.get('report_id')}")
            except (json.JSONDecodeError, OSError):
                pass
        else:
            console.print("[dim]No reflection reports yet. Run: grmc reflect[/dim]")

    if stats["proposals_pending"]:
        console.print(
            f"[dim]Review proposals: grmc propose  "
            f"({stats['proposals_pending']} pending)[/dim]"
        )


@app.command()
def reflect(
    recent: bool = typer.Option(
        False,
        "--recent",
        help="Analyze recent episodes (default when --topic omitted).",
    ),
    limit: int = typer.Option(
        30,
        "--limit",
        "-n",
        help="Max recent episodes (SQLite; ignored if --topic)",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Topic mode: semantic retrieve via Chroma",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Also write full report JSON"
    ),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Skip JSON file audit under reflections/"
    ),
    no_propose: bool = typer.Option(
        False,
        "--no-propose",
        help="Do not enqueue concept candidates as pending proposals",
    ),
    no_edge_propose: bool = typer.Option(
        False,
        "--no-edge-propose",
        help="Do not enqueue soft edge suggestions as pending edge proposals",
    ),
    llm: Optional[bool] = typer.Option(
        None,
        "--llm/--no-llm",
        help="LLM verification: --llm on, --no-llm off, omit=env GRMC_LLM (default off)",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    embedder: str = typer.Option("auto", "--embedder"),
):
    """Reflect (think only). Optionally enqueue proposals — never writes graph."""
    if recent and topic:
        console.print(
            "[yellow]Both --recent and --topic given; using topic mode.[/yellow]"
        )

    manager = _manager(data_dir, embedder=embedder)

    with console.status("[bold]Reflecting (report-only)...[/bold]"):
        report = manager.reflect(
            recent_limit=limit,
            topic=topic,
            persist=not no_persist,
            enqueue_proposals=not no_propose,
            enqueue_edge_suggestions=not no_edge_propose,
            llm=llm,
        )

    title = "Reflection Report (non-mutating)"
    if topic:
        title += f" — topic: {topic}"

    embedder_name = getattr(manager.embedder, "name", "unknown")
    enqueued = report.metadata.get("proposals_enqueued", 0)
    edge_enq = report.metadata.get("edge_proposals_enqueued", 0)
    llm_meta = report.metadata.get("llm") or {}
    llm_state = "on" if llm_meta.get("enabled") else "off"
    console.print(
        Panel.fit(
            f"[bold]{report.report_id}[/bold]\n"
            f"mode={report.mode}  episodes={report.episodes_analyzed}  "
            f"confidence={report.confidence_level}  mutates_memory={report.mutates_memory}\n"
            f"embedder={embedder_name}  llm={llm_state}  "
            f"concept_proposals={enqueued}  edge_proposals={edge_enq}",
            title=title,
            border_style="cyan",
        )
    )

    if report.concept_candidates:
        table = Table(title="Concept candidates → pending proposals (if enqueued)")
        table.add_column("Label", style="cyan")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right", style="green")
        table.add_column("Source")
        for c in report.concept_candidates[:15]:
            table.add_row(c.label, str(c.frequency), f"{c.confidence:.2f}", c.source)
        console.print(table)
    else:
        console.print("[yellow]No concept candidates extracted.[/yellow]")

    if report.potential_contradictions:
        console.print(
            f"\n[bold yellow]Potential tensions "
            f"({len(report.potential_contradictions)}) — human review[/bold yellow]"
        )
        for i, flag in enumerate(report.potential_contradictions[:10], 1):
            method = getattr(flag, "method", "?")
            sim = getattr(flag, "similarity", None)
            sim_s = f" sim={sim:.3f}" if sim is not None else ""
            console.print(
                f"  {i}. [{method}] conf={flag.confidence:.2f}{sim_s}  "
                f"{flag.episode_id_a} ↔ {flag.episode_id_b}"
            )
            console.print(f"     reason: {flag.reason}")
    else:
        console.print(
            "\n[dim]No contradiction heuristic fired (not proof of consistency).[/dim]"
        )

    if report.edge_suggestions:
        console.print(
            f"\n[bold]Soft edge suggestions ({len(report.edge_suggestions)}) "
            "— not written[/bold]"
        )
        for s in report.edge_suggestions[:10]:
            console.print(
                f"  • {s.source_label} -[{s.edge_type}]-> {s.target_label} "
                f"conf={s.confidence:.2f}"
            )

    if report.suggested_actions:
        console.print("\n[bold]Suggested actions (manual)[/bold]")
        for a in report.suggested_actions:
            console.print(f"  • {a}")

    if report.limitations:
        console.print("\n[bold]Known limitations[/bold]")
        for lim in report.limitations:
            console.print(f"  • [dim]{lim}[/dim]")

    if report.notes:
        console.print("\n[bold]Notes[/bold]")
        for n in report.notes:
            console.print(f"  • {n}")

    report_path = report.metadata.get("report_path")
    if report_path:
        console.print(f"\n[green]✓[/green] Audit report: {report_path}")
    if enqueued or edge_enq:
        console.print(
            f"[green]✓[/green] Enqueued concept={enqueued} edge={edge_enq}. "
            "Next: [bold]grmc propose[/bold] / [bold]grmc propose --kind edge[/bold]"
        )

    if output:
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] Wrote copy to: {output}")


@app.command()
def propose(
    status: str = typer.Option(
        "pending",
        "--status",
        "-s",
        help="Filter: pending | approved | rejected | all",
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        "-k",
        help="Filter: concept_candidate | manual | edge",
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    label: Optional[str] = typer.Option(
        None,
        "--add",
        help="Manually add a pending concept proposal with this label (no graph write)",
    ),
):
    """List approval-queue proposals, or manually add one with --add."""
    manager = _manager(data_dir, embedder="hashing")

    if label:
        created = manager.approval.enqueue_many_labels([label], source="manual")
        if created:
            console.print(f"[green]✓[/green] Pending proposal: {created[0].proposal_id} ({label})")
        else:
            console.print(
                f"[yellow]Skipped (already pending or graph node exists): {label}[/yellow]"
            )

    filter_status = None if status == "all" else status
    items = manager.approval.list(status=filter_status, limit=limit, kind=kind)
    if not items:
        console.print(
            f"[dim]No proposals with status={status!r}"
            + (f" kind={kind!r}" if kind else "")
            + ".[/dim]"
        )
        return

    table = Table(title=f"Proposals ({status}" + (f", {kind}" if kind else "") + ")")
    table.add_column("ID", style="cyan")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Label")
    table.add_column("Conf", justify="right")
    table.add_column("Source")
    for p in items:
        table.add_row(
            p.proposal_id,
            p.kind,
            p.status,
            p.label[:48],
            f"{p.confidence:.2f}",
            p.source,
        )
    console.print(table)
    console.print(
        "[dim]Approve writes graph: grmc approve <id>  ·  Reject: grmc reject <id>[/dim]"
    )


@app.command()
def approve(
    proposal_id: str = typer.Argument(..., help="Proposal id (prop_...)"),
    note: Optional[str] = typer.Option(None, "--note", help="Optional review note"),
    confidence_cap: Optional[float] = typer.Option(
        None,
        "--cap",
        help="Max confidence (default 0.55 nodes / 0.45 edges)",
    ),
    node_type: str = typer.Option(
        "concept",
        "--type",
        help="Graph node type: concept | belief | fact | user_model | self_model",
    ),
    link_to: Optional[str] = typer.Option(
        None,
        "--link-to",
        help="After node approve, enqueue a *pending* related_to edge to this node id",
    ),
    link_type: str = typer.Option(
        "related_to",
        "--link-type",
        help="Edge type used with --link-to",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Approve a pending proposal → write GraphNode or GraphEdge (+ provenance)."""
    manager = _manager(data_dir, embedder="hashing")
    try:
        result = manager.approval.approve(
            proposal_id,
            node_type=node_type,  # type: ignore[arg-type]
            confidence_cap=confidence_cap,
            note=note,
            also_link_related=bool(link_to),
            related_to_node_id=link_to,
            related_edge_type=link_type,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if result["kind"] == "edge":
        edge = result["edge"]
        console.print(
            Panel.fit(
                f"[bold]{edge.edge_id}[/bold]\n"
                f"{edge.source_node_id} -[{edge.edge_type}]-> {edge.target_node_id}\n"
                f"confidence={edge.confidence:.2f}\n"
                f"episodes={len(edge.supporting_episode_ids)}",
                title="GraphEdge written (human approved)",
                border_style="green",
            )
        )
        return

    node = result["node"]
    n_links = len(result.get("provenance_links") or [])
    console.print(
        Panel.fit(
            f"[bold]{node.node_id}[/bold]\n"
            f"label={node.label}\n"
            f"type={node.type}  confidence={node.confidence:.2f}  version={node.version}\n"
            f"supporting_episodes={len(node.supporting_episodes)}  "
            f"provenance_links_written={n_links}",
            title="GraphNode written (human approved)",
            border_style="green",
        )
    )
    related = result.get("related_edge_proposal")
    if related:
        console.print(
            f"[yellow]Pending edge proposal enqueued:[/yellow] {related.proposal_id}\n"
            f"  {related.label}\n"
            f"  Approve with: grmc approve {related.proposal_id}"
        )


@app.command()
def reject(
    proposal_id: str = typer.Argument(..., help="Proposal id (prop_...)"),
    note: Optional[str] = typer.Option(None, "--note", help="Optional reason"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Reject a pending proposal (no graph write)."""
    manager = _manager(data_dir, embedder="hashing")
    try:
        prop = manager.approval.reject(proposal_id, note=note)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Rejected {prop.proposal_id} ({prop.label})")


@app.command("nodes")
def nodes_cmd(
    limit: int = typer.Option(50, "--limit", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List approved GraphNodes (written only via approve)."""
    items = _sqlite(data_dir).list_graph_nodes(limit=limit)
    if not items:
        console.print("[dim]No graph nodes yet. Approve a proposal first.[/dim]")
        return
    table = Table(title="Graph nodes")
    table.add_column("Node ID", style="cyan")
    table.add_column("Type")
    table.add_column("Label")
    table.add_column("Conf", justify="right")
    table.add_column("Ver", justify="right")
    table.add_column("Supports", justify="right")
    for n in items:
        table.add_row(
            n.node_id,
            n.type,
            n.label[:40],
            f"{n.confidence:.2f}",
            str(n.version),
            str(len(n.supporting_episodes)),
        )
    console.print(table)
    console.print("[dim]Detail: grmc node <id> --with-edges[/dim]")


@app.command("node")
def node_cmd(
    node_id: str = typer.Argument(..., help="Node id (node_...)"),
    with_edges: bool = typer.Option(
        True,
        "--with-edges/--no-edges",
        help="Show incident edges",
    ),
    with_provenance: bool = typer.Option(
        True,
        "--with-provenance/--no-provenance",
        help="Show episode↔node provenance links",
    ),
    provenance: bool = typer.Option(
        False,
        "--provenance",
        help="Alias to force-show provenance (same as --with-provenance)",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Show one GraphNode with optional edges and provenance ('why this node?').

    Examples:
      grmc node node_abc --provenance
      grmc node node_abc --with-edges --with-provenance
    """
    show_provenance = with_provenance or provenance
    db = _sqlite(data_dir)
    node = db.get_graph_node(node_id)
    if node is None:
        console.print(f"[red]Unknown node: {node_id}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]{node.node_id}[/bold]\n"
            f"label={node.label}\n"
            f"type={node.type}  confidence={node.confidence:.2f}  version={node.version}\n"
            f"supporting_episodes={node.supporting_episodes}\n"
            f"metadata={node.metadata}",
            title="GraphNode",
            border_style="cyan",
        )
    )

    if show_provenance:
        links = db.list_links_for_node(node_id)
        if not links:
            console.print("[dim]No provenance links recorded for this node.[/dim]")
        else:
            table = Table(title="Provenance (episode → node)")
            table.add_column("Link ID", style="cyan")
            table.add_column("Episode")
            table.add_column("Relation")
            table.add_column("Conf", justify="right")
            table.add_column("Proposal")
            table.add_column("Episode summary")
            for link in links:
                ep = db.get_episode(link.episode_id)
                summary = (ep.get("summary") if ep else "") or ""
                if len(summary) > 50:
                    summary = summary[:50] + "…"
                table.add_row(
                    link.link_id,
                    link.episode_id,
                    link.relation,
                    f"{link.confidence:.2f}",
                    (link.proposal_id or "")[:14],
                    summary,
                )
            console.print(table)

    if with_edges:
        edges = db.list_graph_edges(node_id=node_id, limit=50)
        if not edges:
            console.print("[dim]No edges touch this node.[/dim]")
        else:
            table = Table(title="Incident edges")
            table.add_column("Edge ID", style="cyan")
            table.add_column("Direction")
            table.add_column("Type")
            table.add_column("Other")
            table.add_column("Conf", justify="right")
            for e in edges:
                if e.source_node_id == node_id:
                    direction = "out"
                    other_id = e.target_node_id
                else:
                    direction = "in"
                    other_id = e.source_node_id
                other = db.get_graph_node(other_id)
                table.add_row(
                    e.edge_id,
                    direction,
                    e.edge_type,
                    (other.label if other else other_id)[:30],
                    f"{e.confidence:.2f}",
                )
            console.print(table)


if __name__ == "__main__":
    app()
