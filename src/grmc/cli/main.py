"""GRMC CLI — memory, reflect, propose/approve, graph, ops, bridge."""

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
    help=(
        "GRMC — Grok Reflective Memory Core.\n\n"
        "Loop: ingest → reflect (think) → propose → approve (write) → inspect.\n"
        "Reflection never writes the knowledge graph; only approve does."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
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
    text: str = typer.Option(..., "--text", "-t", help="Content to store as an episode"),
    source: str = typer.Option("cli", "--source", "-s", help="Origin label"),
    conversation_id: Optional[str] = typer.Option(None, "--conv", help="Conversation id"),
    concepts: Optional[str] = typer.Option(
        None, "--concepts", "-c", help="Comma-separated concept labels"
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Data directory"),
    embedder: str = typer.Option(
        "auto", "--embedder", help="auto | sentence-transformers | hashing"
    ),
):
    """Store a new episode (SQLite system-of-record + Chroma vector)."""
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
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max results"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    embedder: str = typer.Option("auto", "--embedder"),
):
    """Semantic search over episodes (Chroma vectors)."""
    manager = _manager(data_dir, embedder=embedder)
    results = manager.retrieve(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant memories found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} for: {query}")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Summary")
    table.add_column("Score", style="green", justify="right")

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
    limit: int = typer.Option(10, "--limit", "-n", help="Max rows"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List recent episodes (SQLite timestamp index)."""
    episodes = _sqlite(data_dir).list_recent(limit=limit)

    if not episodes:
        console.print("[yellow]No episodes stored yet.[/yellow]")
        return

    table = Table(title=f"Recent episodes (up to {limit})")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Source", style="magenta")
    table.add_column("Summary")

    for ep in episodes:
        summary = ep.get("summary") or ""
        if len(summary) > 70:
            summary = summary[:70] + "..."
        table.add_row(
            ep.get("episode_id", ""),
            str(ep.get("timestamp") or "")[:19],
            str(ep.get("source") or ""),
            summary,
        )

    console.print(table)
    console.print("[dim]SoR: SQLite episodes (not Chroma ordering).[/dim]")


@app.command()
def status(
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Dashboard: counts, last reflection, pending queue, LLM audit snapshot."""
    data_path = Path(data_dir).resolve()
    db = _sqlite(data_dir)
    stats = db.stats()

    # Pending breakdown
    pending = db.list_proposals(status="pending", limit=500)
    n_concept = sum(1 for p in pending if p.kind in ("concept_candidate", "manual"))
    n_edge = sum(1 for p in pending if p.kind == "edge")

    console.print(
        Panel.fit(
            f"[bold]GRMC v0.8[/bold]  ·  think ≠ write  ·  approve-only graph\n"
            f"data: [cyan]{data_path}[/cyan]\n"
            f"sqlite: {stats['db_path']}",
            title="Status",
            border_style="cyan",
        )
    )

    counts = Table(title="Memory & graph", show_header=True)
    counts.add_column("Metric")
    counts.add_column("Value", justify="right", style="cyan")
    counts.add_row("Episodes", str(stats.get("episodes", 0)))
    counts.add_row("Graph nodes", str(stats.get("graph_nodes", 0)))
    counts.add_row("Graph edges", str(stats.get("graph_edges", 0)))
    counts.add_row("Provenance links", str(stats.get("episode_node_links", 0)))
    counts.add_row("Reflection reports", str(stats.get("reflections", 0)))
    counts.add_row("Schema version", str(stats.get("schema_version", "?")))
    console.print(counts)

    prop_table = Table(title="Approval queue", show_header=True)
    prop_table.add_column("State")
    prop_table.add_column("Count", justify="right")
    prop_table.add_row("Pending (all)", str(stats.get("proposals_pending", 0)))
    prop_table.add_row("  · concepts/manual", str(n_concept))
    prop_table.add_row("  · edges", str(n_edge))
    prop_table.add_row("Total proposals ever", str(stats.get("proposals_total", 0)))
    console.print(prop_table)

    if pending:
        preview = Table(title="Pending preview (up to 5)")
        preview.add_column("ID", style="yellow")
        preview.add_column("Kind")
        preview.add_column("Label")
        preview.add_column("Conf", justify="right")
        for p in pending[:5]:
            preview.add_row(
                p.proposal_id,
                p.kind,
                p.label[:40],
                f"{p.confidence:.2f}",
            )
        console.print(preview)
        console.print("[dim]Next: grmc propose  ·  grmc approve <id>  ·  grmc reject <id>[/dim]")

    latest = db.latest_reflection()
    if latest:
        mm = bool(latest.get("mutates_memory"))
        mm_style = "red" if mm else "green"
        console.print(
            Panel.fit(
                f"report_id: [cyan]{latest.get('report_id')}[/cyan]\n"
                f"timestamp: {latest.get('timestamp')}\n"
                f"mode: {latest.get('mode')}  episodes_analyzed: {latest.get('episodes_analyzed')}\n"
                f"mutates_memory: [{mm_style}]{mm}[/{mm_style}] "
                f"[dim](must be False)[/dim]",
                title="Last reflection",
                border_style="green" if not mm else "red",
            )
        )
    else:
        console.print("[dim]No reflections yet — run: grmc reflect[/dim]")

    # LLM audit
    try:
        from ..llm.audit import LLMAuditLog

        llm_sum = LLMAuditLog(data_dir).summary()
        console.print(
            Panel.fit(
                f"calls={llm_sum['calls']}  success={llm_sum['success']}  "
                f"failed={llm_sum['failed']}\n"
                f"total_tokens_est={llm_sum['total_tokens_est']}  "
                f"by_purpose={llm_sum['by_purpose'] or '{}'}\n"
                f"[dim]{llm_sum['path']}[/dim]\n"
                f"[dim]LLM default OFF — enable with GRMC_LLM=1 or reflect --llm[/dim]",
                title="LLM audit",
                border_style="dim",
            )
        )
    except Exception as exc:
        console.print(f"[dim]LLM audit unavailable: {exc}[/dim]")

    console.print(
        "[dim]Docs: docs/QUICKSTART.md · docs/DESIGN_PRINCIPLES.md · docs/HANDOVER.md[/dim]"
    )


@app.command()
def reflect(
    recent: bool = typer.Option(
        False, "--recent", help="Recent mode (default when --topic omitted)"
    ),
    limit: int = typer.Option(30, "--limit", "-n", help="Max recent episodes"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Semantic topic mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report JSON"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip reflections/ JSON file"),
    no_propose: bool = typer.Option(
        False, "--no-propose", help="Do not enqueue concept proposals"
    ),
    no_edge_propose: bool = typer.Option(
        False, "--no-edge-propose", help="Do not enqueue soft edge proposals"
    ),
    llm: Optional[bool] = typer.Option(
        None,
        "--llm/--no-llm",
        help="LLM verification: --llm on · --no-llm off · omit=env (default off)",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    embedder: str = typer.Option("auto", "--embedder"),
):
    """Think only: build a reflection report. Never writes the knowledge graph."""
    if recent and topic:
        console.print("[yellow]Both --recent and --topic: using topic mode.[/yellow]")

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
            f"confidence={report.confidence_level}  "
            f"mutates_memory=[green]{report.mutates_memory}[/green]\n"
            f"embedder={embedder_name}  llm={llm_state}  "
            f"concept_proposals={enqueued}  edge_proposals={edge_enq}",
            title=title,
            border_style="cyan",
        )
    )

    if report.concept_candidates:
        table = Table(title="Concept candidates (not graph nodes yet)")
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
            console.print(f"     {flag.reason}")
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
            "Next: [bold]grmc propose[/bold]"
        )

    if output:
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] Wrote copy to: {output}")


@app.command()
def propose(
    status: str = typer.Option(
        "pending", "--status", "-s", help="pending | approved | rejected | all"
    ),
    kind: Optional[str] = typer.Option(
        None, "--kind", "-k", help="concept_candidate | manual | edge"
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    label: Optional[str] = typer.Option(
        None, "--add", help="Manually queue a concept label (still needs approve)"
    ),
):
    """List the approval queue, or add a manual concept proposal."""
    manager = _manager(data_dir, embedder="hashing")

    if label:
        created = manager.approval.enqueue_many_labels([label], source="manual")
        if created:
            console.print(
                f"[green]✓[/green] Pending: {created[0].proposal_id} ({label})"
            )
        else:
            console.print(
                f"[yellow]Skipped (already pending or node exists): {label}[/yellow]"
            )

    filter_status = None if status == "all" else status
    items = manager.approval.list(status=filter_status, limit=limit, kind=kind)
    if not items:
        console.print(
            f"[dim]No proposals status={status!r}"
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
        "[dim]Write graph: grmc approve <id>  ·  Dismiss: grmc reject <id>[/dim]"
    )


@app.command()
def approve(
    proposal_id: str = typer.Argument(..., help="Proposal id (prop_...)"),
    note: Optional[str] = typer.Option(None, "--note", help="Review note"),
    confidence_cap: Optional[float] = typer.Option(
        None, "--cap", help="Max conf (default 0.55 node / 0.45 edge)"
    ),
    node_type: str = typer.Option(
        "concept",
        "--type",
        help="Node type if concept: concept|belief|fact|user_model|self_model",
    ),
    link_to: Optional[str] = typer.Option(
        None, "--link-to", help="Also enqueue pending related edge to this node"
    ),
    link_type: str = typer.Option("related_to", "--link-type"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Human gate: write GraphNode or GraphEdge (+ provenance). Only write path."""
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
                title="GraphEdge written (approved)",
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
            f"type={node.type}  confidence={node.confidence:.2f}  v{node.version}\n"
            f"supporting_episodes={len(node.supporting_episodes)}  "
            f"provenance_links={n_links}",
            title="GraphNode written (approved)",
            border_style="green",
        )
    )
    related = result.get("related_edge_proposal")
    if related:
        console.print(
            f"[yellow]Pending edge proposal:[/yellow] {related.proposal_id}\n"
            f"  {related.label}\n"
            f"  Approve: grmc approve {related.proposal_id}"
        )


@app.command()
def reject(
    proposal_id: str = typer.Argument(..., help="Proposal id"),
    note: Optional[str] = typer.Option(None, "--note"),
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
    """List approved GraphNodes."""
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
    console.print("[dim]Detail: grmc node <id> --provenance[/dim]")


@app.command("node")
def node_cmd(
    node_id: str = typer.Argument(..., help="Node id (node_...)"),
    with_edges: bool = typer.Option(True, "--with-edges/--no-edges"),
    with_provenance: bool = typer.Option(
        True, "--with-provenance/--no-provenance", help="Show episode links"
    ),
    provenance: bool = typer.Option(
        False, "--provenance", help="Force-show provenance"
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Show one node with edges and provenance (why does this exist?)."""
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
            f"supporting_episodes={node.supporting_episodes}",
            title="GraphNode",
            border_style="cyan",
        )
    )

    if show_provenance:
        links = db.list_links_for_node(node_id)
        if not links:
            console.print("[dim]No provenance links for this node.[/dim]")
        else:
            table = Table(title="Provenance (episode → node)")
            table.add_column("Link ID", style="cyan")
            table.add_column("Episode")
            table.add_column("Relation")
            table.add_column("Conf", justify="right")
            table.add_column("Summary")
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
            table.add_column("Dir")
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
