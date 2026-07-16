"""GRMC CLI — ingest, retrieve, list, status, reflect."""

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
from ..storage.chroma_store import ChromaMemoryStore
from .bridge_cmd import bridge_app

app = typer.Typer(
    help="GRMC - Grok Reflective Memory Core CLI (memory + reflect + bridge)",
    no_args_is_help=True,
)
app.add_typer(bridge_app, name="bridge")
console = Console()

DEFAULT_DATA_DIR = "./grmc_data"


def _manager(
    data_dir: str = DEFAULT_DATA_DIR,
    embedder: str = "auto",
) -> MemoryManager:
    store = ChromaMemoryStore(persist_directory=data_dir)
    return MemoryManager(store, embedder_prefer=embedder)


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
    """Ingest a new piece of memory (episode + embedding)."""
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
    """Retrieve relevant memories using semantic search."""
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
        summary = r["summary"]
        if len(summary) > 80:
            summary = summary[:80] + "..."
        score = 1 - r["distance"] if r.get("distance") is not None else 0.0
        table.add_row(r["episode_id"], summary, f"{score:.3f}")

    console.print(table)


@app.command("list")
def list_cmd(
    limit: int = typer.Option(10, "--limit", "-n", help="Max episodes to show"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
):
    """List recent episodes (sorted by metadata timestamp, client-side)."""
    store = ChromaMemoryStore(persist_directory=data_dir)
    episodes = store.list_recent(limit=limit)

    if not episodes:
        console.print("[yellow]No episodes stored yet.[/yellow]")
        return

    table = Table(title=f"Recent episodes (up to {limit})")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Source", style="magenta")
    table.add_column("Summary", style="white")

    for ep in episodes:
        meta = ep.get("metadata") or {}
        summary = ep.get("summary") or ""
        if len(summary) > 70:
            summary = summary[:70] + "..."
        table.add_row(
            ep.get("episode_id", ""),
            str(meta.get("timestamp", "")),
            str(meta.get("source", "")),
            summary,
        )

    console.print(table)
    console.print(
        "[dim]Note: ordering is client-side sort on metadata.timestamp "
        "(Chroma has no native chronological index).[/dim]"
    )


@app.command()
def status(
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
):
    """Show current memory status and last reflection pointer (if any)."""
    store = ChromaMemoryStore(persist_directory=data_dir)
    count = store.count()
    console.print("[bold]GRMC Memory Status[/bold]")
    console.print(f"Total episodes stored: [cyan]{count}[/cyan]")
    console.print(f"Storage location: {Path(data_dir).resolve()}")
    # Prefer hashing probe for status so a broken torch install cannot block diagnostics
    try:
        from ..core.embedder import create_embedder

        emb = create_embedder("auto")
        console.print(f"Embedder (auto): [cyan]{emb.name}[/cyan]")
    except Exception as exc:  # pragma: no cover - env diagnostics only
        console.print(f"[yellow]Embedder unavailable: {exc}[/yellow]")

    latest = Path(data_dir) / "reflections" / "latest.json"
    if latest.exists():
        try:
            info = json.loads(latest.read_text(encoding="utf-8"))
            console.print("[bold]Last reflection[/bold]")
            console.print(f"  report_id: {info.get('report_id')}")
            console.print(f"  timestamp: {info.get('timestamp')}")
            console.print(f"  episodes_analyzed: {info.get('episodes_analyzed')}")
            console.print(f"  path: {info.get('path')}")
            console.print(
                f"  mutates_memory: {info.get('mutates_memory')} "
                "[dim](always false — report-only)[/dim]"
            )
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"[yellow]Could not read last reflection: {exc}[/yellow]")
    else:
        console.print("[dim]No reflection reports yet. Run: grmc reflect[/dim]")


@app.command()
def reflect(
    recent: bool = typer.Option(
        False,
        "--recent",
        help="Analyze recent episodes (default mode when --topic is omitted).",
    ),
    limit: int = typer.Option(
        30,
        "--limit",
        "-n",
        help="Max recent episodes to analyze (recent mode; ignored if --topic)",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Topic mode: reflect on semantically related episodes",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Also write full report JSON to this path",
    ),
    no_persist: bool = typer.Option(
        False,
        "--no-persist",
        help="Do not write audit report under data_dir/reflections/",
    ),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Storage directory"),
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="Embedding backend: auto | sentence-transformers | hashing (topic mode)",
    ),
):
    """Run a conservative reflection pass (report only — never mutates the graph).

    Modes:
      grmc reflect
      grmc reflect --recent -n 20
      grmc reflect --topic "長期記憶"

    Produces concept candidates, soft contradiction flags, and suggested actions
    for human review. Knowledge graph is not written by this command.
    """
    if recent and topic:
        console.print(
            "[yellow]Both --recent and --topic given; using topic mode "
            "(semantic retrieve).[/yellow]"
        )

    manager = _manager(data_dir, embedder=embedder)

    with console.status("[bold]Reflecting (report-only)...[/bold]"):
        report = manager.reflect(
            recent_limit=limit,
            topic=topic,  # None → recent mode inside engine
            persist=not no_persist,
        )

    title = "Reflection Report (non-mutating)"
    if topic:
        title += f" — topic: {topic}"

    embedder_name = getattr(manager.embedder, "name", "unknown")
    console.print(
        Panel.fit(
            f"[bold]{report.report_id}[/bold]\n"
            f"mode={report.mode}  episodes={report.episodes_analyzed}  "
            f"confidence={report.confidence_level}  mutates_memory={report.mutates_memory}\n"
            f"embedder={embedder_name}",
            title=title,
            border_style="cyan",
        )
    )

    # Concepts
    if report.concept_candidates:
        table = Table(title="Concept candidates (not yet graph nodes)")
        table.add_column("Label", style="cyan")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right", style="green")
        table.add_column("Source")
        for c in report.concept_candidates[:15]:
            table.add_row(
                c.label,
                str(c.frequency),
                f"{c.confidence:.2f}",
                c.source,
            )
        console.print(table)
    else:
        console.print("[yellow]No concept candidates extracted.[/yellow]")

    # Contradictions
    if report.potential_contradictions:
        console.print(
            f"\n[bold yellow]Potential tensions "
            f"({len(report.potential_contradictions)}) — human review required[/bold yellow]"
        )
        for i, flag in enumerate(report.potential_contradictions[:10], 1):
            console.print(
                f"  {i}. conf={flag.confidence:.2f}  "
                f"{flag.episode_id_a} ↔ {flag.episode_id_b}"
            )
            console.print(f"     reason: {flag.reason}")
            console.print(f"     A: {flag.summary_a[:100]}")
            console.print(f"     B: {flag.summary_b[:100]}")
    else:
        console.print(
            "\n[dim]No contradiction heuristic fired "
            "(not proof of consistency).[/dim]"
        )

    # Actions / limitations / notes
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

    if output:
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] Wrote copy to: {output}")


if __name__ == "__main__":
    app()
