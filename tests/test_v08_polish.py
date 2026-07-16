"""v0.8 polish: dump quality + path describe + docs presence."""

from pathlib import Path

from grmc.core.export_dump import build_dump, dump_markdown
from grmc.core.graph_query import GraphPath, PathStep
from grmc.models.episode import Episode
from grmc.models.graph_edge import GraphEdge
from grmc.models.graph_node import GraphNode
from grmc.storage.sqlite_store import SQLiteStore


def test_path_describe_readable():
    path = GraphPath(
        node_ids=["n1", "n2"],
        length=1,
        steps=[
            PathStep(
                from_node_id="n1",
                to_node_id="n2",
                edge_id="e1",
                edge_type="supports",
                direction="out",
                edge_confidence=0.3,
                from_label="Alpha",
                to_label="Beta",
            )
        ],
    )
    text = path.describe()
    assert "Alpha" in text and "Beta" in text
    assert "supports" in text
    assert "--[" in text


def test_dump_includes_labels_and_guidance(tmp_path: Path):
    db = SQLiteStore(tmp_path / "d.db")
    db.add_episode(
        Episode(episode_id="ep1", content_summary="Oversight matters", source="t")
    )
    a = GraphNode(node_id="node_a", label="oversight", confidence=0.4)
    b = GraphNode(node_id="node_b", label="memory", confidence=0.35)
    db.add_graph_node(a)
    db.add_graph_node(b)
    db.add_graph_edge(
        GraphEdge(
            source_node_id="node_a",
            target_node_id="node_b",
            edge_type="supports",
            confidence=0.3,
        )
    )
    data = build_dump(db)
    assert data["graph_edges"][0]["source_label"] == "oversight"
    md = dump_markdown(db)
    assert "oversight" in md and "supports" in md
    assert "QUICKSTART" in md or "approve" in md


def test_handover_docs_exist():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "README.md",
        "docs/QUICKSTART.md",
        "docs/DESIGN_PRINCIPLES.md",
        "docs/HANDOVER.md",
    ):
        path = root / name
        assert path.exists(), name
        assert path.stat().st_size > 200
