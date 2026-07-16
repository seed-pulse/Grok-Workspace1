"""v0.7: path query, LLM audit, export dump."""

from pathlib import Path

from grmc.core.export_dump import dump_markdown
from grmc.core.graph_query import find_paths
from grmc.llm.audit import LLMAuditLog
from grmc.llm.client import MockLLMClient
from grmc.llm.config import LLMConfig
from grmc.llm.verification import LLMVerifier
from grmc.models.episode import Episode
from grmc.models.graph_edge import GraphEdge
from grmc.models.graph_node import GraphNode
from grmc.models.reflection_report import ReflectionReport
from grmc.storage.sqlite_store import SQLiteStore


def _chain(db: SQLiteStore):
    for nid, lab in [("node_a", "A"), ("node_b", "B"), ("node_c", "C"), ("node_d", "D")]:
        db.add_graph_node(GraphNode(node_id=nid, label=lab, confidence=0.4))
    db.add_graph_edge(
        GraphEdge(
            source_node_id="node_a",
            target_node_id="node_b",
            edge_type="supports",
            confidence=0.3,
        )
    )
    db.add_graph_edge(
        GraphEdge(
            source_node_id="node_b",
            target_node_id="node_c",
            edge_type="related_to",
            confidence=0.25,
        )
    )
    db.add_graph_edge(
        GraphEdge(
            source_node_id="node_c",
            target_node_id="node_d",
            edge_type="supports",
            confidence=0.3,
        )
    )


def test_find_path_shortest(tmp_path: Path):
    db = SQLiteStore(tmp_path / "p.db")
    _chain(db)
    result = find_paths(db, "node_a", "node_d", max_depth=3)
    assert result.found
    assert result.paths
    assert result.paths[0].length == 3
    assert result.paths[0].node_ids[0] == "node_a"
    assert result.paths[0].node_ids[-1] == "node_d"


def test_find_path_depth_limit(tmp_path: Path):
    db = SQLiteStore(tmp_path / "p.db")
    _chain(db)
    result = find_paths(db, "node_a", "node_d", max_depth=2)
    assert result.found is False


def test_find_path_type_filter(tmp_path: Path):
    db = SQLiteStore(tmp_path / "p.db")
    _chain(db)
    # Only supports edges: a-b and c-d, no supports b-c → no path
    result = find_paths(db, "node_a", "node_d", max_depth=3, edge_type="supports")
    assert result.found is False
    # a to b with supports works
    r2 = find_paths(db, "node_a", "node_b", max_depth=1, edge_type="supports")
    assert r2.found and r2.paths[0].length == 1


def test_llm_audit_log(tmp_path: Path):
    audit = LLMAuditLog(tmp_path / "data")
    cfg = LLMConfig(enabled=True, provider="mock", model="mock-model")
    canned = {
        "concepts": [
            {"label": "human_oversight", "confidence": 0.4, "episode_ids": ["e1"]}
        ]
    }
    verifier = LLMVerifier(
        config=cfg, client=MockLLMClient(canned), audit=audit
    )
    report = ReflectionReport(report_id="refl_audit_test")
    eps = [{"episode_id": "e1", "summary": "Human oversight matters."}]
    out = verifier.enrich_report(report, eps)
    assert out.mutates_memory is False
    assert out.metadata["llm"]["enabled"] is True
    summary = audit.summary()
    assert summary["calls"] >= 1
    assert summary["success"] >= 1
    recent = audit.list_recent(5)
    assert recent[0].purpose in {"concept_extract", "contradiction_review"}


def test_export_markdown(tmp_path: Path):
    db = SQLiteStore(tmp_path / "e.db")
    db.add_episode(
        Episode(episode_id="ep_x", content_summary="Hello memory world", source="t")
    )
    db.add_graph_node(GraphNode(label="hello", confidence=0.4, supporting_episodes=["ep_x"]))
    md = dump_markdown(db)
    assert "GRMC Memory Dump" in md
    assert "ep_x" in md
    assert "hello" in md
