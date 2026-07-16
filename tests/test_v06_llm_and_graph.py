"""v0.6: LLM verification (mock) + graph neighborhood."""

from pathlib import Path

from grmc.core.embedder import HashingEmbedder
from grmc.core.graph_query import neighborhood
from grmc.core.memory_manager import MemoryManager
from grmc.llm.client import MockLLMClient
from grmc.llm.config import LLMConfig
from grmc.llm.verification import LLMVerifier
from grmc.models.episode import Episode
from grmc.models.graph_edge import GraphEdge
from grmc.models.graph_node import GraphNode
from grmc.models.reflection_report import ConceptCandidate, ReflectionReport
from grmc.reflection.reflection_engine import ReflectionEngine
from grmc.storage.sqlite_store import SQLiteStore


def test_llm_default_off(tmp_path: Path):
    m = MemoryManager.from_data_dir(tmp_path / "d", embedder=HashingEmbedder())
    m.ingest_episode(Episode(content_summary="Human oversight is essential.", source="t"))
    report = m.reflect(llm=False, enqueue_proposals=False, enqueue_edge_suggestions=False)
    assert report.mutates_memory is False
    assert report.metadata.get("llm", {}).get("enabled") is False


def test_llm_enrichment_with_mock(tmp_path: Path):
    m = MemoryManager.from_data_dir(tmp_path / "d", embedder=HashingEmbedder())
    m.ingest_episode(
        Episode(
            episode_id="ep1",
            content_summary="Long-term memory requires human oversight for safety.",
            source="t",
        )
    )
    canned = {
        "concepts": [
            {
                "label": "human_oversight",
                "confidence": 0.9,  # will be capped
                "episode_ids": ["ep1"],
                "rationale": "central theme",
            }
        ]
    }
    cfg = LLMConfig(enabled=True, provider="mock", concept_conf_cap=0.5)
    verifier = LLMVerifier(config=cfg, client=MockLLMClient(canned))
    engine = ReflectionEngine(m, report_dir=str(tmp_path / "r"), llm_verifier=verifier)
    report = engine.reflect(recent_limit=5, llm=True, persist=False)
    assert report.mutates_memory is False
    assert report.metadata["llm"]["enabled"] is True
    labels = {c.label: c for c in report.concept_candidates}
    assert "human_oversight" in labels
    assert labels["human_oversight"].confidence <= 0.5


def test_llm_failure_falls_back(tmp_path: Path):
    class Boom(MockLLMClient):
        def complete_json(self, system, user, *, temperature=0.1):
            raise RuntimeError("network down")

    m = MemoryManager.from_data_dir(tmp_path / "d", embedder=HashingEmbedder())
    m.ingest_episode(Episode(content_summary="Reflection is conservative.", source="t"))
    # Inject broken verifier via engine
    from grmc.llm.config import LLMConfig
    from grmc.llm.verification import LLMVerifier

    # Boom will be raised inside client; LLMVerifier wraps LLMError - need LLMError
    # Boom raises RuntimeError which enrich catches as LLMError only - complete_json
    # raises RuntimeError from our Boom - verification catches LLMError only.
    # OpenAI client raises LLMError; Mock Boom raises RuntimeError.
    # enrich_report catches LLMError - RuntimeError will propagate to _apply_llm
    # which catches Exception. Good.
    cfg = LLMConfig(enabled=True, provider="mock")
    verifier = LLMVerifier(config=cfg, client=Boom())
    engine = ReflectionEngine(m, report_dir=str(tmp_path / "r"), llm_verifier=verifier)
    report = engine.reflect(llm=True, persist=False)
    assert report.mutates_memory is False
    assert any("failed" in n.lower() or "heuristic" in n.lower() for n in report.notes)


def test_neighborhood_depth(tmp_path: Path):
    db = SQLiteStore(tmp_path / "g.db")
    a = GraphNode(node_id="node_a", label="A", confidence=0.4)
    b = GraphNode(node_id="node_b", label="B", confidence=0.4)
    c = GraphNode(node_id="node_c", label="C", confidence=0.4)
    db.add_graph_node(a)
    db.add_graph_node(b)
    db.add_graph_node(c)
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

    n1 = neighborhood(db, "node_a", depth=1)
    assert len(n1.hops) == 1
    assert n1.hops[0].node.node_id == "node_b"

    n2 = neighborhood(db, "node_a", depth=2)
    ids = {h.node.node_id for h in n2.hops}
    assert "node_b" in ids and "node_c" in ids

    filt = neighborhood(db, "node_a", depth=2, edge_type="supports")
    assert all(h.via_edge_type == "supports" for h in filt.hops)
