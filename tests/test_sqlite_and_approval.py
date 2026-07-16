"""SQLite SoR + approval queue integration tests (hashing embedder, no torch)."""

from pathlib import Path

from grmc.core.approval_queue import ApprovalQueue
from grmc.core.embedder import HashingEmbedder
from grmc.core.memory_manager import MemoryManager
from grmc.models.episode import Episode
from grmc.models.reflection_report import ConceptCandidate, ReflectionReport
from grmc.storage.sqlite_store import SQLiteStore


def test_sqlite_list_recent_order(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    e1 = Episode(
        episode_id="ep_old",
        content_summary="older",
        timestamp=__import__("datetime").datetime(2026, 1, 1),
    )
    e2 = Episode(
        episode_id="ep_new",
        content_summary="newer",
        timestamp=__import__("datetime").datetime(2026, 6, 1),
    )
    db.add_episode(e1)
    db.add_episode(e2)
    recent = db.list_recent(10)
    assert [r["episode_id"] for r in recent] == ["ep_new", "ep_old"]


def test_dual_write_ingest_and_reflect_enqueue(tmp_path: Path):
    data = tmp_path / "data"
    manager = MemoryManager.from_data_dir(data, embedder=HashingEmbedder())
    for i, text in enumerate(
        [
            "Human oversight is essential for long-term memory safety.",
            "Human oversight prevents wrong high-confidence beliefs.",
            "Reflection should stay report-only until approval.",
        ]
    ):
        manager.ingest_episode(
            Episode(
                content_summary=text,
                source="test",
                extracted_concepts=["human_oversight", "reflection"] if i < 2 else ["reflection"],
            )
        )

    assert manager.count_episodes() == 3
    recent = manager.list_recent(2)
    assert len(recent) == 2

    report = manager.reflect(recent_limit=10, persist=True, enqueue_proposals=True)
    assert report.mutates_memory is False
    assert manager.sqlite.count_graph_nodes() == 0  # no auto graph write
    pending = manager.approval.list(status="pending")
    assert len(pending) >= 1
    assert report.metadata.get("proposals_enqueued", 0) >= 1


def test_approve_is_only_graph_write(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    report = ReflectionReport(
        report_id="refl_test",
        concept_candidates=[
            ConceptCandidate(
                label="conservative_memory",
                frequency=3,
                confidence=0.5,
                supporting_episode_ids=["ep1", "ep2"],
                source="episode_field",
            )
        ],
    )
    created = queue.enqueue_from_report(report)
    assert len(created) == 1
    assert db.count_graph_nodes() == 0

    result = queue.approve(created[0].proposal_id, confidence_cap=0.55)
    node = result["node"]
    assert result["kind"] == "node"
    assert node.label == "conservative_memory"
    assert node.confidence <= 0.55
    assert db.count_graph_nodes() == 1
    # Provenance links for supporting episodes
    assert len(result["provenance_links"]) == 2
    assert db.count_episode_node_links() == 2

    prop = db.get_proposal(created[0].proposal_id)
    assert prop is not None
    assert prop.status == "approved"
    assert prop.resulting_node_id == node.node_id


def test_reject_no_graph(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    prop = queue.enqueue_candidate(
        ConceptCandidate(
            label="noise_token",
            confidence=0.4,
            frequency=1,
            source="episode_field",
        ),
        report_id="r2",
    )
    assert prop is not None
    queue.reject(prop.proposal_id, note="too noisy")
    assert db.count_graph_nodes() == 0
    assert db.get_proposal(prop.proposal_id).status == "rejected"


def test_duplicate_pending_skipped(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    c = ConceptCandidate(label="same_label", confidence=0.3, frequency=2)
    r1 = ReflectionReport(report_id="a", concept_candidates=[c])
    r2 = ReflectionReport(report_id="b", concept_candidates=[c])
    assert len(queue.enqueue_from_report(r1)) == 1
    assert len(queue.enqueue_from_report(r2)) == 0
