"""Graph edges + provenance tests."""

from pathlib import Path

from grmc.core.approval_queue import ApprovalQueue
from grmc.models.graph_node import GraphNode
from grmc.models.reflection_report import ConceptCandidate, ReflectionReport
from grmc.storage.sqlite_store import SQLiteStore


def _approve_two_nodes(queue: ApprovalQueue):
    r = ReflectionReport(
        report_id="refl_e",
        concept_candidates=[
            ConceptCandidate(
                label="human_oversight",
                frequency=2,
                confidence=0.5,
                supporting_episode_ids=["ep_a", "ep_b"],
                source="episode_field",
            ),
            ConceptCandidate(
                label="long_term_memory",
                frequency=2,
                confidence=0.45,
                supporting_episode_ids=["ep_a"],
                source="episode_field",
            ),
        ],
    )
    props = queue.enqueue_from_report(r)
    assert len(props) == 2
    n1 = queue.approve(props[0].proposal_id)["node"]
    n2 = queue.approve(props[1].proposal_id)["node"]
    return n1, n2


def test_edge_requires_approve(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    n1, n2 = _approve_two_nodes(queue)

    assert db.count_graph_edges() == 0
    prop = queue.enqueue_edge(
        n1.node_id,
        n2.node_id,
        "supports",
        confidence=0.4,
        supporting_episode_ids=["ep_a"],
        note="oversight supports LTM safety",
    )
    assert prop.kind == "edge"
    assert prop.status == "pending"
    assert db.count_graph_edges() == 0  # still no write

    result = queue.approve(prop.proposal_id)
    assert result["kind"] == "edge"
    edge = result["edge"]
    assert edge.edge_type == "supports"
    assert edge.confidence <= 0.45
    assert edge.source_node_id == n1.node_id
    assert edge.target_node_id == n2.node_id
    assert db.count_graph_edges() == 1
    assert db.list_graph_edges(node_id=n1.node_id)
    assert db.find_edge(n1.node_id, n2.node_id, "supports") is not None


def test_edge_rejects_unknown_type_and_duplicate(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    n1, n2 = _approve_two_nodes(queue)

    try:
        queue.enqueue_edge(n1.node_id, n2.node_id, "teleports")
        assert False, "should raise"
    except ValueError:
        pass

    queue.enqueue_edge(n1.node_id, n2.node_id, "related_to")
    try:
        queue.enqueue_edge(n1.node_id, n2.node_id, "related_to")
        assert False, "duplicate pending should raise"
    except ValueError:
        pass


def test_provenance_queryable(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    # Need real episode rows for linked_graph_nodes update (optional)
    from grmc.models.episode import Episode
    from datetime import datetime

    db.add_episode(
        Episode(
            episode_id="ep_a",
            content_summary="oversight and memory discussion",
            timestamp=datetime.utcnow(),
        )
    )
    r = ReflectionReport(
        report_id="r",
        concept_candidates=[
            ConceptCandidate(
                label="human_oversight",
                confidence=0.5,
                frequency=2,
                supporting_episode_ids=["ep_a"],
                source="episode_field",
            )
        ],
    )
    props = queue.enqueue_from_report(r)
    node = queue.approve(props[0].proposal_id)["node"]
    links = db.list_links_for_node(node.node_id)
    assert len(links) == 1
    assert links[0].episode_id == "ep_a"
    assert links[0].proposal_id == props[0].proposal_id
    ep = db.get_episode("ep_a")
    assert node.node_id in (ep.get("linked_graph_nodes") or [])


def test_approve_node_can_suggest_related_edge(tmp_path: Path):
    db = SQLiteStore(tmp_path / "t.db")
    queue = ApprovalQueue(db)
    n1, _ = _approve_two_nodes(queue)
    # third node proposal
    r = ReflectionReport(
        report_id="r3",
        concept_candidates=[
            ConceptCandidate(
                label="confidence_caps",
                confidence=0.4,
                frequency=1,
                supporting_episode_ids=["ep_x"],
                source="episode_field",
            )
        ],
    )
    props = queue.enqueue_from_report(r)
    result = queue.approve(
        props[0].proposal_id,
        also_link_related=True,
        related_to_node_id=n1.node_id,
        related_edge_type="related_to",
    )
    assert result["node"] is not None
    assert result["related_edge_proposal"] is not None
    assert result["related_edge_proposal"].status == "pending"
    assert db.count_graph_edges() == 0  # edge not auto-written
