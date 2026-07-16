"""Unit tests for ReflectionEngine (no Chroma / embedder required)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from grmc.models.reflection_report import ReflectionReport
from grmc.reflection.reflection_engine import ReflectionEngine


class FakeStore:
    def __init__(self, episodes: Optional[List[Dict[str, Any]]] = None):
        self._episodes = episodes or []
        self.persist_dir = "/tmp/grmc_test_data"

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._episodes[:limit]

    def count(self) -> int:
        return len(self._episodes)


class FakeManager:
    def __init__(self, episodes: Optional[List[Dict[str, Any]]] = None):
        self.store = FakeStore(episodes)
        self._retrieve_results: List[Dict[str, Any]] = []

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._retrieve_results[:top_k]


def test_reflect_empty_is_safe():
    engine = ReflectionEngine(FakeManager([]), report_dir="/tmp/grmc_test_reflections")
    report = engine.reflect(persist=False)
    assert isinstance(report, ReflectionReport)
    assert report.mutates_memory is False
    assert report.confidence_level == "conservative"
    assert report.episodes_analyzed == 0
    assert report.potential_issues


def test_reflect_extracts_concepts_and_stays_non_mutating():
    episodes = [
        {
            "episode_id": "ep1",
            "summary": "Long-term memory is essential for AI continuity and reflection.",
            "metadata": {"timestamp": "2026-07-15T10:00:00"},
            "extracted_concepts": ["long_term_memory", "reflection"],
        },
        {
            "episode_id": "ep2",
            "summary": "Human oversight prevents wrong high-confidence beliefs in memory.",
            "metadata": {"timestamp": "2026-07-15T11:00:00"},
            "extracted_concepts": ["human_oversight", "confidence"],
        },
        {
            "episode_id": "ep3",
            "summary": "Reflection should consolidate long-term memory carefully.",
            "metadata": {"timestamp": "2026-07-15T12:00:00"},
            "extracted_concepts": ["long_term_memory", "reflection"],
        },
    ]
    engine = ReflectionEngine(FakeManager(episodes), report_dir="/tmp/grmc_test_reflections")
    report = engine.reflect(recent_limit=10, persist=False)

    assert report.episodes_analyzed == 3
    assert report.mutates_memory is False
    labels = {c.label for c in report.concept_candidates}
    assert "long_term_memory" in labels or "reflection" in labels
    # Explicit field concepts should appear with modest confidence
    field_candidates = [c for c in report.concept_candidates if c.source == "episode_field"]
    assert field_candidates
    assert all(c.confidence <= 0.55 for c in field_candidates)
    assert any("was not modified" in n for n in report.notes)


def test_simple_contradiction_check_polarity():
    engine = ReflectionEngine(FakeManager([]), report_dir="/tmp/grmc_test_reflections")
    # Shared substance + one negation
    assert engine.simple_contradiction_check(
        "Long-term memory is essential for continuity.",
        "Long-term memory is not essential for continuity.",
    )
    # Same polarity → no flag
    assert not engine.simple_contradiction_check(
        "Reflection is important for growth.",
        "Reflection is important for understanding.",
    )


def test_topic_mode_uses_retrieve():
    manager = FakeManager([])
    manager._retrieve_results = [
        {
            "episode_id": "ep_topic",
            "summary": "Knowledge graph nodes need provenance and confidence scores.",
            "distance": 0.2,
            "extracted_concepts": ["knowledge_graph", "confidence"],
        }
    ]
    engine = ReflectionEngine(manager, report_dir="/tmp/grmc_test_reflections")
    report = engine.reflect(topic="knowledge graph", persist=False)
    assert report.mode == "topic"
    assert report.topic == "knowledge graph"
    assert report.episodes_analyzed == 1
    assert report.episode_ids == ["ep_topic"]
