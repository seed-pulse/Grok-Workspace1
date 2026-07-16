"""Tests for client-side recent listing helpers (no embedder)."""

from grmc.storage.chroma_store import _deserialize_concepts, _serialize_concepts


def test_concept_roundtrip():
    raw = _serialize_concepts(["a", "b", " c "])
    assert _deserialize_concepts(raw) == ["a", "b", "c"]
    assert _deserialize_concepts("") == []
    assert _deserialize_concepts(None) == []
