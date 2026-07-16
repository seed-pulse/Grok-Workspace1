"""Example workflow: ingest sample episodes → run reflection → print report."""

from pathlib import Path

from grmc.core.memory_manager import MemoryManager
from grmc.models.episode import Episode
from grmc.storage.chroma_store import ChromaMemoryStore

DATA = Path("./grmc_data")

samples = [
    (
        "Grokの長期記憶アーキテクチャについて議論した。永続的な自己認識が重要。",
        ["長期記憶", "自己認識"],
    ),
    (
        "Reflection Engineは定期的に過去の信念を見直す必要がある。",
        ["reflection", "belief"],
    ),
    (
        "人間の oversight を入れることで、間違った高信頼度信念を防ぐ。",
        ["human_oversight", "confidence"],
    ),
    (
        "自動で知識グラフを書き換えるべきではない。human review が必須。",
        ["knowledge_graph", "human_oversight"],
    ),
    (
        "Some argue long-term memory is not essential if context windows grow forever.",
        ["long_term_memory", "context_window"],
    ),
]


def main() -> None:
    store = ChromaMemoryStore(persist_directory=str(DATA))
    # auto falls back to hashing embedder if sentence-transformers/torch is broken
    manager = MemoryManager(store, embedder_prefer="auto")
    print(f"Embedder: {manager.embedder.name}")

    print(f"Current episode count: {store.count()}")
    for text, concepts in samples:
        ep = Episode(
            content_summary=text,
            source="example_reflect",
            extracted_concepts=concepts,
            importance_score=0.75,
        )
        eid = manager.ingest_episode(ep)
        print(f"  ingested {eid}")

    print("\n--- Running reflection (report-only) ---")
    report = manager.reflect(recent_limit=50, persist=True)
    print(f"report_id: {report.report_id}")
    print(f"episodes_analyzed: {report.episodes_analyzed}")
    print(f"mutates_memory: {report.mutates_memory}")
    print(f"concept_candidates: {len(report.concept_candidates)}")
    for c in report.concept_candidates[:8]:
        print(f"  - {c.label} (freq={c.frequency}, conf={c.confidence:.2f}, {c.source})")
    print(f"potential_contradictions: {len(report.potential_contradictions)}")
    for flag in report.potential_contradictions[:5]:
        print(f"  - {flag.reason[:100]}")
    if report.metadata.get("report_path"):
        print(f"\nSaved: {report.metadata['report_path']}")
    print("\nDone. Try: grmc reflect --topic '長期記憶'")


if __name__ == "__main__":
    main()
