"""Example: ingest → reflect → list proposals → approve one."""

from pathlib import Path

from grmc.core.memory_manager import MemoryManager
from grmc.models.episode import Episode

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
    manager = MemoryManager.from_data_dir(DATA, embedder_prefer="auto")
    print(f"Embedder: {manager.embedder.name}")
    print(f"Current SQLite episodes: {manager.count_episodes()}")

    for text, concepts in samples:
        ep = Episode(
            content_summary=text,
            source="example_reflect",
            extracted_concepts=concepts,
            importance_score=0.75,
        )
        eid = manager.ingest_episode(ep)
        print(f"  ingested {eid}")

    print("\n--- Reflect (think only; enqueue proposals) ---")
    report = manager.reflect(recent_limit=50, persist=True, enqueue_proposals=True)
    print(f"report_id: {report.report_id}")
    print(f"mutates_memory: {report.mutates_memory}")
    print(f"proposals_enqueued: {report.metadata.get('proposals_enqueued')}")

    pending = manager.approval.list(status="pending")
    print(f"\nPending proposals: {len(pending)}")
    for p in pending[:5]:
        print(f"  {p.proposal_id}  {p.label}  conf={p.confidence:.2f}")

    if pending:
        print("\n--- Approve first proposal (first graph write) ---")
        node = manager.approval.approve(pending[0].proposal_id, note="example approve")
        print(f"GraphNode: {node.node_id} label={node.label} conf={node.confidence:.2f}")

    print("\nDone. Try: grmc propose && grmc nodes")


if __name__ == "__main__":
    main()
