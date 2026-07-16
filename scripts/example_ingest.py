"""Example script to ingest sample memories."""

from grmc.models.episode import Episode
from grmc.storage.chroma_store import ChromaMemoryStore
from grmc.core.memory_manager import MemoryManager

store = ChromaMemoryStore(persist_directory="./grmc_data")
manager = MemoryManager(store)

samples = [
    "Grokの長期記憶アーキテクチャについて議論した。永続的な自己認識が重要。",
    "Reflection Engineは定期的に過去の信念を見直す必要がある。",
    "人間の oversight を入れることで、間違った高信頼度信念を防ぐ。",
]

for text in samples:
    ep = Episode(content_summary=text, source="example_script")
    ep_id = manager.ingest_episode(ep)
    print(f"Ingested: {ep_id}")

print("\nDone. Try: grmc retrieve '長期記憶'")
