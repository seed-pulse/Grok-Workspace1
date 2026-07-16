# Reflection Engine v0.1 — Integrated

**Status:** Integrated into the main scaffold (report-only)  
**Location:** `src/grmc/reflection/reflection_engine.py`  
**CLI:** `grmc reflect` / `grmc reflect --recent` / `grmc reflect --topic "..."`  
**Principles:** Conservative · Truth-seeking · Human oversight · Non-mutating

## What it does

`ReflectionEngine.reflect()` scans a set of episodes and returns a
`ReflectionReport` containing:

- concept candidates (heuristic + optional `extracted_concepts` from ingest)
- soft contradiction / tension flags (low confidence)
- suggested actions for humans
- explicit **limitations** of the current engine

It does **not**:

- write or update knowledge graph nodes
- raise confidence on beliefs
- delete or rewrite episodes

Optional side effect: JSON audit file under `grmc_data/reflections/`.

## Wiring

```
CLI (grmc reflect)
  → MemoryManager.reflect()
    → ReflectionEngine.reflect()
      → ChromaMemoryStore.list_recent() | MemoryManager.retrieve(topic)
      → ReflectionReport (Pydantic)
      → optional JSON under grmc_data/reflections/
```

## Phase 0 limitations (honest)

1. ChromaDB has no native chronological index — "recent" is client-side sort on `metadata.timestamp`.
2. Concept extraction is regex/heuristic, not LLM-assisted.
3. Contradiction detection is surface polarity + opposing term pairs.
4. No approval gate UI yet — the report *is* the gate for humans.

## CLI examples

```bash
grmc reflect
grmc reflect --recent -n 20
grmc reflect --topic "human oversight"
grmc reflect -o /tmp/report.json
grmc status   # shows last reflection pointer
```

## Next (after this integration)

1. SQLite episode log for true recent index  
2. Approval queue: promote concept candidates → GraphNode only after human sign-off  
3. Embedding pairwise tension checks (still report-only)  
4. Optional LLM verification behind a feature flag  
