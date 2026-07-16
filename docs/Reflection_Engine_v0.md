# Reflection Engine v0.1 — Integrated Design Notes

**Status:** Implemented in scaffold (report-only)  
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

## CLI

```bash
grmc reflect                 # recent episodes
grmc reflect --topic "長期記憶"
grmc reflect -n 50 -o report.json
grmc status                  # shows last reflection pointer
```

## Honest limitations (Phase 0/1)

1. ChromaDB has no native chronological index — "recent" is client-side sort on `metadata.timestamp`.
2. Concept extraction is regex/heuristic, not LLM-assisted.
3. Contradiction detection is surface polarity + opposing term pairs; no embedding pairwise check yet.
4. No approval gate UI — reports are the gate for now.

## Next (v0.2 proposals)

See README "Next Steps / v0.2".
