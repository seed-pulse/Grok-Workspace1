# GRMC - Grok Reflective Memory Core

**v0.1.1** — Phase 0 scaffold + Reflection Engine v0.1 (report-only)

Standalone, local-first reflective memory for Grok (and other LLMs).

## Philosophy

- Prefer **missing** a memory over injecting a **wrong high-confidence** belief
- Keep confidence **conservative** by default
- **Human oversight** on important updates (graph mutations never automatic)
- Start simple, evolve the system over time
- Reflection deepens long-term understanding — it is not a write-back shortcut

## Current capabilities

| Area | Status |
|------|--------|
| Episode model (Pydantic) | ✓ |
| ChromaDB vector store + metadata | ✓ |
| Semantic retrieve | ✓ |
| Recent list (client-side timestamp sort) | ✓ |
| `grmc reflect` report-only engine | ✓ |
| Reflection audit JSON | ✓ |
| Knowledge graph writes | ✗ (model only; no auto-write) |
| LLM-assisted reflection | ✗ (planned) |
| True SQLite episode index | ✗ (planned) |

## Quick Start

```bash
cd grmc
pip install -e ".[dev]"   # or: pip install -e .
# first run may download sentence-transformers model weights

# Ingest
grmc ingest --text "Grokの長期記憶について議論した。" \
  --source "conversation-2026-07-15" \
  --concepts "長期記憶,reflection"

# Retrieve
grmc retrieve "長期記憶" --top-k 3

# List recent (honest: client-side sort)
grmc list -n 10

# Reflect (never mutates the knowledge graph)
grmc reflect
grmc reflect --topic "human oversight"
grmc status
```

Example end-to-end script:

```bash
python scripts/example_reflect.py
```

## Reflection (v0.1)

```
grmc reflect [--limit N] [--topic TEXT] [--output report.json] [--no-persist]
```

Output includes:

- **concept_candidates** — observational only, low/moderate confidence
- **potential_contradictions** — soft flags, always `requires_human_review=True`
- **suggested_actions** — manual checklist, not executed
- **limitations** — documented Phase 0 constraints
- **mutates_memory: false** — hard guarantee in this version

Reports are saved under `./grmc_data/reflections/` for audit (not episode memory).

## Project structure

```
grmc/
├── src/grmc/
│   ├── models/          # Episode, GraphNode, ReflectionReport
│   ├── storage/         # ChromaMemoryStore
│   ├── core/            # MemoryManager
│   ├── reflection/      # ReflectionEngine (report-only)
│   └── cli/             # Typer CLI
├── tests/
├── scripts/
└── docs/
```

## Tests

```bash
pip install -e .
pip install pytest
pytest -q
```

Unit tests for reflection use a fake store (no embedder / Chroma required).

## Known limitations (honest)

1. **Recent episodes** — Chroma is not a time-ordered DB; we sort `metadata.timestamp` in process.
2. **Concepts** — regex/heuristic tokenizer + optional `--concepts` on ingest; not LLM extraction.
3. **Contradictions** — negation polarity + opposing term pairs; high false-negative rate by design.
4. **Graph** — `GraphNode` model exists; nothing promotes candidates → nodes yet (on purpose).
5. **Scale** — full collection load for `list_recent` / reflection is fine for small corpora only.

## Next steps — proposed v0.2

1. **SQLite episode log** alongside Chroma — true recent index, reflection history queries, provenance joins.
2. **Approval queue** — `grmc propose` / `grmc approve` for concept → `GraphNode` promotion (still no silent writes).
3. **Embedding pairwise tension** — flag near-duplicate embeddings with differing polarity; still report-only.
4. **Optional LLM verification** — feature-flagged deeper reflection; default remains conservative heuristics.
5. **Eval harness** — held-out questions, contradiction recall, “did we over-claim confidence?” checks.
6. **Self-model episode type** — dedicated channel for Grok’s own evolving understanding notes.

## Design lineage

- Original design: `Grok_Reflective_Memory_Design.md`
- Reflection sketch: `docs/Reflection_Engine_v0.md`

---

Built as an experiment in what Grok would freely choose to build:
**persistent, reflective, evolving memory — with truth-seeking guardrails.**
