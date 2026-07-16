# GRMC Quick Start

**Version:** 0.8.1 · Local-first reflective memory with human-gated graph writes

## Install

```bash
cd grmc   # or: git clone https://github.com/seed-pulse/Grok-Workspace1.git
pip install -e .
# optional: pytest
pip install -e ".[dev]"
```

Data lives under `./grmc_data/` by default (`grmc.db`, `chroma/`, `reflections/`, `llm_audit/`).

## 5-minute loop

```bash
# 1) Remember
grmc ingest -t "Human oversight is essential for long-term memory safety." \
  -c "human_oversight,long_term_memory" --embedder hashing

# 2) Think (never writes the knowledge graph)
grmc reflect --embedder hashing

# 3) Review proposals
grmc propose
grmc propose --kind edge

# 4) Write only what a human approves
grmc approve prop_............ --note "looks solid"
# or
grmc reject prop_............ --note "noise"

# 5) Inspect
grmc status
grmc node node_............ --provenance
grmc list -n 10
grmc ops dump
```

## Common commands

| Goal | Command |
|------|---------|
| Dashboard | `grmc status` |
| Semantic search | `grmc retrieve "長期記憶"` |
| Think | `grmc reflect` / `grmc reflect --llm` |
| Approval queue | `grmc propose` → `approve` / `reject` |
| Graph read | `grmc graph neighbors …` / `grmc graph path a b` |
| Edges (propose only) | `grmc edges propose --from … --to …` |
| Health | `grmc ops eval` |
| Overview file | `grmc ops export -o dump.md` |
| LLM call log | `grmc ops llm-log` (empty unless LLM used) |

## LLM (optional, default OFF)

```bash
export GRMC_LLM=1
export GRMC_LLM_API_KEY=...
export GRMC_LLM_BASE_URL=https://api.x.ai/v1   # optional
export GRMC_LLM_MODEL=grok-2-latest            # optional
grmc reflect --llm
```

Without this, reflection uses heuristics only (no API cost).

## Embedder note

If `sentence-transformers` / torch is broken on your machine:

```bash
grmc ingest ... --embedder hashing
grmc reflect --embedder hashing
```

Hashing is lower quality for search but keeps the stack runnable.

## Safety one-liners

- Reflection: **report only** (`mutates_memory=False`)
- Graph changes: **only** `grmc approve`
- Confidence: capped and conservative by default

Next: [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) · [HANDOVER.md](HANDOVER.md)
