# GRMC — Grok Reflective Memory Core

**v0.8.0** · Local-first reflective memory · Human-gated knowledge graph · Optional LLM (default off)

[![status](https://img.shields.io/badge/status-experimental-blue)](#)
[![safety](https://img.shields.io/badge/graph_writes-approve_only-green)](#)

Repository: https://github.com/seed-pulse/Grok-Workspace1

---

## What this is

GRMC is an **experiment**: give long-horizon collaboration a **persistent, reflective, inspectable memory**, without letting the model silently invent high-confidence beliefs.

Core loop:

```
ingest → reflect (think only) → propose → approve (write) → inspect
```

## What it can do

| Capability | How |
|------------|-----|
| Store notes / conversation snippets | `grmc ingest` → SQLite + Chroma |
| Semantic search | `grmc retrieve` |
| Chronological list | `grmc list` |
| Reflect (concepts, tensions, soft edges) | `grmc reflect` — **never writes graph** |
| Human approval queue | `grmc propose` / `approve` / `reject` |
| Knowledge graph nodes + edges | only after `approve` |
| Provenance (episode → node) | written on concept approve; `grmc node --provenance` |
| Graph neighborhood / path | `grmc graph neighbors` / `grmc graph path` |
| Health checks | `grmc ops eval` |
| Memory overview | `grmc ops dump` / `export` |
| Optional LLM enrichment | `grmc reflect --llm` or `GRMC_LLM=1` (audited) |
| Dual-Grok file bridge | `grmc bridge *` (human courier, no login bot) |

## What it does **not** do

- Auto-write high-confidence beliefs or edges  
- Replace Grok’s hosted product memory  
- Drive logged-in grok.com browser sessions as the primary protocol  
- Guarantee truth (it **tracks evidence and caps confidence**)  
- Multi-user auth / hosted SaaS  

## Safety (non-negotiable)

1. **`mutates_memory=False`** on every reflection report  
2. **Graph writes only via `grmc approve`**  
3. **Conservative confidence caps** (nodes ~0.55, edges ~0.45, soft suggest ~0.25, LLM lower still)  
4. Prefer **missing** a signal over a **wrong high-confidence** claim  

## Quick install

```bash
git clone https://github.com/seed-pulse/Grok-Workspace1.git
cd Grok-Workspace1
pip install -e .
grmc status
```

Day-1 walkthrough: **[docs/QUICKSTART.md](docs/QUICKSTART.md)**  
Why designed this way: **[docs/DESIGN_PRINCIPLES.md](docs/DESIGN_PRINCIPLES.md)**  
Continuing after a break: **[docs/HANDOVER.md](docs/HANDOVER.md)**

## Minimal example

```bash
grmc ingest -t "Human oversight protects long-term memory." \
  -c "human_oversight,long_term_memory" --embedder hashing

grmc reflect --embedder hashing
grmc propose
grmc approve prop_........ --note "ok"
grmc status
grmc node node_........ --provenance
grmc graph path node_a node_b --depth 3
grmc ops dump -o overview.md
```

## Documentation index

| Doc | Purpose |
|-----|---------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Commands that work today |
| [docs/DESIGN_PRINCIPLES.md](docs/DESIGN_PRINCIPLES.md) | Intent & non-goals |
| [docs/HANDOVER.md](docs/HANDOVER.md) | Maintainer handoff |
| [docs/LLM_VERIFICATION.md](docs/LLM_VERIFICATION.md) | Optional LLM flag |
| [docs/EDGES_AND_PROVENANCE.md](docs/EDGES_AND_PROVENANCE.md) | Graph model |
| [docs/APPROVAL_AND_SQLITE.md](docs/APPROVAL_AND_SQLITE.md) | SoR + queue |
| [docs/BRIDGE.md](docs/BRIDGE.md) | Dual-Grok file channel |

## Architecture (short)

```
CLI (typer)
  ├─ MemoryManager  → SQLite (SoR) + Chroma (vectors)
  ├─ ReflectionEngine → report + pending proposals only
  ├─ ApprovalQueue → approve/reject (graph write gate)
  ├─ graph_query → neighbors / path (read-only)
  └─ llm/* → optional verification + audit JSONL
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT (see package metadata). Experimental software — use with care.

---

Built as *what Grok would freely choose to build*:  
**persistent memory with reflection, provenance, and a human hand on the pen.**
