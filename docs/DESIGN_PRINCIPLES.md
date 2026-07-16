# Design Principles — Why GRMC is built this way

This project is an experiment in **persistent, reflective, truth-seeking memory** for long-horizon collaboration with Grok. Features are secondary to these principles.

## 1. Prefer missing a memory over a wrong high-confidence belief

False confidence compounds. The system is biased toward:

- low default confidence
- soft flags instead of hard assertions
- empty results over fabricated structure

## 2. Separate “think” from “write”

| Phase | Command | May write graph? |
|-------|---------|------------------|
| Think | `reflect` | **No** |
| Propose | (side effect of reflect / `edges propose`) | pending queue only |
| Write | `approve` | **Yes** (nodes, edges, provenance) |
| Dismiss | `reject` | No graph write |

`ReflectionReport.mutates_memory` is always `False`. That is intentional and tested.

## 3. Human oversight is the gate, not a UI afterthought

Anything that changes the semantic graph (nodes, edges) goes through the approval queue. Soft edge suggestions from reflection still require `approve`.

## 4. Provenance over clever inference

Approved concepts should answer: **which episodes justify this node?**  
`episode_node_links` and `grmc node --provenance` exist so beliefs stay accountable.

## 5. Dual storage with clear roles

| Store | Role |
|-------|------|
| **SQLite** | System of record: episodes, reflections, proposals, nodes, edges, provenance |
| **Chroma** | Vector search only |

Chronology and truth-tracking are not delegated to a vector DB.

## 6. LLM is optional and audited

- Default **OFF** (no cost, deterministic heuristics)
- When ON: report enrichment only, confidence still capped
- Failures fall back to heuristics
- Calls land in `llm_audit/calls.jsonl`

## 7. Start simple, evolve without rewriting safety

Phases (0.1 → 0.7) added capability **around** the same safety core:

```
ingest → reflect → propose → approve → inspect
```

New features that violate this loop are rejected by design.

## 8. Local-first, inspectable artifacts

Everything important is on disk under `grmc_data/` (or `--data-dir`):

- `grmc.db` — SoR  
- `chroma/` — embeddings  
- `reflections/` — JSON audits  
- `llm_audit/` — optional LLM logs  

No silent cloud write path for the knowledge graph.

## What we deliberately do *not* do

- Auto-promote concepts to beliefs with high confidence  
- Auto-write edges from every contradiction flag  
- Login automation against grok.com (bridge uses human/Git courier)  
- Treat embedding similarity as proof of truth  

See also: [QUICKSTART.md](QUICKSTART.md) · [HANDOVER.md](HANDOVER.md)
