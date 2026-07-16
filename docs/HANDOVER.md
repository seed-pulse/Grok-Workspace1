# Handover — GRMC (Grok Reflective Memory Core)

**As of:** 2026-07-17 · **Code version:** 0.8.x  
**Repo:** https://github.com/seed-pulse/Grok-Workspace1  

This note is for a future maintainer (including future-you) who needs to continue the experiment without re-deriving the intent from scratch.

---

## What this is

A **local-first** Python package that:

1. Stores conversation/note **episodes** (SQLite + Chroma embeddings)
2. Runs **reflection** (report-only: concepts, tensions, soft edge ideas)
3. Queues **proposals** for human review
4. Writes **graph nodes / edges / provenance** only on `approve`
5. Offers read-only graph queries, eval, export, optional LLM enrichment

It is **not** a hosted product, not a multi-tenant SaaS, and not a drop-in replacement for Grok’s internal memory.

---

## Where things live

```
grmc/
├── README.md                 # overview + can/cannot
├── docs/
│   ├── QUICKSTART.md         # day-1 commands
│   ├── DESIGN_PRINCIPLES.md  # why
│   ├── HANDOVER.md           # this file
│   ├── LLM_VERIFICATION.md
│   ├── EDGES_AND_PROVENANCE.md
│   ├── APPROVAL_AND_SQLITE.md
│   └── ...
├── src/grmc/
│   ├── cli/                  # Typer entrypoints
│   ├── core/                 # MemoryManager, approval, graph_query, eval, export
│   ├── reflection/           # ReflectionEngine
│   ├── storage/              # SQLite SoR + Chroma vectors
│   ├── llm/                  # optional verification + audit
│   ├── models/               # Pydantic models
│   └── bridge/               # dual-Grok file channel
└── tests/                    # pytest (prefer these over ad-hoc scripts)
```

**Runtime data (gitignored patterns):** `./grmc_data/` by default.

---

## Mental model (do not break)

```
ingest  →  reflect (think)  →  propose queue  →  approve (write)  →  inspect
                 │                                      │
                 └── mutates_memory=False               └── only graph write path
```

Hard rules:

1. Reflection never writes graph nodes/edges  
2. Confidence stays conservative (caps in approval + LLM modules)  
3. Prefer false negative over false high-confidence belief  

If a PR violates these, reject it even if “clever.”

---

## How to run tests

```bash
pip install -e ".[dev]"
pytest -q
```

Current suite covers reflection safety, approval gates, edges/provenance, path query, LLM mock/audit, export.

---

## Environment quirks known

- **Torch / NumPy mismatch** on some Macs → use `--embedder hashing`  
- **LLM** off by default; without keys, `--llm` falls back safely  
- **Legacy Chroma** at data-dir root: `grmc ops migrate-legacy` (episodes only, no vectors rebuilt)

---

## Suggested next work (not required for “done”)

Ordered for *continuing the experiment*, not productizing:

1. Stronger eval fixtures loaded by one command  
2. Better multi-path ranking for `graph path` (still read-only)  
3. Optional LLM cost table (model → $/1K) in audit summary  
4. Episode redaction / privacy controls  
5. GraphML export for external visualization  

Avoid: silent auto-approve, raising default conf caps, coupling to grok.com login automation as the primary bridge.

---

## Dual-Grok bridge (context)

`grmc bridge` is a **file courier** between web Grok and CLI Grok.  
It intentionally does **not** drive a logged-in browser session.

---

## Contacts / ownership

- Collaborative experiment: user (Quiet-Lab) + Grok  
- Code history: git log on `main` from scaffold → v0.7/v0.8 polish  

When resuming: read `DESIGN_PRINCIPLES.md` first, then `QUICKSTART.md`, then run `pytest -q` and `grmc status`.

---

## One-sentence summary

**GRMC is a conservative memory lab: think freely in reflection, write carefully through human approval, always keep provenance.**
