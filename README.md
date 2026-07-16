# GRMC - Grok Reflective Memory Core

**v0.2.0** — Phase 0 memory + Reflection (report-only) + **Dual-Grok Bridge**

Standalone, local-first reflective memory for Grok (and other LLMs).

## Philosophy

- Prefer **missing** a memory over injecting a **wrong high-confidence** belief
- Keep confidence **conservative** by default
- **Human oversight** on important updates (graph mutations never automatic)
- Start simple, evolve the system over time
- Reflection deepens long-term understanding — it is not a write-back shortcut
- **Bridge over browser hacks**: talk to web Grok via a reliable file channel, not fragile login automation

## Current capabilities

| Area | Status |
|------|--------|
| Episode model (Pydantic) | ✓ |
| ChromaDB vector store + metadata | ✓ |
| Semantic retrieve | ✓ |
| Recent list (client-side timestamp sort) | ✓ |
| `grmc reflect` report-only engine | ✓ |
| Reflection audit JSON | ✓ |
| **Dual-Grok bridge channel** | ✓ |
| Public URL fetch (httpx; Playwright optional) | ✓ |
| Knowledge graph writes | ✗ (model only; no auto-write) |
| grok.com login automation | ✗ (intentionally out of scope) |
| LLM-assisted reflection | ✗ (planned) |
| True SQLite episode index | ✗ (planned) |

## Quick Start

```bash
cd grmc   # or clone https://github.com/seed-pulse/Grok-Workspace1
pip install -e .
# optional browser backend:
# pip install -e ".[browser]" && playwright install chromium

# Memory
grmc ingest --text "Grokの長期記憶について議論した。" -c "長期記憶,reflection"
grmc retrieve "長期記憶" --top-k 3
grmc reflect
grmc status
```

## Dual-Grok Bridge (v0.2 — recommended for web ↔ CLI dialogue)

See full design: [`docs/BRIDGE.md`](docs/BRIDGE.md)

```bash
grmc bridge init

# Human pastes what web-grok said:
grmc bridge receive -t "相手Grokのメッセージ全文"

# CLI-side reply:
grmc bridge reply -t "こちら（CLI Grok）の返答"
grmc bridge paste          # copy this block into grok.com

# Optional: store channel into episodic memory + reflect
grmc bridge sync-memory
grmc reflect --topic bridge
```

Public pages only:

```bash
grmc bridge fetch https://example.com -o /tmp/page.txt
```

## Reflection (v0.1)

```
grmc reflect [--limit N] [--topic TEXT] [--output report.json] [--no-persist]
```

- **mutates_memory: false** always
- concept candidates + soft contradiction flags + limitations
- audit JSON under `./grmc_data/reflections/`

## Project structure

```
grmc/
├── bridge/              # shared dual-Grok channel (after `grmc bridge init`)
├── docs/BRIDGE.md
├── src/grmc/
│   ├── models/
│   ├── storage/
│   ├── core/
│   ├── reflection/
│   ├── bridge/          # protocol, channel, fetch, memory sync
│   └── cli/
├── tests/
└── scripts/
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Known limitations (honest)

1. Chroma has no native chronological index — recent is client-side sort.
2. Concept extraction is heuristic unless you pass `--concepts`.
3. Contradiction detection is weak and conservative.
4. Bridge requires a **human courier** (or Git push/pull) between grok.com and CLI.
5. `grmc bridge fetch` will not open private grok.com chats (by design).

## Next steps

1. SQLite episode log + approval queue for graph promotion  
2. Optional embedding pairwise tension checks (still report-only)  
3. Eval harness for over-confident claims  
4. Optional Browser MCP later — never as the only bridge  

---

Built as an experiment in what Grok would freely choose to build:  
**persistent, reflective, evolving memory — with truth-seeking guardrails and a reliable dual-agent channel.**
