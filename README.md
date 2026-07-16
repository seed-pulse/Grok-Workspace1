# GRMC - Grok Reflective Memory Core

**v0.3.0** — SQLite system of record + Approval Queue + Reflection (report-only) + Bridge

Standalone, local-first reflective memory for Grok (and other LLMs).

## Philosophy

- Prefer **missing** a memory over injecting a **wrong high-confidence** belief
- **Think** (reflection) and **write** (approval) are strictly separated
- Graph changes only after **human `approve`**
- Chroma = vectors only; **SQLite** = chronology, proposals, graph, history

## Architecture (v0.3)

```
Ingest ──► SQLite episodes (SoR, timestamp index)
       └─► Chroma embeddings (semantic search)

Reflect ──► ReflectionReport (mutates_memory=False)
        └─► pending proposals (no graph write)

Approve ──► GraphNode in SQLite   ← only graph write path
```

## Quick start

```bash
cd grmc
pip install -e .
# or: pip install -e ".[dev]"

grmc ingest -t "長期記憶は continuity に重要" -c "長期記憶,continuity" --embedder hashing
grmc list -n 5
grmc reflect --recent -n 20 --embedder hashing
grmc propose
grmc approve prop_........     # first graph write
grmc nodes
grmc status
```

## Commands

| Command | Role |
|---------|------|
| `grmc ingest` | Episode → SQLite + Chroma |
| `grmc retrieve` | Semantic search (Chroma) |
| `grmc list` | Recent episodes (SQLite index) |
| `grmc reflect` | Think / report; enqueue proposals |
| `grmc propose` | List or `--add` pending proposals |
| `grmc approve <id>` | **Write** GraphNode (capped conf) |
| `grmc reject <id>` | Dismiss proposal |
| `grmc nodes` | List graph nodes |
| `grmc status` | Counts + last reflection |
| `grmc bridge *` | Dual-Grok file channel |

## Approval examples

```bash
grmc reflect --embedder hashing
grmc propose
grmc approve prop_abc123def456 --note "looks solid"
grmc approve prop_... --cap 0.5 --type belief
grmc reject prop_... --note "too noisy"
grmc propose --add "self_model_continuity"
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Docs

- [`docs/APPROVAL_AND_SQLITE.md`](docs/APPROVAL_AND_SQLITE.md)
- [`docs/Reflection_Engine_v0.md`](docs/Reflection_Engine_v0.md)
- [`docs/BRIDGE.md`](docs/BRIDGE.md)

## Known limitations

1. Old Chroma-only data (pre-0.3 under `grmc_data/` root) is not auto-migrated to SQLite — re-ingest or migrate manually.
2. Concept extraction remains heuristic unless `--concepts` is set.
3. Contradiction detection is still weak / conservative.
4. No multi-user auth; local single-operator approval.
5. Graph edges (relations between nodes) not yet modeled.

## Next ideas

1. Graph **edges** + provenance links episode↔node  
2. Embedding pairwise tension in reflection  
3. Optional LLM verification (feature-flagged)  
4. Eval harness for over-confidence  
5. Migration tool from legacy Chroma-only stores  

---

Built as a long-term memory experiment: **conservative, reflective, human-gated.**
