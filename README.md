# GRMC - Grok Reflective Memory Core

**v0.5.0** — Soft edge suggestions · embedding tension · eval · migrator · edges/provenance · approval · reflection · bridge

## Philosophy

- Prefer **missing** a signal over a **wrong high-confidence** belief
- **Think** (`reflect`) never writes the graph (`mutates_memory=False`)
- **Write** only via human `approve` (nodes **and** edges)
- Provenance: every approved concept should answer *which episodes justify this?*

## Architecture

```
Ingest     → SQLite episodes + Chroma vectors
Reflect    → report + pending concept/edge proposals (no graph write)
Approve    → GraphNode / GraphEdge + episode_node_links
ops eval   → health checks (over-confidence, provenance)
ops migrate-legacy → Chroma → SQLite (additive)
```

## Quick start

```bash
pip install -e .
grmc ingest -t "..." -c "human_oversight,long_term_memory" --embedder hashing
grmc reflect --embedder hashing
grmc propose
grmc propose --kind edge
grmc approve prop_...
grmc node node_... --provenance
grmc edges list
grmc ops eval
```

## Commands

| Command | Role |
|---------|------|
| `grmc reflect` | Think; optional concept + soft edge proposals |
| `grmc propose [--kind edge]` | Approval queue |
| `grmc approve / reject` | Human gate writes |
| `grmc node <id> --provenance` | Why this node? + edges |
| `grmc edges propose/list/types` | Manual edge proposals |
| `grmc ops eval` | Conservative health score |
| `grmc ops migrate-legacy` | Import old Chroma episodes into SQLite |
| `grmc bridge *` | Dual-Grok file channel |

## Docs

- `docs/EDGES_AND_PROVENANCE.md`
- `docs/APPROVAL_AND_SQLITE.md`
- `docs/V05_NOTES.md`
- `docs/BRIDGE.md`

## Tests

```bash
pytest -q
```

## Safety caps (defaults)

| Object | Soft max on approve |
|------|---------------------|
| Node | 0.55 |
| Edge | 0.45 |
| Soft edge suggestion | 0.30 |

## Known limitations

1. Soft edges require **existing** endpoint nodes (will not invent nodes)
2. Embedding tension quality depends on embedder (hashing is weaker)
3. Migrator does not re-build Chroma vectors for migrated rows
4. No multi-hop graph query UI yet
5. No LLM verification yet (feature-flag candidate)

## Next (beyond 0.5)

- LLM-assisted verification (flagged, default off)
- Graph neighborhood / path CLI
- Stronger eval suites & fixtures
- Optional encrypted raw_turns

---

**Conservative · Reflective · Provenance-aware · Human-gated.**
