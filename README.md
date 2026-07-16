# GRMC - Grok Reflective Memory Core

**v0.4.0** — Graph edges + provenance · SQLite SoR · Approval queue · Reflection · Bridge

Standalone, local-first reflective memory for Grok (and other LLMs).

## Philosophy

- Prefer **missing** a memory over injecting a **wrong high-confidence** belief
- **Think** (reflection) and **write** (approval) are strictly separated
- Graph changes (nodes **and** edges) only after **human `approve`**
- Every approved concept should answer: **which episodes justify this?**

## Architecture

```
Ingest ──► SQLite episodes (SoR) + Chroma vectors

Reflect ──► report (mutates_memory=False) + pending concept proposals

Approve concept ──► GraphNode + episode_node_links (provenance)
Approve edge    ──► GraphEdge (node → node)

edges propose   ──► pending edge proposal only
```

## Quick start

```bash
pip install -e .
grmc ingest -t "Human oversight protects long-term memory." \
  -c "human_oversight,long_term_memory" --embedder hashing
grmc reflect --embedder hashing
grmc propose
grmc approve prop_........ --note "ok"
grmc node node_........ --with-provenance --with-edges

# Edges (second node first)
grmc approve prop_other...
grmc edges propose --from node_a --to node_b --type supports -e ep_...
grmc approve prop_edge...
grmc edges list
```

## Commands (v0.4)

| Command | Role |
|---------|------|
| `grmc reflect` | Think; enqueue concept proposals |
| `grmc propose` | List proposals (`kind` column) |
| `grmc approve <id>` | Write node **or** edge + provenance |
| `grmc approve … --link-to node_x` | Also enqueue related edge (still pending) |
| `grmc node <id>` | Detail + provenance + incident edges |
| `grmc nodes` | List nodes |
| `grmc edges list / propose / types` | Edge inspection & propose |
| `grmc reject <id>` | Dismiss proposal |
| `grmc bridge *` | Dual-Grok file channel |

## Docs

- [`docs/EDGES_AND_PROVENANCE.md`](docs/EDGES_AND_PROVENANCE.md)
- [`docs/APPROVAL_AND_SQLITE.md`](docs/APPROVAL_AND_SQLITE.md)
- [`docs/Reflection_Engine_v0.md`](docs/Reflection_Engine_v0.md)
- [`docs/BRIDGE.md`](docs/BRIDGE.md)

## Tests

```bash
pytest -q
```

## Known limitations

1. Edges are **not** auto-inferred from reflection (manual `edges propose` only for now)
2. No multi-hop path queries / graph visualization UI
3. Provenance is written on concept approve; historical nodes approved before v0.4 lack links unless re-approved/merged
4. Legacy Chroma-only data (pre-0.3) still needs re-ingest
5. Contradiction edges are not auto-created from reflection flags

## Next (v0.5 candidates)

1. Soft edge *suggestions* from reflection contradictions (still pending, low conf)
2. Graph path / neighborhood query CLI
3. Optional LLM verification of edge proposals (feature-flagged)
4. Eval harness for provenance coverage
5. Export graph as JSON/GraphML for external tools

---

**Conservative · Reflective · Provenance-aware · Human-gated.**
