# Graph Edges + Provenance (v0.4)

## Goals

1. **Traceability** — answer “why does this node exist?” via episode links  
2. **Structure** — relate nodes without silent, high-confidence claims  
3. **Human gate** — edges and nodes both go through the approval queue  

## Edge types (minimal vocabulary)

| Type | Meaning (conservative) |
|------|------------------------|
| `supports` | A lends weight to B |
| `contradicts` | A is in tension with B |
| `related_to` | Soft association (default) |
| `implies` | A suggests B (weak logical lean) |
| `derived_from` | A was abstracted from B |
| `part_of` | A is a component of B |

Default edge confidence cap on approve: **0.45** (stricter than nodes’ 0.55).

## Provenance: episode ↔ node

Table `episode_node_links`:

- written **only when a concept proposal is approved**
- fields: episode_id, node_id, relation (`supports` | `contradicts` | `mentioned_in`),
  proposal_id, report_id, confidence, note
- also updates `episodes.linked_graph_nodes` denormalized list

Query path:

```bash
grmc node node_........ --with-provenance --with-edges
```

## Write paths (safety)

| Action | Graph write? |
|--------|----------------|
| `grmc reflect` | No (may enqueue concept proposals) |
| `grmc edges propose` | No (enqueues edge proposal) |
| `grmc approve prop_…` (concept) | Yes → node + provenance links |
| `grmc approve prop_…` (edge) | Yes → edge only |
| `grmc approve … --link-to node_x` | Node write + **pending** edge proposal (edge still needs approve) |

## Schema

SQLite schema version **2**:

- `graph_edges` (UNIQUE source, target, type)
- `episode_node_links` (UNIQUE episode, node, relation)

Existing v0.3 DBs pick up new tables via `CREATE IF NOT EXISTS` on open.

## CLI

```bash
grmc edges types
grmc edges propose --from node_a --to node_b --type supports -e ep1,ep2
grmc propose                 # shows kind=edge|concept_candidate
grmc approve prop_edge...
grmc edges list
grmc edges list --node node_a
grmc node node_a
```
