# SQLite SoR + Approval Queue (v0.3)

## Separation of concerns

| Layer | Role | Writes graph? |
|-------|------|----------------|
| **Reflection** | Think — report + optional pending proposals | No |
| **Approval queue** | Human review of proposals | No until approve |
| **`grmc approve`** | Explicit graph write | **Yes** |
| **ChromaDB** | Vector search only | No |
| **SQLite** | Episodes, reflection history, proposals, nodes | SoR |

## SQLite schema (essentials)

- `episodes` + index on `timestamp DESC`
- `reflection_reports` (full JSON + metadata)
- `proposals` (`pending` / `approved` / `rejected`)
- `graph_nodes` (created only via approve)

Path: `{data_dir}/grmc.db`  
Chroma: `{data_dir}/chroma/`

## Commands

```bash
grmc reflect                 # think; enqueue proposals
grmc propose                 # list pending
grmc approve prop_xxx        # write GraphNode (confidence capped)
grmc reject prop_xxx
grmc nodes                   # list graph
grmc list / grmc status
```

## Safety defaults

- `mutates_memory=False` on every reflection report
- Approve caps confidence at **0.55** by default (`--cap`)
- Duplicate pending labels are not re-enqueued
- Existing graph nodes with same label: merge supports, still respect cap
