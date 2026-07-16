# GRMC v0.7

## Graph path

```bash
grmc graph path node_a node_b
grmc graph path node_a node_b --depth 3 --max-paths 3
grmc graph path node_a node_b --type supports
grmc graph path node_a node_b -o path.json
```

- Undirected traversal over stored edges
- Max depth **3**
- Returns shortest length path(s), up to `--max-paths`
- Optional provenance for nodes on the path
- **Read-only** — never writes graph

## LLM audit

When LLM is enabled (`GRMC_LLM=1` / `--llm`), each call is appended to:

`{data_dir}/llm_audit/calls.jsonl`

Fields: model, purpose, success/error, latency, token estimate (or provider usage).

```bash
grmc ops llm-log
grmc status   # includes audit summary line
```

Default remains **LLM OFF** (no cost).

## Export / dump

```bash
grmc ops export --format md
grmc ops export --format json -o dump.json
grmc ops dump -o overview.md
```

Read-only overview of episodes, nodes, edges, pending proposals.

## Fixtures

`tests/fixtures/scenario_conservative.json` — fixed episode set for demos/eval.
