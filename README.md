# GRMC - Grok Reflective Memory Core

**v0.7.0** — Graph path · LLM audit log · export/dump · optional LLM · neighborhood · edges/provenance

## Philosophy

- Prefer missing a signal over wrong high-confidence beliefs  
- Reflection never writes the graph (`mutates_memory=False`)  
- Graph writes only via human `approve`  
- LLM assist is opt-in (default **OFF**) and fully audited when used  

## Commands (highlights)

```bash
# Think
grmc reflect --embedder hashing
grmc reflect --llm                 # needs API key; logged

# Write (human only)
grmc propose / grmc approve prop_...

# Read graph
grmc graph neighbors node_a --depth 2
grmc graph path node_a node_b --depth 3
grmc node node_a --provenance

# Ops
grmc ops eval
grmc ops llm-log
grmc ops export --format md -o dump.md
grmc ops dump
grmc ops migrate-legacy
```

## Safety caps

| Path | Cap |
|------|-----|
| Node approve | 0.55 |
| Edge approve | 0.45 |
| Soft edge suggest | 0.25 |
| LLM concept | 0.50 |
| LLM contradiction | 0.35 |

## Docs

- `docs/LLM_VERIFICATION.md`
- `docs/EDGES_AND_PROVENANCE.md`
- `docs/V07_NOTES.md`

## Tests

```bash
pytest -q
```

---

**Conservative · Reflective · Provenance-aware · Human-gated · LLM-optional & audited.**
