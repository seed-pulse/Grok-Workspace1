# GRMC - Grok Reflective Memory Core

**v0.6.0** — Optional LLM verification (default off) · graph neighborhood · edges/provenance · approval · eval

## Philosophy

- Prefer missing a signal over wrong high-confidence beliefs
- Reflection **never** writes the graph (`mutates_memory=False`)
- Graph writes only via human `approve`
- LLM assist is **opt-in** and report-only

## Quick start

```bash
pip install -e .
grmc ingest -t "..." -c "human_oversight" --embedder hashing
grmc reflect --embedder hashing          # heuristics only
grmc reflect --llm                       # optional LLM (needs API key)
grmc propose / grmc approve prop_...
grmc node node_... --provenance
grmc graph neighbors node_... --depth 2
grmc ops eval
```

## LLM (default OFF)

```bash
export GRMC_LLM=1
export GRMC_LLM_API_KEY=sk-...
export GRMC_LLM_BASE_URL=https://api.x.ai/v1
export GRMC_LLM_MODEL=grok-2-latest
grmc reflect --llm
```

See `docs/LLM_VERIFICATION.md`.

## Graph neighborhood

```bash
grmc graph neighbors node_abc --depth 1
grmc graph neighbors node_abc --depth 2 --type supports
grmc graph neighbors node_abc -o /tmp/nb.json
```

## Safety caps

| Path | Cap |
|------|-----|
| Node approve | 0.55 |
| Edge approve | 0.45 |
| Soft edge suggest | 0.25 |
| LLM concept | 0.50 |
| LLM contradiction | 0.35 |

## Tests

```bash
pytest -q
```

---

**Conservative · Reflective · Provenance-aware · Human-gated · LLM-optional.**
