# LLM Verification (v0.6) — feature flag, default OFF

## Enable

```bash
# Environment
export GRMC_LLM=1
export GRMC_LLM_API_KEY=...
export GRMC_LLM_BASE_URL=https://api.openai.com/v1   # or https://api.x.ai/v1
export GRMC_LLM_MODEL=gpt-4o-mini                    # or grok-2-latest

# CLI (forces on for this run)
grmc reflect --llm
grmc reflect --no-llm   # force off even if env set
```

Omit flag → env only; unset env → **fully off**.

## What it does (report only)

When enabled, during `reflect`:

1. **Concept enrichment** — merge LLM labels with heuristics; conf capped (default ≤ 0.50)
2. **Contradiction review** — may drop weak flags; conf capped (default ≤ 0.35)

It does **not**:

- write GraphNodes or GraphEdges
- auto-approve anything
- raise confidences above hard caps

`mutates_memory` remains **False**.

## Failure mode

Any HTTP/JSON/key error → log a note, keep heuristic results, continue.

## Provider

OpenAI-compatible `POST {base_url}/chat/completions` with `response_format=json_object`.
Works with OpenAI and xAI (and similar gateways).
