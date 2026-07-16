# GRMC v0.5 Notes

## Confirmed from v0.4

- Graph edges + episode provenance + human-gated approve path

## Added in v0.5

1. **Soft edge suggestions** from reflection (only if both nodes already exist)
   - types limited to `supports` / `contradicts` / `related_to` on the soft path
   - enqueued as **pending edge proposals** (`source=reflection-soft`)
   - confidence capped at **0.30** before human approval

2. **Embedding pairwise tension** (report-only)
   - high cosine similarity + differing negation polarity
   - method=`embedding_polarity`, low confidence

3. **Eval harness** — `grmc ops eval`
   - provenance coverage, confidence soft-caps, reflection non-mutation

4. **Legacy migrator** — `grmc ops migrate-legacy`
   - Chroma → SQLite episodes (additive; no graph writes)

## Safety invariants

- `mutates_memory=False` on every reflection report
- No automatic graph node/edge writes
- Soft suggestions never raise confidence aggressively
