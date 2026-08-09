# Rejected Approaches

Approaches that were evaluated, prototyped, and deliberately not adopted. Each
entry names the branch the work sits on so a reviewer scanning for unmerged
branches can see the decision rather than re-open it.

**These branches are intentionally unmerged. Do not propose merging them.**

---

## Superlinked Inference Engine (SIE) as the serving layer

**Branch:** `sie-serving-consolidation` (2 commits, intentionally unmerged)
**Decision:** not adopted. The inference tier stays on vLLM.

### What was evaluated

SIE (`github.com/superlinked/sie`, Apache-2.0) as a replacement for the
per-model vLLM pods. One SIE server hosts many models behind a model registry
with lazy load and LRU eviction, exposes `encode` / `score` / `extract` in one
process, and ships a purpose-built ColQwen3 adapter — so it could in principle
collapse the ColQwen3, DenseOn, LateOn and GLiNER pods into one or two.

The two commits on the branch consolidate embedding and GLiNER inference onto
an SIE service, then make vLLM the default with SIE as a configurable
fallback.

### What the prototype established

SIE runs on this hardware: its ColQwen3 adapter emits the expected
`(16, 320)` multivector on gfx1151 when attention is forced to SDPA.
`flash_attention_2` segfaults on ROCm, so a port would have to pin SDPA.

### Why it was not adopted

The consolidation is real but partial, and it buys less than it costs:

- CLAP audio and Whisper ASR have no SIE adapter and stay on separate pods
  regardless, so the pod count never collapses to the number the pitch implies.
- Generative VLM captioning cannot co-reside — generation is pool-isolated.
- SIE is catalog-driven: a new architecture is a YAML entry rather than a model
  string, which moves work rather than removing it.
- It is 0.x software, and adopting it would put the whole retrieval tier behind
  its release cadence.

vLLM already serves every retrieval model the system uses, and the pieces SIE
would have handled server-side (query/document prefixes, `is_query`) are
handled client-side today at no ongoing cost.

### What would reopen this

A concrete pod-count or latency problem that vLLM cannot solve, plus SIE
adapters for the audio and ASR models. Absent both, re-evaluating is not worth
the branch churn.
