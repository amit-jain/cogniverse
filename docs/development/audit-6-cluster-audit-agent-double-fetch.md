# Audit Cycle 6 — Cluster: AuditExplanationAgent double memory fetch

Review summary for the PERF finding in `audit_explanation_agent.py`.

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `_process_impl` (lines 213 + 253) | PERF | the source-building loop fetched each memory via `_fetch_memory`, then the contradiction pass re-fetched the **same** memories (`src.memory_id == node.memory_id`) — every source fetched twice on the `include_contradictions` happy path | the loop caches results in `fetched_memories`; the contradiction pass reads from that cache instead of re-fetching |

## Test (`tests/agents/unit/test_audit_explanation_agent.py`, fails on pre-fix)

`test_contradiction_pass_adds_no_extra_memory_fetches` runs `_process_impl`
with `include_contradictions` off and on against a counting `mm.memory.get`,
and asserts the per-id fetch counts are **identical** — the contradiction pass
adds zero fetches. Pre-fix the on-run added one fetch per source.

## Deeper follow-up (out of this finding's scope, noted so it isn't lost)

`ProvenanceWalker.walk` (`core/memory/provenance.py:284`) already fetches every
visited memory to build the per-node `content_excerpt`, and `_process_impl`
then fetches each again at line 215 for trust extraction. Eliminating that
walker↔agent double fetch needs `CitationNode` to carry the full memory (or at
least the trust block) — a cross-module change touching `citation_tracing_agent`
too. Tracked as a separate item, not folded into this bounded fix.
