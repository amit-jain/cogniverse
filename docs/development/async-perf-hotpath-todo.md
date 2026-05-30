# Async hot-path perf — batched feed + per-request artefact cache (deferred, audit cycle 6)

Two HIGH/PERF findings on hot paths. Both are perf optimizations where a subtle
bug harms CORRECTNESS (not just latency), so they must be verified against a real
Vespa / real Phoenix before shipping — deferred here with a plan + inline
pointer-TODOs at the sites (the `pipeline-cache-multi-pod-todo.md` model).

## 1. One-doc-per-segment Vespa feed → batch

`embedding_generator_impl._process_multi_documents` (libs/runtime/.../embedding_generator_impl.py)
calls `self._feed_document(doc)` once per segment inside the segment loop;
`_feed_document` (line ~1113) does `backend_client.ingest_documents([document], schema)`
— one Vespa feed per keyframe. With `wait_for_indexing` (backend.py:311-340,
default True) each feed also runs a per-document Document-v1 visibility probe
loop. For an N-keyframe video that is N feeds + N probe loops.

**Plan:** accumulate `Document`s across the segment loop and feed in batches
(e.g. 50–100, plus a final flush) via a single `ingest_documents(batch, schema)`
per batch. Keep the per-segment embedding GC (the loop drops each embedding after
feeding to bound memory — batching must not re-introduce the ~370 KB/frame
pile-up, so cap the batch and clear fed docs after each flush).

**Verify (real Vespa):** ingest an N-keyframe video, assert the same document
count lands (count Document-v1 docs) and the same field values as the
one-by-one path, and that a mid-batch backend 400 surfaces (doesn't silently
drop the rest of the batch).

## 2. `load_for_request` per-request Phoenix reads → short-TTL cache

`ArtifactManager.load_for_request` (artifact_manager.py:785) does 2–3 uncached
telemetry/Phoenix dataset reads on EVERY agent dispatch (get_artefact_state →
load_blob → get_dataset, plus a get_dataset for the canary or active versioned
prompts). The dispatcher calls this per request when an artifact_manager_factory
is wired (agent_dispatcher.dispatch → resolve_artefact_for_request →
load_for_request).

**Plan:** add a short-TTL per-(tenant, agent, variant) cache for the artefact
state blob + resolved prompts. **Invalidation is the correctness-critical part:**
bust the cache on `promote_to_canary` / `promote_canary_to_active` /
`retire_canary` (artifact_manager.py ~670+) so a freshly promoted canary/variant
takes effect immediately — a stale cache would serve the WRONG prompts (and skew
the canary traffic split). State changes are rare vs reads, so a small TTL +
explicit invalidation is safe.

**Verify (real Phoenix):** dispatch twice, assert the second read is served from
cache (no second get_dataset); then promote a canary and assert the next dispatch
reflects it immediately (cache busted). This is exactly the kind of bug a unit
test can't catch — it needs the real dataset store + the real promote path.

Tracked from audit cycle 6 (`docs/development/audit-6-findings.md`, H29 + the
load_for_request PERF finding).
