# DocumentAgent search fix + document_visual ingestion — phased plan

Status: APPROVED (scope) — execute phase by phase, each ships a real-boundary
regression test (fails on the pre-fix code) + lint + commit; ≤5 files/phase.

## Why
`DocumentAgent` is dispatched in production (`agent_dispatcher.py:1586`) and does
dual-strategy PDF search: visual (ColPali page-as-image → `document_visual`),
text (ColBERT extracted-text → `document_text`), hybrid, auto. Every path is
broken against the real schema, and the visual half has **no ingestion producer
at all**. No e2e test existed because there was no data to test against and the
unit tests mocked `requests.post` and asserted the (broken) payload.

Root facts established by investigation:
- `document_text` IS ingested as ColBERT multi-vector (128-d/token, LateOn) via
  `DocumentTextEmbeddingStrategy`. `ColBERTQueryEncoder` (LateOn) exists.
- `document_visual` is NEVER ingested (zero producers); its inline schema is
  mis-modeled — `colpali_embedding tensor<float>(x[1024],d[128])` with an
  element-wise MaxSim, instead of the mapped `patch{}`/`querytoken{}` form
  `image_colpali_mv` uses. No `configs/schemas` JSON for it.
- `_search_text` sends profile `hybrid_bm25_semantic` (absent from
  document_text), input `query(q)` (undeclared), and a 768-d dense sentence
  embedding into a `token{},v[128]` field — wrong model. Reads `source_url`
  (document_text has `document_path`).
- `_search_visual` sends `str(query_embedding.flatten().tolist())` (the H2
  stringify bug) of a variable-token ColPali output against a fixed `x[1024]`.
- Both query the BARE schema name (`document_text`/`document_visual`), not the
  tenant-scoped `<base>_<canonical_tenant>` the registry deploys.
- `search_agent` hardcodes `strategy="binary_binary"` for all modalities (7
  sites); audio has no such profile → audio A2A raises. The backend already
  auto-selects per-profile `default_ranking` when no strategy is passed.

## Phases

### Phase 1 — `_search_text` matches the real ColBERT producer
- Use `ColBERTQueryEncoder` (LateOn) for the query; send mapped
  `{str(i): vec.tolist()}` as `input.query(qt)`; profile `hybrid_float_bm25`;
  tenant-scope the schema (`document_text_<canonical_tenant>`); read
  `document_path` for the url and the real summary fields.
- Test (real Vespa): deploy `document_text_<tenant>`, feed a doc with a known
  ColBERT token embedding + title/full_text/document_path, run `_search_text`,
  assert the matching `document_id` is returned, `document_url == document_path`,
  `strategy_used == "text"`. Controlled embedding (model not the boundary risk).

### Phase 2 — `search_agent` per-modality default strategy
- Drop the 7 hardcoded `binary_binary` fallbacks; pass `kwargs.get("ranking")`
  (None when absent) so the backend auto-selects per profile/type. Audio must
  not raise.
- Test (real Vespa): SearchAgent `_process_impl` for video returns
  `len(results) > 0` with no explicit ranking; an audio-modality call returns
  gracefully (no `ValueError`).
- RISK FOUND DURING EXECUTION (not yet implemented): backend auto-select
  (`search_backend.py:711-730`) RAISES when a profile has >1 strategy and no
  `default_ranking` is resolved in `profile_config` at search time. NOTHING
  currently exercises the no-strategy path (`test_ranking_strategies_real`
  always passes an explicit strategy), so dropping `binary_binary`→None could
  break the working VIDEO path, not just fix audio. DO FIRST: a real-Vespa
  SearchAgent `_process_impl` test (deployed multi-strategy video profile +
  ColPali encoder) that proves auto-select resolves `default_ranking`; only
  then drop the hardcode. If auto-select does NOT resolve, the fix also needs
  `backend.default_profiles.<type>.strategy` config (bigger). The 7 sites:
  search_agent.py:877, 1021, 1147, 1249, 1438, 1458, 1830.

### Phase 3 — `document_visual` schema redesign + `_search_visual` query fix
- Add `configs/schemas/document_visual_schema.json` mirroring `image_colpali_mv`
  (`colpali_embedding tensor<bfloat16>(patch{}, v[128])` + `embedding_binary`,
  profiles `float_float`/`hybrid_float_bm25`, proper per-token MaxSim). Retire
  the mis-modeled inline schema in `vespa_schema_manager.py` (or align it to the
  JSON).
- Fix `_search_visual`: ColPali query encoder → mapped dict (NOT `str(flatten)`)
  → `input.query(qt)`; profile `float_float`/`hybrid_float_bm25`; tenant-scope.
- Test (real Vespa): feed a page doc with a controlled colpali embedding, assert
  `_search_visual` retrieves + ranks it (image-test pattern).

### Phase 4 — page-image ColPali ingestion for `document_visual`
- New `DocumentVisualSegmentationStrategy` (render PDF pages → images) +
  `DocumentVisualEmbeddingStrategy` (ColPali page-image embedding), wired into
  the strategy/pipeline registry; add a `document_visual` processing profile.
- Decisions to confirm at this phase: PDF render lib (pymupdf vs pdf2image),
  page-image source/limits, profile/config shape.
- Test (real ingestion e2e): ingest a small PDF, assert `document_visual` docs
  with `colpali_embedding` land in Vespa and `_search_visual` retrieves them.

## Done criteria
All 4 phases shipped, each with a real-boundary test that fails on the pre-fix
code; document_agent search returns real results end-to-end for text and
(post-Phase-4) visual; `search_agent` audio no longer raises.
