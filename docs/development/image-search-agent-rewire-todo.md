# ImageSearchAgent → VespaSearchBackend rewire (deferred HIGH, audit cycle 6)

`ImageSearchAgent._search_vespa` (libs/agents/cogniverse_agents/image_search_agent.py)
hand-rolls a raw Vespa `/search/` POST that is comprehensively out of sync with
the deployed image schema `configs/schemas/image_colpali_mv_schema.json`
(document type `image_colpali_mv`). Four independent mismatches, each enough to
make every image search return nothing — all masked by the method's broad
`try/except` that returns `[]`:

1. **Source name** — YQL is `select * from image_content`, but the schema's
   document type is `image_colpali_mv`. No `image_content` type exists.
2. **Rank profiles** — the agent sends `colpali_similarity` (semantic) /
   `hybrid_image` (hybrid); the schema defines `float_float`, `binary_binary`,
   `float_binary`, `phased`, `hybrid_float_bm25`, `hybrid_binary_bm25`. Neither
   name the agent uses exists.
3. **Query input name** — the agent posts `input.query(q)`; the schema's float
   input is `query(qt)` (type `tensor<float>(querytoken{}, v[128])`).
4. **Tensor format** — the agent sends `str(query_embedding.flatten().tolist())`
   (a stringified flat 1-D list); the mapped `querytoken{}` tensor needs the
   dict-of-vectors form `{str(i): vector.tolist() for i, vector in enumerate(emb)}`
   that `VespaSearchBackend` already builds (search_backend.py ~1011-1017).

## Why not a point fix

Fixing only #4 (the format, the originally-flagged item) leaves #1–#3 broken, so
the agent would still return nothing — a "looks fixed but isn't" change. The
mismatches are exactly the things `VespaSearchBackend` already derives from the
schema at query time (it reads `inputs_needed` from the rank profile and encodes
2-D embeddings as the dict tensor). The agent should not maintain a second,
drifting copy of that query-construction logic.

## Plan (NEEDS_LIVE_BOUNDARY — verify against a real Vespa + image schema)

1. Resolve the image `VespaSearchBackend` for the tenant/profile (the same
   `get_backend_registry().get_search_backend(...)` path the text/video search
   uses) instead of `requests.post` to `self._vespa_endpoint`.
2. Encode the query image via `ColPaliQueryEncoder` (already done) and pass the
   2-D embedding straight to the backend's search — let it pick the rank
   profile, query-input name, and tensor encoding from the deployed schema.
3. Map the backend hits to `ImageSearchResult` as today.
4. Delete the hand-rolled YQL/rank-profile/format code in `_search_vespa`.

## Test contract (real Vespa)

Deploy `image_colpali_mv` for a tenant, ingest 2–3 image docs with known
ColPali embeddings, query with an image whose embedding is closest to doc B, and
assert the returned order puts B first (a misleading high-text-overlap doc must
not outrank it). A 400-vs-200 capture confirms the query format is accepted —
the current code 400s (wrong input name / source) and the `except` hides it.

Tracked from audit cycle 6 (`docs/development/audit-6-findings.md`, H-image).
