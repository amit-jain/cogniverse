# Graph Claim Extraction

`DocExtractor` reads the validated GLiNER endpoint from system config and runs
`ClaimExtractor` after entity extraction for each segment.

`ClaimExtractionSignature` returns at most four claims, with `claims` as the
final output field.

`ClaimExtractor` raises on malformed or length-capped completions and includes
the source document and segment ID in the error text.

Its first extraction loads the tenant's compiled state from the artifact
`("model", "claim_extraction")`; a tenant with none runs the base module. A
dataset store that cannot be read fails that extraction with the store's
`DatasetStoreUnavailableError` as the cause, and the next extraction reads the
artifact again.

`DocExtractor.extract_claims_from_text(...)` counts failed segments in
`claim_segments_failed`. The ingest path raises when every segment fails
instead of returning an empty graph.

Segments long enough to promote run through `InstrumentedRLM` under
`rlm_run_span`, which emits one `InstrumentedRLM.run` span per call into the
tenant's project, nested under the KG pass's span. Export is asynchronous: a
collector that refuses connections changes neither the edges nor the call's
latency.
