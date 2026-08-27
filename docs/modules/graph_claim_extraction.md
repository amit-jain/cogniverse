# Graph Claim Extraction

`DocExtractor` reads the validated GLiNER endpoint from system config and runs
`ClaimExtractor` after entity extraction for each segment.

`ClaimExtractionSignature` returns at most four claims, with `claims` as the
final output field.

`ClaimExtractor` raises on malformed or length-capped completions and includes
the source document and segment ID in the error text.

`DocExtractor.extract_claims_from_text(...)` counts failed segments in
`claim_segments_failed`. The ingest path raises when every segment fails
instead of returning an empty graph.
