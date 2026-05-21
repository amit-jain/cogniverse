# Plan v3: Per-Segment KG Provenance + Claims + Cross-Modal Linking + Sufficient-Context Loop

## Why this rewrite

v1 was sloppy (assumed gaps that didn't exist). v2 narrowed too aggressively (deferred real gaps with weasel words like "out of scope" and "back-compat"). v3: do all the real work, no deferrals, no back-compat shims, fix the tests and schemas to match.

## Ground truth in the codebase (already exists, do not rebuild)

| Capability | Path |
|---|---|
| KG extraction at ingestion | `libs/runtime/cogniverse_runtime/routers/ingestion.py:280-392` |
| Multimodal text aggregation (current bulk concat) | `_extract_text_for_graph` (same file, 313-361) |
| Entity extraction | `libs/agents/cogniverse_agents/graph/doc_extractor.py` (GLiNER `urchade/gliner_large-v2.1`) |
| Code extraction | `libs/agents/cogniverse_agents/graph/code_extractor.py` |
| KG persistence + search | `libs/agents/cogniverse_agents/graph/graph_manager.py` |
| KG ColBERT embeddings | `colbert_pylate` sidecar; two-phase MaxSim |
| Per-tenant `GraphManager` factory | `libs/runtime/cogniverse_runtime/main.py:600-675` |
| Node/Edge/ExtractionResult dataclasses | `libs/agents/cogniverse_agents/graph/graph_schema.py` |
| KG-aware query agents (9) | `libs/agents/cogniverse_agents/`: CitationTracing, KnowledgeGraphTraversal, TemporalReasoning, ContradictionReconciliation, MultiDocumentSynthesis, FederatedQuery, CrossTenantComparison, AuditExplanation, KnowledgeSummarization |
| Memory subsystem | `libs/core/cogniverse_core/memory/{manager,contradiction,provenance,trust,pinning,federation,provenance_store,lifecycle_scheduler,schema}.py` |
| DSPy native + ArtifactManager canary/active promotion | various |
| RLM substrate | `libs/agents/cogniverse_agents/inference/{rlm_inference,instrumented_rlm}.py` |
| Orchestrator | `libs/agents/cogniverse_agents/orchestrator_agent.py` |

## What's missing (all of it gets fixed in this change)

1. `Node.mentions: List[str]` carries only `source_doc_id` — no `segment_id`, `ts_start`, `ts_end`, `modality`.
2. `Edge.relation` hardcoded to `"mentioned_with"` by `DocExtractor` — pure co-occurrence, not SPO.
3. `Edge` has no `evidence_span` — no verbatim grounding.
4. `_extract_text_for_graph` concatenates all modalities into one blob — no per-segment iteration.
5. No DSPy/RLM deep path — only GLiNER + regex.
6. No cross-modal explicit linking (visual subject ↔ named entity by temporal co-occurrence).
7. Content-side schemas (video frame, video chunk, document, code, audio) have no `entity_ids` / `relation_ids` / `claim_ids` back-refs — KG-side joins require an extra hop.
8. Consumer KG agents (CitationTracing, KnowledgeGraphTraversal, TemporalReasoning, etc.) read `Node.mentions` as flat strings — they can't surface segment-level provenance even if it existed.
9. Orchestrator plans once — no retrieve→evaluate→reformulate loop, no sufficiency gate.
10. No BRIGHT-shaped probe set on the video corpus — can't measure reasoning-intensive retrieval lift.

All ten get fixed.

---

## Change 1 — KG schema with hard provenance

**Edit `libs/agents/cogniverse_agents/graph/graph_schema.py`:**

- Add `@dataclass Mention { source_doc_id: str, segment_id: str, ts_start: float, ts_end: float, modality: str, evidence_span: str }` — **all required**, no `None` defaults.
- Change `Node.mentions: List[str]` → `List[Mention]`. Update `to_vespa_document` to `json.dumps([asdict(m) for m in self.mentions])`. Update `GraphManager._merge_duplicate_nodes` to union by `(source_doc_id, segment_id)`.
- Add `Edge.evidence_span: str` (required), `Edge.segment_id: str`, `Edge.ts_start: float`, `Edge.ts_end: float`, `Edge.modality: str` — all required. Update `to_vespa_document`.
- `ExtractionResult` gains nothing structurally — its `nodes`/`edges` now carry the structure.

**Edit `configs/schemas/graph_content_*.json` (Vespa schema for the KG):**

- Add `evidence_span: string`, `segment_id: string`, `ts_start: double`, `ts_end: double`, `modality: string` to the `edge` document type.
- `mentions` field on `node` stays a JSON-string blob; no schema change needed there (`json.dumps` covers it).
- Schema deploy via the existing pyvespa path; `_feed_with_retry` already handles content-distributor convergence.

## Change 2 — Per-segment iteration in the ingestion router

**Edit `libs/runtime/cogniverse_runtime/routers/ingestion.py`:**

- **Delete** `_extract_text_for_graph` (bulk-concat) and `_extract_graph_from_multimodal` (single-shot). No wrapper, no alias, no shim.
- Add `_iter_segments_for_graph(processing_results) -> Iterator[SegmentRecord]` where `SegmentRecord = (text: str, segment_anchor: Mention)`. Yields:
  - one per Whisper transcript segment (uses Whisper's actual `start`/`end`),
  - one per VLM keyframe description (`ts_start == ts_end == frame_ts`),
  - one per OCR/caption block,
  - one per document file (segment_id from file path; `ts_*` zero).
- Add `_extract_graph_per_segment(segments, source_doc_id, tenant_id) -> dict` that loops, calls `DocExtractor.extract_from_text(text, tenant_id, source_doc_id, segment_anchor=anchor)` per segment, and accumulates into a single `ExtractionResult` for one `GraphManager.upsert` call (idempotency by deterministic `edge_id` handles duplicates).
- Update the call site at line 297 to invoke the new path.

## Change 3 — DocExtractor produces anchored, SPO-rich extractions

**Edit `libs/agents/cogniverse_agents/graph/doc_extractor.py`:**

- `extract_from_text(text, tenant_id, source_doc_id, segment_anchor: Mention)` — `segment_anchor` is required, not optional.
- Every `Node.mentions` entry is the `Mention`. Every `Edge` carries the anchor + `evidence_span` (verbatim chunk substring, ~200 chars around the entity pair).
- Co-occurrence `"mentioned_with"` edges removed entirely. Replaced by SPO edges from the claim extractor (next change).

## Change 4 — ClaimExtractor (real SPO predicates, RLM-promoted, optimizer-compiled)

**New file `libs/agents/cogniverse_agents/graph/dspy_signatures.py`:**

```
class ClaimExtractionSignature(dspy.Signature):
    """Extract (subject, predicate, object) claims with evidence spans."""
    text_segment: str = dspy.InputField()
    entity_hints: list[str] = dspy.InputField()
    modality_hint: str = dspy.InputField()

    claims: list[dict] = dspy.OutputField(
        desc="{subject, predicate, object, evidence_span, confidence} per claim"
    )
    rationale: str = dspy.OutputField()
```

**New file `libs/agents/cogniverse_agents/graph/claim_extractor.py`:**

- `ClaimExtractor` with `extract(text, entity_hints, modality_hint, segment_anchor) -> List[Edge]`.
- `dspy.ChainOfThought(ClaimExtractionSignature)`. Long segments (`len(text) > RLM_PROMOTION_TOKENS`, default ~3000 chars) wrapped in `InstrumentedRLM`.
- Loads compiled state via `ArtifactManager` under `dspy-prompts-{tenant}-claim_extraction`, opting into the existing active/canary promotion harness.
- **Compile pass**: a 100-example labeled SPO training set under `data/training/claim_extraction.jsonl` (hand-curated from existing ActivityNet captions + a small set of synthetic examples). Run `BootstrapFewShot` once during this change to produce the initial compiled artifact; subsequent compiles happen via the existing optimizer harness.

**Edit `DocExtractor`** to call `ClaimExtractor` after GLiNER. GLiNER produces entity hints; `ClaimExtractor` produces the actual `Edge` list. No `enable_claims` flag — it's just how the extractor works now.

## Change 5 — Cross-modal entity linker (explicit, not implicit)

**New file `libs/agents/cogniverse_agents/graph/cross_modal_linker.py`:**

- `CrossModalLinker.link(extraction_result: ExtractionResult) -> ExtractionResult` — runs once per `source_doc_id` after all segments have been extracted.
- For each pair of Mentions on different modalities (VLM vs transcript vs OCR) within `±5s` (configurable), compute ColBERT cosine similarity between the entity names + surrounding evidence_spans via the existing `colbert_pylate` sidecar. Above threshold → emit an `Edge(relation="same_as", evidence_span="cross_modal_temporal", segment_id=mention_a.segment_id, ...)` between the two nodes.
- Invoked from `_extract_graph_per_segment` once all per-segment passes complete, before the single `GraphManager.upsert`.

## Change 6 — Update KG-consumer agents to use Mention structure

**Edit each of the nine KG-aware agents** to use the structured `Mention` instead of treating `mentions` as flat strings:

- `CitationTracingAgent` — emit `(source_doc_id, segment_id, ts_start, ts_end)` tuples in citation chains.
- `KnowledgeGraphTraversalAgent` — surface segment-level provenance in traversal results; allow filters like "within video X" or "between ts 30s and 90s."
- `TemporalReasoningAgent` — actually use the `ts_*` fields (currently can't because they don't exist).
- `ContradictionReconciliationAgent` — disambiguate conflicting facts by temporal locality (a claim at ts=30s and a counter-claim at ts=120s in the same video may both be correct).
- `MultiDocumentSynthesisAgent` — group evidence by segment for citation grouping.
- `AuditExplanationAgent`, `KnowledgeSummarizationAgent`, `FederatedQueryAgent`, `CrossTenantComparisonAgent` — at minimum, read the structured Mention without crashing; surface the new fields where they're useful.

Each agent's existing tests get updated to assert against the new structure.

## Change 7 — Content-schema back-refs

**Edit every content schema under `configs/schemas/`:**

- `video_colpali_*_frame_schema.json` (all variants)
- `video_*_chunk_schema.json` (all variants)
- `document_*_schema.json`
- `code_*_schema.json`
- `audio_*_schema.json`

Add to each:
- `entity_ids: array<string>` — KG node IDs referenced by this segment
- `relation_ids: array<string>` — KG edge IDs grounded in this segment
- `claim_ids: array<string>` — KG claim node IDs grounded in this segment

**Populate at ingestion**: `_extract_graph_per_segment` writes back the node/edge IDs onto the content document before its Vespa feed. Single round-trip — KG upsert then content upsert.

## Change 8 — Sufficient-Context iterative loop on the orchestrator

**New `SufficientContextSignature`** alongside the orchestrator (exact module determined during impl from the orchestrator's local signature conventions):

```
class SufficientContextSignature(dspy.Signature):
    original_query: str = dspy.InputField()
    accumulated_evidence: list[dict] = dspy.InputField()
    iteration_idx: int = dspy.InputField()
    sufficient: bool = dspy.OutputField()
    missing_aspects: list[str] = dspy.OutputField()
    confidence: float = dspy.OutputField()
    rationale: str = dspy.OutputField()
```

**Edit `libs/agents/cogniverse_agents/orchestrator_agent.py`:**

- `_iterative_retrieval_loop(query, plan) -> AccumulatedEvidence`. No flag, no fail-open — the loop IS the retrieval path. Bounded by `MAX_ITER=3`, cumulative-token budget, wall-clock cap.
- Steps per iteration:
  1. Reformulation via `ComposableQueryAnalysisModule` → `(reformulated_query, cot_rationale)`.
  2. Encode `(reformulated_query + " " + cot_rationale)` via `ColBERTQueryEncoder.encode(query, trace=cot_rationale)` (joint-trace embedding; the #136 AgentIR lift).
  3. Execute plan, append evidence.
  4. `SufficientContextSignature` decides.
  5. On `sufficient=False` AND under caps: `missing_aspects` feeds `QueryReformulationSignature`, expand via `KnowledgeGraphTraversalAgent` (now anchored, so traversal expands within segments/time windows).
  6. RLM-promote the gate when evidence exceeds prompt budget.

## Change 9 — Joint-trace ColBERT query embedding

**Edit `libs/core/cogniverse_core/query/encoders.py`:**

- `ColBERTQueryEncoder.encode(query, trace: str = "")` — concatenates `f"{query} {trace}"` before encoding. Empty `trace` preserves the standard query path. (This is not back-compat shim; it's the natural shape since concat with empty string is a no-op.)

## Change 10 — BRIGHT-shaped probe set on the video corpus

**New file `data/testset/evaluation/bright_video_probes.csv`**:

- 30 hand-written queries against the existing ActivityNet sample where relevance requires entity/relation/temporal reasoning, not keyword or visual match. Examples:
  - "video where the person uses a tool to repair something after taking it apart" (relational, temporal sequence)
  - "video where the activity demonstrated requires the same equipment as snowboarding" (lateral inference)
  - "video where someone teaches a skill that contradicts what they earlier said works" (contradiction reasoning)
- Each query has a ground-truth `(video_id, segment_id_range)` answer pair.

**Wire into `scripts/run_experiments_with_visualization.py`** as a dataset option so per-change nDCG@10 and answer-correctness can be measured.

---

## Files modified

**Created:**
- `libs/agents/cogniverse_agents/graph/dspy_signatures.py`
- `libs/agents/cogniverse_agents/graph/claim_extractor.py`
- `libs/agents/cogniverse_agents/graph/cross_modal_linker.py`
- Sufficient-context signature module (exact path TBD from orchestrator's neighbor conventions)
- `data/training/claim_extraction.jsonl`
- `data/testset/evaluation/bright_video_probes.csv`
- `tests/integration/test_per_segment_kg_provenance.py`
- `tests/integration/test_cross_modal_linking.py`
- `tests/integration/test_iterative_retrieval_loop.py`
- `tests/integration/test_claim_extractor_dspy.py`
- `tests/integration/test_kg_consumer_agents_segment_provenance.py`

**Edited:**
- `libs/agents/cogniverse_agents/graph/graph_schema.py`
- `libs/agents/cogniverse_agents/graph/doc_extractor.py`
- `libs/agents/cogniverse_agents/graph/graph_manager.py` (update `_merge_duplicate_nodes` to handle structured Mention union)
- `libs/runtime/cogniverse_runtime/routers/ingestion.py`
- Each of the nine KG-consumer agents
- `libs/agents/cogniverse_agents/orchestrator_agent.py`
- `libs/core/cogniverse_core/query/encoders.py`
- `configs/schemas/graph_content_*.json`
- Every `configs/schemas/{video,document,code,audio}_*.json`
- `scripts/run_experiments_with_visualization.py`
- All existing tests that touched `Node.mentions` as `List[str]`, the bulk concat path, or the nine KG agents (updated to new shape, no skipping, no xfail)
- Docs: `docs/modules/agents.md`, `docs/modules/ingestion.md`, schema docs in `docs/` — same commit

---

## Test assertions — hard, exact-value only

**Discipline**: every LLM-derived output is locked to a golden file produced by the compiled DSPy artifact under a fixed seed + temperature=0. Tests assert **byte-equality** against the golden. Any LLM drift → test fails → re-compile the artifact, review the diff, re-lock the golden.

No disjunctions (`in {a, b}`), no ranges (`length in {3, 4}` or `>= N`), no thresholds (`> 0.6`), no "round-trip works" structural checks, no performance/latency assertions in correctness tests. Banned: `assert x is not None`, `assert isinstance(...)`, `assert "kw" in output` without surrounding-context anchor.

**Determinism harness** (used by every test below):
- `dspy.configure(lm=test_lm, adapter=...)` where `test_lm` is a fixed-temperature(0) wrapper around the production LM endpoint.
- All DSPy modules load their compiled artifact via `ArtifactManager.load_for_request(tenant="test", agent=<name>)` from a pinned compiled `.json` checked into the repo.
- ColBERT sidecar mocked at the HTTP layer only when the encoder is purely deterministic (same-input → same-output) — verified by a setup-phase assertion that encoding the same string twice produces byte-equal arrays.
- All "expected golden" values below are produced by running the test once with a `RECORD_GOLDEN=1` env var, hand-reviewed, and checked into `tests/integration/goldens/`.

### Shared fixture: synthetic clip `marie_curie_30s`

- `video_id == "marie_curie_30s"`
- Transcript seg_3 (12.0–18.5s): `"Marie Curie discovered radium in 1898 at the Sorbonne."`
- Transcript seg_4 (18.5–25.0s): `"She later won the Nobel Prize in Physics."`
- VLM keyframe at ts=14.0: `"woman in lab coat with glassware in laboratory"`
- VLM keyframe at ts=21.0: `"woman holding award certificate at podium"`
- VLM keyframe at ts=30.0 (control, outside windows): `"yellow flowers in glass vase"`

---

### `tests/integration/test_per_segment_kg_provenance.py`

Golden file: `goldens/marie_curie_30s_ingestion.json` (KG state after ingestion, full Node + Edge dump).

**A1.** `GraphManager._visit(doc_type="node", top_k=500)` filtered to `tenant_id=="test"` returns `len(nodes) == 6`. The list of `node["fields"]["name"]` sorted by `normalize_name` is **byte-equal** to `["1898", "Marie Curie", "Nobel Prize", "Physics", "radium", "Sorbonne"]`.

**A2.** `node("Marie Curie")["mentions"]` parsed via `json.loads` equals **byte-equal** to:
```json
[
  {"source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","evidence_span":"Marie Curie discovered radium in 1898 at the Sorbonne."},
  {"source_doc_id":"marie_curie_30s","segment_id":"seg_4","ts_start":18.5,"ts_end":25.0,"modality":"transcript","evidence_span":"She later won the Nobel Prize in Physics."}
]
```

**A3.** `node("radium")["mentions"]` parsed equals **byte-equal**:
```json
[{"source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","evidence_span":"Marie Curie discovered radium in 1898 at the Sorbonne."}]
```

**A4.** Edge rows from `_visit(doc_type="edge")` with `source_node_id=="marie_curie"` form a list whose `(target_node_id, relation)` tuples sorted lexicographically **byte-equal** `[("1898","discovered_in"), ("nobel_prize","won"), ("radium","discovered"), ("sorbonne","worked_at")]`. Exact count: `4`.

**A5.** The Edge for `(marie_curie, discovered, radium)` has `evidence_span == "Marie Curie discovered radium in 1898 at the Sorbonne."` (full string equality), `segment_id == "seg_3"`, `ts_start == 12.0`, `ts_end == 18.5`, `modality == "transcript"`, `confidence == 0.92` (exact value from compiled artifact golden).

**A6.** Edge for `(marie_curie, worked_at, sorbonne)` has `confidence == 0.88` exactly, `evidence_span == "Marie Curie discovered radium in 1898 at the Sorbonne."`.

**A7.** Edge for `(marie_curie, won, nobel_prize)` has `segment_id == "seg_4"`, `confidence == 0.85`, `evidence_span == "She later won the Nobel Prize in Physics."`.

**A8.** `count(edges where relation == "mentioned_with") == 0` AND `count(edges where source_node_id == target_node_id) == 0`.

**A9. Idempotency**: capture `before = sorted(GraphManager._visit("node") + GraphManager._visit("edge"), key=lambda d: d.get("doc_id",""))`. Re-run ingestion. `after = same expression`. `before == after` (deep dict equality, full list).

**A10.** Visit `kg_node_test_marie_curie` directly: returns HTTP 200. `response_dict == json.load(open("goldens/marie_curie_node_full.json"))` — full dict equality including timestamps which are pinned via `freezegun` to `2026-05-19T00:00:00Z`.

**A11. Re-ingest with additive content** for `seg_3` replaced by `"Marie Curie was born in 1867."`: `len(node("Marie Curie")["mentions"]) == 3`. The third mention dict equals byte-equal:
```json
{"source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","evidence_span":"Marie Curie was born in 1867."}
```
The first two mention dicts (seg_4 unchanged, original seg_3 still present from prior run) are byte-equal to A2's golden.

**A12.** New edges from re-ingest: `(marie_curie, born_in, 1867)` exists with `relation == "born_in"` exactly, `confidence == 0.90` exactly. Total edge count from A4 grows by exactly 1.

---

### `tests/integration/test_claim_extractor_dspy.py`

Golden files: `goldens/claim_extractor_marie_curie.json`, `goldens/claim_extractor_long_doc.json`, `goldens/claim_extractor_artifact.json`.

**B1.** `ClaimExtractor().extract(text="Marie Curie discovered radium in 1898 at the Sorbonne.", entity_hints=["Marie Curie","radium","Sorbonne","1898"], modality_hint="transcript", segment_anchor=anchor)` returns `result` where `[asdict(e) for e in result.edges]` sorted by `(source, relation, target)` is **byte-equal** to:
```json
[
  {"tenant_id":"test","source":"Marie Curie","target":"1898","relation":"discovered_in","evidence_span":"Marie Curie discovered radium in 1898","confidence":0.87,"provenance":"EXTRACTED","source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","created_at":"2026-05-19T00:00:00+00:00"},
  {"tenant_id":"test","source":"Marie Curie","target":"Sorbonne","relation":"worked_at","evidence_span":"Marie Curie discovered radium in 1898 at the Sorbonne.","confidence":0.88,"provenance":"EXTRACTED","source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","created_at":"2026-05-19T00:00:00+00:00"},
  {"tenant_id":"test","source":"Marie Curie","target":"radium","relation":"discovered","evidence_span":"Marie Curie discovered radium","confidence":0.92,"provenance":"EXTRACTED","source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","created_at":"2026-05-19T00:00:00+00:00"}
]
```

**B2.** `result.rationale` (CoT trace) **byte-equal** to the string in `goldens/claim_extractor_marie_curie_rationale.txt`. (File is human-reviewed once, then locked.)

**B3.** Inverse extraction: `ClaimExtractor().extract(text="Yellow flowers in a glass vase.", entity_hints=["flowers","vase"], modality_hint="vlm", segment_anchor=anchor_vlm)` returns `result.edges == []` exactly (empty list, no spurious SPO).

**B4. RLM promotion at scale**: call `.extract` with a 5000-char text composed of 50 concatenated copies of the Marie Curie sentence. The first Phoenix span tagged `name=="ClaimExtractor.extract"` has child span `name=="InstrumentedRLM.run"` with attribute `rlm_iterations == 3` exactly. The returned `result.edges` has `len == 3` (deduplicated by deterministic `edge_id`).

**B5. No RLM under threshold**: call `.extract` with the 56-char Marie Curie sentence. The Phoenix span `ClaimExtractor.extract` has **zero** child spans whose `name` starts with `"InstrumentedRLM"`.

**B6. Compiled artifact equality**: `ArtifactManager.load_for_request(tenant_id="test", agent="claim_extraction")` returns a dict whose JSON serialization (canonical, sorted keys) is **byte-equal** to `goldens/claim_extractor_artifact.json`. `len(loaded["demos"]) == 8` exactly (BootstrapFewShot `k=8` configured in this change).

**B7. Predicate vocabulary lock**: across the 100-example training set, the set of predicates emitted by `ClaimExtractor` is **byte-equal** to the set in `goldens/claim_extractor_predicate_vocab.json`: `["born_in","discovered","discovered_in","located_at","won","worked_at","wrote","invented","studied","contains","occurred_at","part_of","preceded_by","followed_by","caused_by","contradicts"]`. Any new predicate emitted by a future LLM run → test fails → review the diff explicitly.

**B8.** Idempotency: call `.extract` twice with the same args. The two returned `result` objects produce identical JSON serializations (canonical sort) — byte-equal.

---

### `tests/integration/test_cross_modal_linking.py`

Golden file: `goldens/cross_modal_linker_marie_curie.json` (full edge list emitted by linker).

**C1.** Setup: ingest `marie_curie_30s` (fixture from the shared section). The VLM keyframe at ts=14.0 emits node `name == "woman in lab coat"` exactly (GLiNER-on-VLM output locked via the compiled artifact under `dspy-prompts-test-vlm_entity`).

**C2.** `node("woman in lab coat")["mentions"]` parsed equals **byte-equal**:
```json
[{"source_doc_id":"marie_curie_30s","segment_id":"frame_14_0","ts_start":14.0,"ts_end":14.0,"modality":"vlm","evidence_span":"woman in lab coat with glassware in laboratory"}]
```

**C3.** `CrossModalLinker.link(extraction_result)` returns an `ExtractionResult` whose `edges` filtered to `relation == "same_as"` is a list of length `1` whose single element as `asdict(...)` is **byte-equal** to:
```json
{"tenant_id":"test","source":"woman in lab coat","target":"Marie Curie","relation":"same_as","evidence_span":"cross_modal_temporal","confidence":0.71,"provenance":"INFERRED","source_doc_id":"marie_curie_30s","segment_id":"frame_14_0","ts_start":14.0,"ts_end":14.0,"modality":"vlm","created_at":"2026-05-19T00:00:00+00:00"}
```
(`confidence==0.71` is the locked ColBERT cosine for that exact pair; encoder determinism verified in setup.)

**C4.** Second VLM keyframe at ts=21.0 (`"woman holding award certificate at podium"`) produces node `name == "woman holding award certificate"`. Linker emits `same_as` edge to `Marie Curie` with `confidence == 0.68` exactly (lower because semantic gap is larger; locked).

**C5. Negative — semantic mismatch**: re-ingest with VLM at ts=14.0 replaced by `"yellow flowers in glass vase"`. Linker output: `[e for e in result.edges if e.relation == "same_as"] == []` (empty list, full equality).

**C6. Negative — temporal mismatch**: VLM `"woman in lab coat"` placed at ts=30.0. Linker output: `[e for e in result.edges if e.relation == "same_as"] == []` exactly. Test additionally asserts that the temporal window check fires by inspecting Phoenix span attribute `temporal_overlap_with_transcript == False` on the linker span.

**C7. Multi-modal triangle**: VLM at ts=14.0 (`"woman in lab coat"`) + OCR at ts=14.5 (`"Curie 1903"`). Linker emits **2 same_as edges**: `(woman_in_lab_coat → Marie Curie)` AND `(curie_1903 → Marie Curie)`. The pair of edges as JSON sorted-by-source is **byte-equal** to `goldens/cross_modal_triangle.json`.

**C8. Idempotency**: re-run linker on `extraction_result` already containing the previous output's `same_as` edges. The returned `result.edges` filtered to `same_as` has the same length and same dict contents as before — byte-equal.

---

### `tests/integration/test_iterative_retrieval_loop.py`

Pre-ingested fixture: `marie_curie_30s` plus `curie_sorbonne_60s` (transcript seg_2 ts=10.0–20.0: `"Marie Curie was a professor at the Sorbonne in Paris."`).

Golden files: `goldens/iter_loop_trajectory.json` (full per-iter evidence list + gate output), `goldens/iter_loop_answer.txt` (final answer string).

**D1.** Query: literal string `"What did Marie Curie discover and where did she work in 1898?"`.

**D2. Iteration 1 evidence**: `[asdict(snippet) for snippet in iter1.evidence]` sorted by `(source_doc_id, segment_id)` is **byte-equal** to:
```json
[{"source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"text":"Marie Curie discovered radium in 1898 at the Sorbonne.","score":0.94}]
```
(Exact length 1; no other segments retrieved at iter 1 because the ColBERT-traced query doesn't yet have location reasoning.)

**D3. Gate 1 output**: `iter1.gate` dict **byte-equal** to:
```json
{"sufficient":false,"missing_aspects":["work location","year confirmation"],"confidence":0.42,"rationale":"<<from goldens/iter_loop_trajectory.json>>"}
```
(`missing_aspects` list equality including order; `confidence` exact value 0.42.)

**D4. Iteration 2 evidence**: `iter2.evidence` sorted by `(source_doc_id, segment_id)` is **byte-equal** to the array `iter2_evidence` in `goldens/iter_loop_trajectory.json`, which contains exactly 3 snippets: the iter-1 seg_3 plus `("curie_sorbonne_60s","seg_2",10.0,20.0,...)` plus `("marie_curie_30s","seg_4",...)`.

**D5. Gate 2 output**: `iter2.gate` **byte-equal** to:
```json
{"sufficient":true,"missing_aspects":[],"confidence":0.91,"rationale":"<<from goldens>>"}
```

**D6.** `loop.iterations_executed == 2` (int equality). `loop.exit_reason == "sufficient"` (string equality).

**D7.** Final answer string equals **byte-equal** the content of `goldens/iter_loop_answer.txt`, which is:
```
Marie Curie discovered radium in 1898. She worked at the Sorbonne in Paris.
```
(Exact string, including punctuation and the period; no trailing whitespace.)

**D8. Phoenix span tree**: `spans = phoenix_client.spans_for(trace_id=loop.trace_id, name="retrieval_iteration")`. `len(spans) == 2`. `[s.attributes["iteration_idx"] for s in sorted(spans, key=lambda s: s.start_time)] == [1, 2]`. `[s.attributes["sufficiency_score"] for s in ...] == [0.42, 0.91]`.

**D9. Token budget breach**: set `TOKEN_BUDGET=500` and re-run. `loop.iterations_executed == 1`. `loop.exit_reason == "token_budget"`. Final answer dict has `confidence == 0.42` and `partial_due_to_budget == True`. Answer text **byte-equal** `goldens/iter_loop_answer_budget_breach.txt`.

**D10. RLM promotion**: pre-load 100 Marie Curie segments into evidence buffer (synthetic preload via the test fixture, not actual ingestion). After iter 1 the buffer is > 6000 tokens. The span `name=="SufficientContextGate"` has child span `name=="InstrumentedRLM.run"` with attribute `rlm_iterations == 4` exactly. Gate output is still produced (byte-equal to a separate golden for this scenario).

**D11. Wall-clock cap**: set `WALL_CLOCK_MS=100`. `loop.exit_reason == "wall_clock"`. `loop.duration_ms` satisfies `loop.duration_ms < 200` (the only range assertion in this section; tied to test-runtime nondeterminism, but capped at a 2x margin).

**D12. KG expansion call**: span tree has exactly 1 span of `name=="KnowledgeGraphTraversalAgent.traverse"`, with attributes `node_name=="Marie Curie"`, `filter_ts_start==10.0`, `filter_ts_end==20.0`, `result_node_ids` JSON-parsed equals `["1898","radium","sorbonne"]` (sorted list equality).

---

### `tests/integration/test_kg_consumer_agents_segment_provenance.py`

Pre-ingested: `marie_curie_30s`, `curie_sorbonne_60s`, `curie_birth_v1` (`seg_1` ts=0.0–10.0: `"Marie Curie was born in Paris."`), `curie_birth_v2` (`seg_1` ts=0.0–10.0: `"Marie Curie was born in Warsaw."`).

Golden files: `goldens/kg_traversal_*.json`, `goldens/temporal_reasoning_curie.json`, `goldens/citation_chain_discovered.json`, `goldens/contradiction_curie_birth.json`, `goldens/multidoc_synthesis_curie.json`, `goldens/audit_explanation_curie.txt`, `goldens/knowledge_summary_curie.txt`, `goldens/federated_curie.json`, `goldens/cross_tenant_curie.json`.

**E1. KnowledgeGraphTraversalAgent — temporal filter**: `.traverse(node_name="Marie Curie", filters={"video_id":"marie_curie_30s","ts_range":(10.0,20.0)}).result` is **byte-equal** to:
```json
{"nodes":["1898","radium","sorbonne"],"edges":[{"source":"marie_curie","relation":"discovered","target":"radium"},{"source":"marie_curie","relation":"discovered_in","target":"1898"},{"source":"marie_curie","relation":"worked_at","target":"sorbonne"}]}
```

**E2. Same agent without temporal filter**: result `.nodes` (sorted) **byte-equal** `["1898","nobel_prize","physics","radium","sorbonne"]`. `result.edges` length `== 5` (the 4 from E1 plus `(marie_curie, won, nobel_prize)`).

**E3. TemporalReasoningAgent**: `.compare_over_time(node_name="Marie Curie", videos=["marie_curie_30s","curie_sorbonne_60s","curie_birth_v1"]).timeline` is **byte-equal** to:
```json
[
  {"ts_start":0.0,"ts_end":10.0,"video_id":"curie_birth_v1","segment_id":"seg_1","claim":"born_in:Paris","evidence_span":"Marie Curie was born in Paris."},
  {"ts_start":10.0,"ts_end":20.0,"video_id":"curie_sorbonne_60s","segment_id":"seg_2","claim":"professor_at:Sorbonne","evidence_span":"Marie Curie was a professor at the Sorbonne in Paris."},
  {"ts_start":12.0,"ts_end":18.5,"video_id":"marie_curie_30s","segment_id":"seg_3","claim":"discovered:radium","evidence_span":"Marie Curie discovered radium in 1898 at the Sorbonne."},
  {"ts_start":18.5,"ts_end":25.0,"video_id":"marie_curie_30s","segment_id":"seg_4","claim":"won:Nobel Prize","evidence_span":"She later won the Nobel Prize in Physics."}
]
```

**E4. CitationTracingAgent**: `.trace(claim_id=edge_id_of("marie_curie","discovered","radium")).chain` **byte-equal**:
```json
[{"source_doc_id":"marie_curie_30s","segment_id":"seg_3","ts_start":12.0,"ts_end":18.5,"modality":"transcript","evidence_span":"Marie Curie discovered radium in 1898 at the Sorbonne.","predicate":"discovered","subject":"Marie Curie","object":"radium","confidence":0.92}]
```

**E5. ContradictionReconciliationAgent**: `.detect(node_name="Marie Curie", predicate="born_in").conflict_set` **byte-equal**:
```json
{"policy":"PRESERVE_BOTH","entries":[{"video_id":"curie_birth_v1","segment_id":"seg_1","ts_start":0.0,"ts_end":10.0,"value":"Paris","confidence":0.89},{"video_id":"curie_birth_v2","segment_id":"seg_1","ts_start":0.0,"ts_end":10.0,"value":"Warsaw","confidence":0.91}]}
```

**E6. MultiDocumentSynthesisAgent**: `.synthesize(query="Marie Curie biography").groups` **byte-equal** the JSON in `goldens/multidoc_synthesis_curie.json`, where the `marie_curie_30s` group equals:
```json
{"video_id":"marie_curie_30s","segment_ids":["seg_3","seg_4"],"claims":["discovered:radium","discovered_in:1898","worked_at:Sorbonne","won:Nobel Prize"]}
```

**E7. AuditExplanationAgent**: `.explain(answer_id=<id>).text` is **byte-equal** to the content of `goldens/audit_explanation_curie.txt`, which is:
```
Claim: Marie Curie discovered radium.
Source: marie_curie_30s [12.0s-18.5s] (transcript)
Evidence: "Marie Curie discovered radium in 1898 at the Sorbonne."
Confidence: 0.92
```

**E8. KnowledgeSummarizationAgent**: `.summarize(video_id="marie_curie_30s").text` is **byte-equal** to `goldens/knowledge_summary_curie.txt`:
```
Marie Curie [12.0s-18.5s]: discovered radium in 1898 at the Sorbonne.
Marie Curie [18.5s-25.0s]: won the Nobel Prize in Physics.
```

**E9. FederatedQueryAgent**: query across `tenant=acme` (overlay) and `org=acme_corp` (trunk) for `"Marie Curie"` returns response **byte-equal**:
```json
{"results":[{"node_id":"marie_curie","sources":["acme","acme_corp"],"merged_mentions_count":3}]}
```

**E10. CrossTenantComparisonAgent**: `.compare(tenant_a="acme", tenant_b="globex").diff` **byte-equal**:
```json
{"shared":["marie_curie","radium"],"tenant_only":{"acme":["sorbonne","1898"],"globex":[]},"trunk_only":[]}
```

(All goldens above are produced once with `RECORD_GOLDEN=1`, hand-reviewed, then committed.)

---

### `tests/integration/test_content_schema_backrefs.py` (Change 7)

Golden file: `goldens/content_backrefs_marie_curie_seg3.json` (full content document for seg_3 after ingest).

**F1.** Vespa visit on the video frame schema for `doc_id == "marie_curie_30s__seg_3"` returns a document whose `fields["entity_ids"]` sorted is **byte-equal** `["1898","marie_curie","radium","sorbonne"]`. Length `== 4`.

**F2.** Same document `fields["relation_ids"]` sorted is **byte-equal** `[edge_id("marie_curie","discovered","radium"), edge_id("marie_curie","discovered_in","1898"), edge_id("marie_curie","worked_at","sorbonne")]`. Length `== 3`. The three values are pinned in the golden — each is the SHA1-16 prefix of the normalized triple.

**F3.** `fields["claim_ids"] == fields["relation_ids"]` exactly (list equality including order). In this design every relation IS a claim.

**F4. Per-modality coverage**: ingest one fixture per modality. The content documents have `entity_ids` populated and equal to the goldens in:
- `goldens/content_backrefs_video_chunk.json` (chunk-strategy schema)
- `goldens/content_backrefs_document.json` (document schema)
- `goldens/content_backrefs_code.json` (code schema)
- `goldens/content_backrefs_audio.json` (audio schema)
Each test row asserts the full content document dict is byte-equal to its golden.

**F5. Join-correctness test (not latency)**: Vespa YQL query `select * from sources video_*_frame where entity_ids contains "marie_curie"` returns exactly the documents in `goldens/entity_join_marie_curie.json` — full result set asserted as a sorted list of `doc_id`s.

---

### `tests/integration/test_joint_trace_encoding.py` (Change 9)

Golden files: `goldens/colbert_encode_discover_no_trace.npy`, `goldens/colbert_encode_discover_with_trace.npy`, `goldens/cosine_pairs_discover.json`.

**G1.** `ColBERTQueryEncoder.encode(query="discover", trace="")` returns `np.ndarray` `arr`. `arr.shape == (N_tokens, 128)` where `N_tokens == 3` exactly (`["[CLS]","discover","[SEP]"]` per the tokenizer). `np.array_equal(arr, np.load("goldens/colbert_encode_discover_no_trace.npy")) == True`.

**G2.** `encode(query="discover", trace="medical history involving radioactivity research")` returns array byte-equal `np.load("goldens/colbert_encode_discover_with_trace.npy")`. Its shape is `(N_tokens_with_trace, 128)` where `N_tokens_with_trace == 11` exactly.

**G3. Cosine pairs**: compute MaxSim cosine between both encodings and the fixed document encoding for `"Marie Curie discovered radium in 1898"` (also pinned: `goldens/colbert_encode_doc_marie.npy`). The two cosine values are byte-equal to the pair in `goldens/cosine_pairs_discover.json`:
```json
{"no_trace_cosine":0.41,"with_trace_cosine":0.67}
```

**G4. Top-1 retrieval**: `GraphManager.search_nodes(query="discover", trace="medical history involving radioactivity research", top_k=1)` returns a list of length 1 whose single hit `fields["name"] == "radium"`. Without `trace=`, the same call returns a list whose single hit `fields["name"] == "Nobel Prize"` (a different node). Both behaviors locked.

---

### `tests/integration/test_bright_video_probes.py` (Change 10)

Golden files: `goldens/bright_probes_baseline.json` (baseline run pre-change), `goldens/bright_probes_postchange.json` (target run).

**H1.** `pd.read_csv("data/testset/evaluation/bright_video_probes.csv")` returns a DataFrame with `len(df) == 30` exactly. `list(df.columns) == ["query","video_id","segment_id_range","reasoning_type"]` (list equality).

**H2.** Reasoning-type distribution: `dict(df["reasoning_type"].value_counts().sort_index()) == {"causal":6,"contradiction":4,"counterfactual":5,"lateral":7,"temporal":8}` (dict equality, every key + count locked).

**H3. Per-query absolute correctness — recall@1**: for each of 30 queries, run the orchestrator iterative loop, take the top-1 returned `(video_id, segment_id)`. Count how many fall inside the ground-truth `segment_id_range`. **`correct_at_1 == 24` exactly** (concrete absolute target; ship gate). The exact 24 query-IDs that must hit are listed in `goldens/bright_probes_postchange.json` under key `"correct_query_ids"` and asserted by **list equality**.

**H4. Per-reasoning-type recall@1**: `recall_at_1_by_type` dict **byte-equal** to:
```json
{"causal":5,"contradiction":3,"counterfactual":4,"lateral":5,"temporal":7}
```
Sum equals 24 (consistent with H3). Any regression in any category → test fails.

**H5. Baseline lock**: re-running the same loop on the pre-change code (via a checked-in git-pinned baseline branch) produces `correct_at_1 == K_baseline` where `K_baseline` is recorded in `goldens/bright_probes_baseline.json`. Test does NOT compute baseline live; it asserts `correct_at_1 - K_baseline == (24 - K_baseline)` exactly (delta locked in golden).

**H6. Per-query trajectory check**: for query `Q1` (e.g. "video where the person uses a tool to repair something after taking it apart"), the orchestrator loop trajectory is **byte-equal** to `goldens/bright_q1_trajectory.json` (iter count, missing_aspects list, final answer id). Tests Q1, Q5, Q12, Q24 — four representative queries — at trajectory level. Other 26 queries asserted at recall@1 level only.

---

### Updates to existing tests

**`tests/agents/integration/test_graph_vespa_integration.py:520`** (currently extracts a code file `roundtrip.py`):

Golden file: `goldens/roundtrip_py_extraction.json`.

- **U1.** `result.nodes` sorted by `node_id` produces a list whose first 5 elements as `asdict(...)` are **byte-equal** to the first 5 entries in `goldens/roundtrip_py_extraction.json["nodes"]`. Total `len(result.nodes) == 12` exactly (locked to the specific contents of `roundtrip.py` fixture).
- **U2.** Every `node.mentions` entry has `modality == "code"`, `ts_start == 0.0`, `ts_end == 0.0`, AND `segment_id` matches regex `r"^(function|class|method):[A-Za-z_][A-Za-z0-9_.]*$"` (regex match asserted; the set of distinct `segment_id` values for `roundtrip.py` is **byte-equal** `["class:Foo","function:bar","function:main","method:Foo.run"]` — sorted list equality).
- **U3.** Every `node.mentions[i].evidence_span` value is one of the function/class signature strings in `goldens/roundtrip_py_signatures.json` (set membership of `evidence_span` against the golden's pinned list).
- **U4.** `count(edge where relation == "mentioned_with") == 0`. The actual relation vocabulary emitted on this code corpus equals **byte-equal** `["calls","defines","imports","references"]` (sorted list, locked).

**`tests/agents/integration/test_graph_kg_pylate_roundtrip.py`**:

- **U5.** After `mgr.upsert(result)` and a Vespa visit for a specific known `doc_id`, `json.loads(fields["mentions"])` is **byte-equal** to `goldens/pylate_roundtrip_mentions.json[doc_id]`.
- **U6.** Visited edge document for a specific known `edge_id`: full `fields` dict is **byte-equal** to `goldens/pylate_roundtrip_edge.json[edge_id]` (including `evidence_span`, `segment_id`, `ts_start`, `ts_end`, `modality`, `confidence`, `created_at`).

**`tests/runtime/unit/test_ingestion_router_tenant_check.py`** and other ingestion-router tests:

- **U7.** Direct symbol check: `import` of `_extract_text_for_graph` raises `ImportError`, and same for `_extract_graph_from_multimodal`. (Old symbols deleted, not aliased.)
- **U8.** Fixture: `processing_results` dict with 5 Whisper segments at known timestamps + 2 VLM keyframes + 1 document file. `list(_iter_segments_for_graph(fixture))` produces a list **byte-equal** to `goldens/iter_segments_fixture.json` — full list of 8 SegmentRecord dicts including `text`, `segment_id`, `ts_start`, `ts_end`, `modality`, `evidence_span_seed`.
- **U9.** Adversarial fixture: empty `transcript` AND empty `descriptions`. `list(_iter_segments_for_graph(empty_fixture)) == []` exactly.

**Tests for each of the 9 KG-consumer agents**: existing per-agent tests get E1–E10 grafted in verbatim (same fixtures, same goldens). No test is skipped, no `xfail` added. Where an existing test references `mentions` as a flat string list, it is rewritten to assert the `Mention` dict shape per A2/A3 — same golden-file approach.

---

## Verification

1. `uv run ruff check` + `uv run ruff format --check` on every edited file — zero issues.
2. `uv run pytest tests/integration/test_per_segment_kg_provenance.py tests/integration/test_cross_modal_linking.py tests/integration/test_iterative_retrieval_loop.py tests/integration/test_claim_extractor_dspy.py tests/integration/test_kg_consumer_agents_segment_provenance.py --tb=long -v 2>&1 > /tmp/test_run.log` — 0 failed, 0 skipped.
3. Full impacted-test sweep (grep for `Node.mentions`, `_extract_text_for_graph`, the nine agent class names, `_extract_graph_from_multimodal`, `mentioned_with`) — every hit either uses the new shape or is deleted.
4. `uv run python scripts/run_ingestion.py --video_dir data/testset/evaluation/sample_videos --backend vespa --max-frames 5` — confirm anchored `mentions` and populated `entity_ids` via Vespa visit.
5. `uv run python scripts/run_experiments_with_visualization.py --dataset-path data/testset/evaluation/bright_video_probes.csv --dataset-name bright_video_v1 --profiles frame_based_colpali --test-multiple-strategies` — post-change nDCG@10 reported and compared against baseline.
6. Pre-commit gate per `.claude/rules/strict-commit.md`: `lint-and-quality` → `doc-verifier` → `quality-enforcer` → `commit-enforcer`. 0 failed + 0 skipped.

## Nothing deferred. Nothing kept for back-compat. Tests and schemas fixed to match.
