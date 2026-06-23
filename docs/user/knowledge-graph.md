# Knowledge Graph

Cogniverse extracts a knowledge graph from any codebase or document corpus you index. Every run of `cogniverse index` produces two things in parallel:

1. **Content index** — the existing semantic search (vectors in Vespa)
2. **Knowledge graph** — nodes (concepts, functions, entities) and typed edges (calls, imports, mentions) in a separate Vespa schema

Both are tenant-scoped and queryable at runtime. The graph answers questions the content index can't: "what connects X to Y?", "what does SearchAgent call?", "what are the hub concepts in this codebase?"

## Commands

### `cogniverse index` (extended)

Extended with graph extraction. No new flags — the existing `--type` flag controls which files are ingested, and graph extraction happens automatically for supported file types.

```bash
cogniverse index ./src --type code   # tree-sitter extraction → nodes + edges
cogniverse index ./docs --type docs  # entity extraction → nodes + edges
```

The `docs` type now fans out per file extension to the right content profile:

| Extension | Content profile | Graph extraction |
|---|---|---|
| `.md` `.txt` `.rst` `.html` | `document_text_semantic` | GLiNER entities + co-mention edges |
| `.pdf` | `document_text_semantic` | PDF text → GLiNER entities |
| `.mp4` `.mov` `.mkv` `.avi` | `video_colpali_smol500_mv_frame` | Whisper transcript + VLM captions → GLiNER entities |
| `.jpg` `.png` `.webp` | `image_colpali_mv` | VLM captions + OCR → GLiNER entities |
| `.wav` `.mp3` `.m4a` | `audio_clap_semantic` | Whisper transcript → GLiNER entities |

Multimodal graph extraction reuses the text that the content pipelines already produce — Whisper transcripts for audio/video, VLM captions for images and keyframes. No extra model calls. After the content pipeline processes a file, the runtime reads its transcript/description outputs and runs the same DocExtractor that text files use.

Code files (`.py`, `.ts`, `.go`, etc.) go to `code_lateon_mv` for content and tree-sitter for graph extraction.

Output shows both content and graph counts:

```
$ cogniverse index ./libs/runtime --type code
Found 47 code files in ./libs/runtime
Indexing ████████████████████ 47/47

Indexed 47/47 files
  Chunks created: 1283
  Documents fed: 1247
  Graph: 312 nodes, 487 edges
```

### `cogniverse graph stats`

Graph statistics: node count, edge count, and top-degree nodes (the hubs).

```bash
cogniverse graph stats
```

```
Knowledge Graph (tenant: default)
  Nodes: 312
  Edges: 487

Top nodes (by degree):
┌─────────────────────┬────────┐
│ Node                │ Degree │
├─────────────────────┼────────┤
│ searchagent         │     23 │
│ codingagent         │     18 │
│ memoryawaremixin    │     15 │
│ vespabackend        │     12 │
└─────────────────────┴────────┘
```

### `cogniverse graph search`

Semantic search over graph nodes. Uses hybrid BM25 + vector ranking on node name + description.

```bash
cogniverse graph search "video retrieval"
cogniverse graph search "authentication" --top-k 5
```

### `cogniverse graph neighbors`

Direct neighbors of a node (edges out and in).

```bash
cogniverse graph neighbors SearchAgent
cogniverse graph neighbors CodingAgent --depth 2
```

Output groups edges by direction and shows each edge's relation and provenance:

```
Neighbors of SearchAgent
  Outgoing:
    → vespabackend (calls, EXTRACTED)
    → codeextractor (imports, EXTRACTED)
    → memoryawaremixin (inherits, EXTRACTED)
  Incoming:
    → routingagent (calls, EXTRACTED)
```

### `cogniverse graph path`

Shortest path between two nodes via BFS traversal of outgoing edges.

```bash
cogniverse graph path SearchAgent Vespa
cogniverse graph path CodingAgent OpenShell --max-depth 4
```

## Graph Model

### Node

Every node, regardless of whether it came from code or docs, has the same shape:

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name (e.g. `SearchAgent`) |
| `node_id` | string | Normalized identifier derived from name (e.g. `searchagent`) |
| `description` | string | Short description (from docstring, caption, or context) |
| `kind` | `entity` \| `concept` | Loose label; `entity` for code symbols, `concept` for extracted doc topics |
| `label` | string | GLiNER entity-type tag (`Person`, `Location`, `Organization`, `Substance`, `Concept`, ...). Defaults to `Concept`. Used by `CrossModalLinker` to gate `same_as` linking — a Person-named transcript mention only co-refers with a Concept/Location VLM caption when the caption contains person-indicator words or shares a name token. |
| `mentions` | list\[Mention\] | Per-segment grounded mentions (source_doc_id, segment_id, ts_start, ts_end, modality, evidence_span) |
| `degree` | int | Number of edges touching this node (computed) |
| `embedding` | tensor(token{}, v[128]) | ColBERT multi-vector (LateOn 128-dim per token) of `name + description` |

The `node_id` is deterministic: "SearchAgent" and "searchagent" normalize to the same id, so the same symbol extracted from different files is a single node with merged `mentions`.

### Edge

Every edge has the same shape:

| Field | Type | Description |
|---|---|---|
| `source_node_id` | string | Normalized source node id |
| `target_node_id` | string | Normalized target node id |
| `relation` | string | Free-text label: `calls`, `imports`, `defines`, `mentioned_with`, etc. |
| `evidence_span` | string | Verbatim text span that grounded this edge |
| `segment_id` | string | Segment (e.g. `frame_0`, `transcript_0`) where the edge was found |
| `ts_start` | float | Segment start timestamp (seconds; 0.0 for non-temporal content) |
| `ts_end` | float | Segment end timestamp |
| `modality` | string | Content modality (`code`, `text`, `video`, `image`, `audio`) |
| `provenance` | `EXTRACTED` \| `INFERRED` | `EXTRACTED` = found structurally (AST); `INFERRED` = LLM guess |
| `source_doc_id` | string | Source file where this edge was found |
| `confidence` | float | 0.0-1.0 confidence score |

The `edge_id` is `sha1(source_node_id | relation | target_node_id | segment_id | ts_start | ts_end)[:16]` — the same (source, relation, target, segment) from two extractors produces the same edge, so upserts are idempotent.

## Extraction

Extractors are an internal detail — every extractor emits the same `Node` / `Edge` shape. The graph manager picks the right extractor per file extension.

### Code extractor (tree-sitter)

Supported languages: Python, JavaScript, TypeScript, Go (via `tree-sitter-python`, `tree-sitter-javascript`, etc., which are already cogniverse runtime deps).

For each file the extractor walks the AST and emits:

| Node type | Source |
|---|---|
| The module itself | File path stem |
| Function/method definitions | `function_definition`, `method_definition`, etc. |
| Class/struct/interface definitions | `class_definition`, `class_declaration`, `interface_declaration`, `struct_item`, `impl_item`, `trait_item` |
| Imported symbols | `import_statement`, `import_from_statement`, `use_declaration` |

| Edge type | Relation | Provenance |
|---|---|---|
| Module → defined symbol | `defines` | EXTRACTED |
| Module → imported symbol | `imports` | EXTRACTED |
| Function → called function | `calls` | EXTRACTED |

All code edges are `EXTRACTED` — these are structural facts, not LLM guesses.

### Doc extractor (GLiNER + regex fallback)

Supported extensions: `.md`, `.txt`, `.rst`, `.html`, `.pdf`.

- **Primary path:** GLiNER (`urchade/gliner_large-v2.1`) predicts entities with labels: Person, Organization, Technology, Concept, Location, Product, Algorithm, Model, Framework, Language
- **Fallback path:** regex for capitalized multi-word phrases when GLiNER is unavailable (stripping leading articles like "The")
- Text is chunked into paragraph-aware blocks of ~2000 chars before extraction

| Node type | Source |
|---|---|
| Named entities | GLiNER prediction |
| Capitalized concepts | Regex fallback |

| Edge type | Relation | Provenance |
|---|---|---|
| Entity A → Entity B (found in same chunk) | `mentioned_with` | INFERRED |

All doc edges are `INFERRED` because co-mention isn't a proven relationship — it's a heuristic.

### Multimodal extractor (video / image / audio)

Unlike the code and doc paths which run in the CLI process, multimodal extraction happens **inside the runtime**, after the content pipeline produces text outputs.

**The flow** (per-segment, with `Mention` provenance + SPO claim edges + cross-modal linking):

1. User runs `cogniverse index ./stuff --type docs` and a `.mp4` is found.
2. CLI uploads it via `POST /ingestion/upload` with the `video_colpali_smol500_mv_frame` profile.
3. Runtime's `VideoIngestionPipeline` processes the file normally — runs Whisper for audio transcription, extracts keyframes, calls the VLM descriptor on each keyframe, generates embeddings, feeds Vespa.
4. After the pipeline returns, `routers/ingestion.py` iterates the result with `_iter_segments_for_graph()`, which yields one `SegmentRecord` per text-emitting source:
   - one per Whisper transcript segment (carries `ts_start`/`ts_end` from the Whisper output)
   - one per VLM keyframe description (`segment_id="frame_<idx>"`, anchored at the frame timestamp)
   - one per OCR/caption block on a keyframe
   - one per document file (PDF / OCR'd page)
5. `_extract_graph_per_segment()` calls `DocExtractor.extract_from_text(text, ..., segment_anchor=mention)` per segment. Each entity GLiNER finds gets a structured `Mention` (source_doc_id + segment_id + ts_start + ts_end + modality + verbatim evidence_span) instead of a bare doc-id string. A cross-segment `entity_pool` is threaded forward so the `ClaimExtractor` resolves pronouns ("She later won the Nobel Prize" → Marie Curie when introduced in an earlier segment).
6. `ClaimExtractor` (DSPy ChainOfThought + `InstrumentedRLM` promotion for inputs > 3000 chars) produces real SPO edges with predicates from a locked 16-element vocabulary (`born_in`, `discovered`, `worked_at`, `won`, `contradicts`, ...). Predicate normalization + a vocabulary filter drop free-form LM emissions ("was_born_in" → `born_in`, "yellow" / "in" / "glass" → dropped).
7. `CrossModalLinker` runs once per `source_doc_id` after all per-segment passes, emitting `same_as` edges between cross-modal `Mention` pairs whose temporal windows overlap within `±temporal_window_s` AND whose Node labels pass the type gate (Person↔Person, or Person↔Concept with shared name token / person-indicator word).
8. The accumulated `ExtractionResult` is `GraphManager.upsert()`'d to the tenant's shared `knowledge_graph_default` schema, then per-segment back-references (`entity_ids` / `relation_ids` / `claim_ids`) are PATCHed onto each content document so a single Vespa join finds every claim grounded in a given segment.

**No new model calls** — the multimodal path reuses Whisper/VLM outputs that the content pipelines already produce. Whether a file gets graph extraction or not depends on whether its pipeline emits text:

| File kind | Source of text | Graph nodes |
|---|---|---|
| Text doc (`.md`, `.txt`, etc.) | File contents | Yes |
| PDF | `PyPDF2` text extraction | Yes |
| Video | Whisper transcript + VLM keyframe captions + optional OCR | Yes |
| Image | VLM caption + optional OCR | Yes (if pipeline produces captions) |
| Audio | Whisper transcript | Yes |
| Silent video / no-caption image | (nothing) | No graph extraction — just content indexing |

**Ingestion response** now includes graph counts so you can see what was extracted per file:

```bash
$ curl -s -X POST http://localhost:28000/ingestion/upload \
    -F "file=@demo.mp4" -F "profile=video_colpali_smol500_mv_frame" -F "tenant_id=default"
{
  "status": "success",
  "video_id": "demo",
  "chunks_created": 47,
  "documents_fed": 47,
  "graph_nodes": 12,
  "graph_edges": 28,
  "processing_time": 34.2
}
```

Graph extraction is fail-safe: if the extractor errors or the GraphManager factory isn't wired, ingestion still succeeds with `graph_nodes: 0` — content indexing is never blocked by graph extraction.

## REST API

The CLI is a thin client over these endpoints at `/graph/`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/graph/upsert` | POST | Batch upsert nodes + edges for a tenant |
| `/graph/search` | GET | Hybrid BM25 + vector search over nodes |
| `/graph/neighbors` | GET | Out/in edges of a node |
| `/graph/path` | GET | Shortest path between two nodes (BFS, default max depth 4, up to 6) |
| `/graph/stats` | GET | Node/edge counts + top-degree nodes |

**Upsert example:**

```bash
curl -X POST http://localhost:28000/graph/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "source_doc_id": "demo.py",
    "nodes": [
      {"name": "Foo", "description": "A class", "kind": "entity"},
      {"name": "Bar", "description": "Another class", "kind": "entity"}
    ],
    "edges": [
      {
        "source": "Foo", "target": "Bar", "relation": "calls",
        "evidence_span": "Foo().bar()", "segment_id": "demo.py",
        "ts_start": 0.0, "ts_end": 0.0, "modality": "code",
        "provenance": "EXTRACTED"
      }
    ]
  }'
```

Response:
```json
{"status": "upserted", "nodes_upserted": 2, "edges_upserted": 1}
```

## Consumer agents (query time)

The knowledge graph built at ingestion is **complementary** to each agent's
own Mem0 memory — not a competing store. Six knowledge agents expose a public
graph method (`KnowledgeGraphTraversalAgent.traverse`,
`TemporalReasoningAgent.compare_over_time`,
`MultiDocumentSynthesisAgent.synthesize`,
`ContradictionReconciliationAgent.detect`,
`KnowledgeSummarizationAgent.summarize`, `CitationTracingAgent.trace`) that
reads the shared, provenance-rich KG.

At dispatch, the tenant's `GraphManager` is bound onto the agent —
`agent_dispatcher._bind_graph_manager` on the orchestrator-routing path, and
`routers/knowledge.py::_bind_graph` on the `/admin/.../knowledge/...` routes.
With the graph bound, the agent's `_process_impl` walks its own Mem0 memory
**and** consults the shared KG, merging the KG result into dedicated output
fields (named below):

| Agent | Output field | Bridge from request |
|---|---|---|
| `KnowledgeGraphTraversalAgent` | `nodes` / `edges` | `start_subject_key` |
| `TemporalReasoningAgent` | `kg_timeline` | `subject_key` |
| `MultiDocumentSynthesisAgent` | `kg_claim_groups` | (query-agnostic; all claims) |
| `ContradictionReconciliationAgent` | `kg_conflict_entries` | `subject_key` + `predicate` |
| `KnowledgeSummarizationAgent` | `kg_video_summaries` | `subject_keys` |
| `CitationTracingAgent` | `kg_primary_sources` | `claim_id` (KG Edge id) |

The complement is fail-safe: with no graph bound (or the backend unconfigured)
the bind is a no-op, the `kg_*` fields stay empty, and the agent returns its
Mem0-only answer. See
[Knowledge System Diagrams → 9-Agent Knowledge Dispatch](../diagrams/knowledge-system-diagrams.md#9-agent-knowledge-dispatch).

## Storage

Each tenant gets its own Vespa schema — `knowledge_graph_<tenant>` — holding both nodes and edges in the same document type, discriminated by a `doc_type` field (`node` or `edge`). The schema is deployed lazily on the first graph upsert or query for a new tenant.

**Namespace:** `graph_content` (Vespa Document v1 API).
**Schema name:** `knowledge_graph_<tenant>` (e.g. `knowledge_graph_acme` for tenant `acme`).
**Queries:** All graph manager operations target the tenant's own schema, so tenants can't see each other's nodes/edges.

## Architecture

```mermaid
flowchart TD
    CLI["<span style='color:#000'><b>cogniverse index ./src</b></span>"] --> Collect["<span style='color:#000'>collect_files</span>"]
    Collect --> Files["<span style='color:#000'>[file1.py, file2.md, file3.mp4, file4.jpg, ...]</span>"]

    Files --> CodePath["<span style='color:#000'><b>CODE / TEXT PATH</b><br/>(local extraction in CLI)</span>"]
    Files --> MultiPath["<span style='color:#000'><b>MULTIMODAL PATH</b><br/>(server-side, video / image / audio)</span>"]

    CodePath --> CodeContent["<span style='color:#000'>POST /ingestion/upload<br/>→ Vespa (code_lateon_mv,<br/>document_text_semantic)</span>"]
    CodePath --> LocalExtract["<span style='color:#000'>CodeExtractor / DocExtractor<br/>(in CLI process)</span>"]
    LocalExtract --> LocalUpsert["<span style='color:#000'>POST /graph/upsert<br/>→ GraphManager.upsert()</span>"]
    LocalUpsert --> Vespa[("<span style='color:#000'><b>Vespa</b><br/>knowledge_graph_&lt;tenant&gt;</span>")]

    MultiPath --> MultiContent["<span style='color:#000'>POST /ingestion/upload<br/>→ Vespa (video_colpali,<br/>image_colpali, audio_clap)</span>"]
    MultiContent --> Pipeline["<span style='color:#000'>VideoIngestionPipeline<br/>runs Whisper / VLM / OCR</span>"]
    Pipeline --> Iter["<span style='color:#000'>_iter_segments_for_graph(result)<br/>yields one SegmentRecord per:<br/>transcript-seg, VLM-keyframe,<br/>OCR-block, document-file</span>"]
    Iter --> PerSegLoop["<span style='color:#000'><b>for each segment:</b></span>"]
    PerSegLoop --> DocExtract["<span style='color:#000'>DocExtractor.extract_from_text(<br/>text, segment_anchor=Mention(...),<br/>prior_entities=&lt;cumulative pool&gt;)</span>"]
    DocExtract --> Claim["<span style='color:#000'><b>ClaimExtractor</b><br/>DSPy CoT, RLM if &gt;3000 chars,<br/>16-predicate vocab → SPO Edges</span>"]
    Claim --> CrossModal["<span style='color:#000'><b>CrossModalLinker.link(combined)</b><br/>same_as edges across modalities<br/>gated by Node.label + ±5s temporal</span>"]
    CrossModal --> MultiUpsert["<span style='color:#000'>GraphManager.upsert(linked_result)</span>"]
    MultiUpsert --> Vespa
    MultiUpsert --> BackRefs["<span style='color:#000'>PATCH content docs<br/>(entity_ids / relation_ids / claim_ids)</span>"]
    BackRefs --> Response["<span style='color:#000'>Response: graph_nodes / graph_edges counts</span>"]
```

**Two extraction paths, one graph.** Code and text files are extracted locally in the CLI process so the runtime doesn't re-read them. Multimodal files (video/image/audio) are extracted server-side, inside the ingestion pipeline, because that's where Whisper/VLM/OCR already run. Both paths write to the tenant's `knowledge_graph_<tenant>` schema in the same way.

## Comparison with graphify

[graphify](https://github.com/safishamsi/graphify) is a Claude Code skill that builds a knowledge graph from any folder. Cogniverse's knowledge graph covers a lot of what graphify does, and a few things that are still gaps.

**Shared features:**

- Tree-sitter code extraction (functions, classes, calls, imports)
- Entity extraction from text docs
- Unified Node + Edge model with typed provenance
- Semantic search over nodes, path queries, stats
- Incremental / idempotent upserts
- Multi-language code support

**What cogniverse has that graphify doesn't:**

- Multi-tenant isolation via `tenant_id`
- Integration with the rest of the cogniverse stack (memory, agents, runtime API)
- Vespa-backed (clustered, persistent) rather than file-based
- Ties to the existing content index — the same `cogniverse index` call feeds both

**What graphify has that cogniverse doesn't (yet):**

- **Community detection** (Leiden clustering) for topic grouping — would need `graspologic`
- **LLM-based edge inference** — currently only simple co-mention; graphify uses Claude to infer typed relationships ("X implements Y", "X depends on Z")
- **Interactive HTML visualization, Obsidian export, Gephi, Neo4j cypher** — cosmetic output formats
- **MCP server, git post-commit hook, watch mode** — integration conveniences
- **Token reduction benchmark** — measuring query-time token savings

## Troubleshooting

**`Graph stats: 0 nodes, 0 edges`** — the extraction ran but either didn't find any entities or the upsert failed. Check runtime logs: `kubectl logs deployment/cogniverse-runtime -n cogniverse -c runtime | grep -i graph`.


**`tree-sitter parser for X unavailable`** — only Python, JavaScript, TypeScript, and Go parsers are bundled. Other code files are silently skipped by the code extractor but still get content-indexed.

**`GLiNER load failed, falling back to no-op`** — GLiNER download failed or the model cache is corrupt. The doc extractor falls back to a regex-based capitalized phrase extractor, which is weaker but still produces nodes. Fix: ensure the HF cache at `/home/cogniverse/.cache/huggingface` is mounted in the runtime pod, or delete it to force a fresh download.

**Entity extracted with the wrong name (e.g. "The ColPali" instead of "ColPali")** — the regex fallback strips leading articles but isn't perfect. Install GLiNER properly for high-quality extraction.
