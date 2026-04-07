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
| `.mp4` `.mov` `.mkv` `.avi` | `video_colpali_smol500_mv_frame` | Content only (graph TBD) |
| `.jpg` `.png` `.webp` | `image_colpali_mv` | Content only (graph TBD) |
| `.wav` `.mp3` `.m4a` | `audio_clap_semantic` | Content only (graph TBD) |

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
cogniverse graph path CodingAgent OpenShell --max-depth 6
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
| `mentions` | list\[str\] | Source document IDs where this node appears |
| `degree` | int | Number of edges touching this node (computed) |
| `embedding` | tensor(768) | nomic-embed-text vector of `name + description` |

The `node_id` is deterministic: "SearchAgent" and "searchagent" normalize to the same id, so the same symbol extracted from different files is a single node with merged `mentions`.

### Edge

Every edge has the same shape:

| Field | Type | Description |
|---|---|---|
| `source_node_id` | string | Normalized source node id |
| `target_node_id` | string | Normalized target node id |
| `relation` | string | Free-text label: `calls`, `imports`, `defines`, `mentioned_with`, etc. |
| `provenance` | `EXTRACTED` \| `INFERRED` | `EXTRACTED` = found structurally (AST); `INFERRED` = LLM guess |
| `source_doc_id` | string | Source file where this edge was found |
| `confidence` | float | 0.0-1.0 confidence score |

The `edge_id` is `sha1(source_node_id | relation | target_node_id)` — two extractors finding the same relationship produce the same edge, so upserts are idempotent.

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

## REST API

The CLI is a thin client over these endpoints at `/graph/`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/graph/upsert` | POST | Batch upsert nodes + edges for a tenant |
| `/graph/search` | GET | Hybrid BM25 + vector search over nodes |
| `/graph/neighbors` | GET | Out/in edges of a node |
| `/graph/path` | GET | Shortest path between two nodes (BFS, max depth 6) |
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
      {"source": "Foo", "target": "Bar", "relation": "calls", "provenance": "EXTRACTED"}
    ]
  }'
```

Response:
```json
{"status": "upserted", "nodes_upserted": 2, "edges_upserted": 1}
```

## Storage

One Vespa schema — `knowledge_graph` — holds both nodes and edges in the same document type, discriminated by a `doc_type` field (`node` or `edge`). Tenant isolation is enforced by the `tenant_id` field on every document, not by schema naming.

Why one schema instead of per-tenant? Vespa refuses to deploy an application package that removes an existing schema without a `validation-overrides.xml` allowlist. Per-tenant schemas would conflict with each other on every new tenant because the deploy would implicitly "remove" the other tenants' schemas. A single schema with `tenant_id` filtering avoids that entirely and matches how Mem0 and the wiki work internally.

**Namespace:** `graph_content` (Vespa Document v1 API).
**Schema name:** `knowledge_graph_default` (the shared name used by all tenants).
**Queries:** All graph manager operations filter by `tenant_id == <tenant>` before returning results, so tenants can't see each other's nodes/edges.

## Architecture

```
cogniverse index ./src --type code
    │
    ▼
  collect_files → [file1.py, file2.py, ...]
    │
    ├── content: upload to runtime /ingestion/upload
    │       └── Vespa (code_lateon_mv_<tenant>)
    │
    └── graph (local CLI extraction):
           │
           ▼
        CodeExtractor (tree-sitter) / DocExtractor (GLiNER)
           │
           ▼  [{nodes: [...], edges: [...]}]
        POST /graph/upsert
           │
           ▼
        GraphManager.upsert()
           │
           ├── merge_duplicate_nodes()
           ├── generate_embedding(name + description)
           └── feed_node/feed_edge via /document/v1/graph_content/
                  │
                  ▼
               Vespa (knowledge_graph_default)
```

Extraction runs locally in the CLI process so the runtime doesn't need to re-read the files. Only the resulting `(nodes, edges)` are shipped over HTTP.

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
- **Multimodal graph extraction** from image/video — content indexing works but graph extraction is code + text only. The plan is to read existing content pipeline outputs (transcriptions, captions) for graph extraction without adding new VLM calls.
- **Interactive HTML visualization** — graphify outputs `graph.html` (vis.js), Obsidian vault, Gephi graphml, Neo4j cypher
- **God node metrics beyond degree** — centrality, betweenness, etc.
- **Watch mode / git post-commit hook** — auto-rebuild on file changes
- **MCP server** — expose the graph as an MCP endpoint for other agents
- **Token reduction benchmark** — measuring query-time token savings vs reading raw files
- **LLM-based edge inference** — currently only simple co-mention; graphify uses Claude to infer typed relationships ("X implements Y", "X depends on Z")

## Troubleshooting

**`Graph stats: 0 nodes, 0 edges`** — the extraction ran but either didn't find any entities or the upsert failed. Check runtime logs: `kubectl logs deployment/cogniverse-runtime -n cogniverse -c runtime | grep -i graph`.

**`Schema 'knowledge_graph_default' is removed in content cluster`** — a previous deploy accidentally tried to create per-tenant schemas. Fixed in the current version by using a single shared schema with `tenant_id` filtering. If you see this on an old cluster, re-run `cogniverse up` or apply the current Helm chart.

**`tree-sitter parser for X unavailable`** — only Python, JavaScript, TypeScript, and Go parsers are bundled. Other code files are silently skipped by the code extractor but still get content-indexed.

**`GLiNER load failed, falling back to no-op`** — GLiNER download failed or the model cache is corrupt. The doc extractor falls back to a regex-based capitalized phrase extractor, which is weaker but still produces nodes. Fix: ensure the HF cache at `/home/cogniverse/.cache/huggingface` is mounted in the runtime pod, or delete it to force a fresh download.

**Entity extracted with the wrong name (e.g. "The ColPali" instead of "ColPali")** — the regex fallback strips leading articles but isn't perfect. Install GLiNER properly for high-quality extraction.
