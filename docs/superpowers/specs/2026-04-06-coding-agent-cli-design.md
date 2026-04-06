# Coding Agent CLI Design Spec

## Goal

Add `cogniverse code` (interactive coding REPL) and `cogniverse index` (codebase indexing) commands to the existing CLI. Developers interact with the coding agent from the terminal with real-time streaming output, multi-turn conversation, and the ability to apply generated code to local files.

## Constraints

- Requires `cogniverse up` — thin HTTP client against the running k3d runtime
- No heavy dependencies in the CLI package (no DSPy, no Vespa client, no ML models)
- Uses A2A SSE streaming for real-time progress (not polling)
- CLI framework: click + rich + httpx (already CLI dependencies)

## Commands

### `cogniverse index <path> --type <type>`

Index local files into Vespa for context search.

| Option | Default | Description |
|--------|---------|-------------|
| `<path>` | required | Directory to index |
| `--type` | `code` | Content type: `code`, `docs`, `video` (future) |
| `--tenant` | from config | Tenant ID |
| `--profile` | auto from type | Vespa profile (`code` → `code_lateon_mv`) |

**Type → profile mapping:**

| Type | Profile | Chunking | Embeddings |
|------|---------|----------|------------|
| `code` | `code_lateon_mv` | tree-sitter AST | LateOn-Code multi-vector |
| `docs` | `document_text_semantic` | markdown/PDF | nomic-embed-text |
| `video` | `video_colpali_smol500_mv_frame` | keyframe extraction | ColPali |

Only `code` is implemented in this spec. Others are future placeholders in the `--type` enum.

**Flow:**
1. Walk directory, respect `.gitignore`
2. Upload files via `POST /ingestion/upload` with the appropriate profile
3. Show progress: files found → uploaded → indexed
4. Print summary: N files indexed, M chunks created

### `cogniverse code`

Interactive coding REPL with streaming agent output.

| Option | Default | Description |
|--------|---------|-------------|
| `--tenant` | from config | Tenant ID |
| `--language` | `python` | Primary programming language |
| `--iterations` | `5` | Max plan-code-execute iterations |
| `--codebase` | none | Path for code context search (must be indexed first) |

**REPL commands:**

| Command | Action |
|---------|--------|
| Free text | Send coding task to agent |
| `/apply` | Write last generated code changes to local files |
| `/diff` | Show diff of proposed changes vs local files |
| `/plan` | Re-display the last plan |
| `/language <lang>` | Change language |
| `/codebase <path>` | Set codebase path |
| `/iterations <n>` | Set max iterations |
| `/clear` | Clear conversation history |
| `/exit` or Ctrl+D | Exit |

**Streaming display phases:**

The coding agent emits progress events: `search`, `plan`, `generate`, `execute`, `evaluate`, `done`, `summarize`. The REPL renders each phase with a spinner and replaces it with the result when complete.

```
>>> add retry decorator with exponential backoff
⏳ Searching code context...
⏳ Planning...

## Plan
1. Create utils/retry.py with retry decorator
2. Support max_retries, base_delay, max_delay params

⏳ Generating code (iteration 1/5)...

utils/retry.py (new):
```python
def retry(max_retries=3, base_delay=1.0):
    ...
```

⏳ Executing in sandbox...
✅ Tests passed (iteration 1)

## Summary
Created retry decorator with exponential backoff.
Files: utils/retry.py

>>> /apply
Applied 1 file: utils/retry.py ✅

>>> make max_retries configurable via env var
⏳ Searching code context...
```

**Conversation context:** Each turn includes the previous turns as `conversation_history` in the A2A request. The coding agent sees prior plans, code, and feedback for coherent multi-turn sessions.

## Architecture

```
cogniverse CLI
├── main.py          — existing, add `code` and `index` commands
├── code.py          — REPL loop, /command dispatch, conversation state
├── index.py         — file walking, upload to ingestion API
└── streaming.py     — SSE consumer, rich rendering of agent events
      │
      ▼
  Runtime (localhost:28000)
      │
      ├── POST /a2a (tasks/sendSubscribe) → SSE stream
      │     └── AgentDispatcher.create_streaming_agent("coding")
      │           └── CodingAgent.process(input, stream=True)
      │
      └── POST /ingestion/upload (for indexing)
```

### CLI files (new)

**`code.py`** — REPL entry point
- `click.command` decorated function
- Main loop: `prompt_toolkit` or `rich.prompt` for input
- Parses `/commands` vs free text
- Builds A2A `tasks/sendSubscribe` payload with conversation history
- Calls `streaming.stream_coding_response()` to consume SSE
- Stores last `CodingOutput` for `/apply`, `/diff`, `/plan`

**`index.py`** — Codebase indexing
- Walks directory with `.gitignore` filtering
- Groups files by language/extension
- Uploads via `POST /ingestion/upload` with `profile=code_lateon_mv`
- Rich progress bar for file processing

**`streaming.py`** — SSE consumer and renderer
- `httpx` SSE client consuming A2A event stream
- Maps event types to rich rendering:
  - `status` events → spinner with phase name
  - `partial` events → incremental text output (token streaming)
  - `final` event → structured result display
- Returns parsed `CodingOutput` when stream completes

### Production changes (small)

**`a2a_executor.py`** — Add `"coding"` to `_STREAMING_CAPABILITIES` set.

**`agent_dispatcher.py`** — Add `"coding"` case in `create_streaming_agent()`. Follows the exact pattern of existing capabilities (search, summary, report). Creates `CodingAgent` with deps, calls `agent.process(input, stream=True)`, yields events.

## Data Flow

### Index flow
```
cogniverse index ./myproject --type code
  → walk directory (respect .gitignore)
  → POST /ingestion/upload (file, profile=code_lateon_mv, tenant_id)
  → Vespa indexes with LateOn-Code embeddings + tree-sitter chunks
  → CLI shows progress + summary
```

### Code flow (per turn)
```
User types: "add retry decorator"
  → CLI builds A2A JSON-RPC request with conversation_history
  → POST /a2a with tasks/sendSubscribe
  → Runtime dispatches to CodingAgent
  → Agent emits SSE events: search → plan → generate → execute → evaluate → done
  → CLI renders each event in real-time
  → Final CodingOutput stored for /apply, /diff
```

### Apply flow
```
User types: /apply
  → CLI reads last CodingOutput.code_changes
  → For each {file_path, content, change_type}:
    → If change_type == "new": write file
    → If change_type == "modify": overwrite file
    → If change_type == "delete": remove file
  → Show applied files with ✅
```

## Error Handling

- **Runtime not running:** Check health on REPL start, show "Run `cogniverse up` first"
- **Streaming disconnected:** Retry once, then show error with last received event
- **Sandbox unavailable:** The coding agent raises `RuntimeError` — display the error, suggest checking sandbox config
- **Index path not found:** Validate path before starting
- **Apply conflict:** If local file changed since generation, show diff and confirm

## Testing

- **Unit tests** (mocked httpx): REPL command parsing, `/apply` file writing, SSE event parsing, conversation history building
- **Integration tests** (real k3d): Index a test directory → verify searchable, send coding task → verify streaming events received, `/apply` → verify files written
- **E2E test**: `cogniverse index tests/fixtures/sample_code --type code` → `cogniverse code` → send task → verify response
