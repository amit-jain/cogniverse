# Coding Agent CLI

Interactive coding agent accessible from the terminal. Plan, generate, and execute code changes against the cogniverse runtime with real-time streaming and multi-turn conversation.

## Commands

### `cogniverse code`

Interactive REPL that streams the coding agent's plan → generate → execute → evaluate loop.

```bash
cogniverse code
```

| Option | Default | Description |
|---|---|---|
| `--tenant` | `$COGNIVERSE_TENANT_ID` or `default` | Tenant identifier |
| `-l`, `--language` | `python` | Primary programming language |
| `-n`, `--iterations` | `5` | Max plan-code-execute iterations per task |
| `-c`, `--codebase` | _(none)_ | Indexed codebase path for context search |

#### REPL commands

| Command | Action |
|---|---|
| _free text_ | Send a coding task to the agent |
| `/apply` | Write the last generated code changes to local files |
| `/diff` | Show a diff between proposed changes and local files |
| `/plan` | Re-display the last plan |
| `/language <lang>` | Change language mid-session |
| `/codebase <path>` | Set codebase path for context search |
| `/iterations <n>` | Set max iterations |
| `/clear` | Clear conversation history |
| `/help` | Show all commands |
| `/exit` or Ctrl+D | Exit the REPL |

#### Example session

```
$ cogniverse code
Cogniverse Coding Agent (tenant: default, lang: python)
Type a coding task, or /help for commands. Ctrl+D to exit.

>>> write a retry decorator with exponential backoff
  >> Searching code context...
  >> Planning implementation...

## Plan
1. Create utils/retry.py with a retry decorator
2. Support max_retries, base_delay, max_delay parameters
3. Use random jitter to avoid thundering herd

  >> Generating code (iteration 1/5)...
  >> Executing in sandbox...
  Iteration 1: passed

## Summary
Created retry decorator with exponential backoff and random jitter.
Files: utils/retry.py

>>> /apply
  utils/retry.py (new)
Applied 1 file(s)

>>> make max_retries configurable via an env var
  >> Searching code context...
  >> Generating code (iteration 1/5)...
...

>>> /exit
```

### `cogniverse index`

Index local files into Vespa so the coding agent can find relevant context when generating code.

```bash
cogniverse index ./src --type code
```

| Option | Default | Description |
|---|---|---|
| `<path>` | _required_ | Directory to index |
| `--type` | `code` | Content type: `code` or `docs` |
| `--tenant` | `$COGNIVERSE_TENANT_ID` or `default` | Tenant identifier |
| `--profile` | _(auto from type)_ | Override Vespa profile |

The `docs` type is a catch-all for any non-code file — it auto-routes per extension to the right Vespa profile:

| Extension | Content profile |
|---|---|
| `.md` `.txt` `.rst` `.html` `.pdf` | `document_text_semantic` |
| `.mp4` `.mov` `.mkv` `.avi` `.webm` | `video_colpali_smol500_mv_frame` |
| `.jpg` `.jpeg` `.png` `.webp` `.gif` | `image_colpali_mv` |
| `.wav` `.mp3` `.m4a` `.flac` | `audio_clap_semantic` |

Code files always go to `code_lateon_mv` (tree-sitter AST chunking, LateOn-Code multi-vector embeddings).

**Knowledge graph extraction** — in addition to content indexing, `cogniverse index` extracts a knowledge graph of entities and relationships from code and text files and writes it to a separate schema. Query it with `cogniverse graph`. See [Knowledge Graph](knowledge-graph.md) for full details.

The indexer walks the directory, respects `.gitignore`, skips `node_modules` / `.venv` / `__pycache__`, and uploads via the runtime's `/ingestion/upload` endpoint. Re-indexing the same files is idempotent.

### `cogniverse sandbox status`

Show the state of the OpenShell gateway that runs sandboxed code execution.

```bash
cogniverse sandbox status
```

Prints the active gateway name, its config directory, whether it's running, and whether its certs are synced into the cluster.

### `cogniverse sandbox sync`

Re-sync the OpenShell gateway's mTLS certs into the cluster after cert rotation (host mode only).

```bash
cogniverse sandbox sync
```

Reads the current certs from `~/.config/openshell/gateways/<name>/` and updates the `openshell-mtls` Secret and `openshell-metadata` / `openshell-active` ConfigMaps. Restart the runtime pod afterwards so the Python client picks up the new certs.

## Architecture

The CLI is a thin HTTP client. All agent logic runs inside the cogniverse runtime.

```
cogniverse code (REPL)
     │
     │ POST /a2a/  (message/stream)       ┌───────────────┐
     │──────────────────────────────────> │  OpenShell    │
     │   SSE: status, partial, final      │  Gateway      │
     │ <──────────────────────────────────│               │
     │                                    │  Sandbox pods │
     │                                    └───────────────┘
     ▼
Cogniverse Runtime (k3d / prod k8s)
  ├── AgentDispatcher.create_streaming_agent("coding")
  ├── CodingAgent (DSPy planning, code generation, evaluation)
  └── SandboxManager → OpenShell gRPC (mTLS)
```

- **REPL loop** sends each turn as a JSON-RPC `message/stream` request to `/a2a/` with `conversation_history` so the agent sees prior plans and code.
- **Streaming** is done via Server-Sent Events. Each `status` event updates the current phase label (`search`, `plan`, `generate`, `execute`, `evaluate`). The final event carries the `CodingOutput` with plan, code changes, execution results, and summary.
- **Sandbox execution** happens inside an OpenShell sandbox pod. Code is written to `/tmp/coding_workspace/solution.<ext>`, run with the generated test command, and the stdout/stderr/exit code come back to the agent for evaluation. If the exit code is non-zero, the agent iterates with the error as context.

## Sandbox Deployment Modes

The coding agent requires an OpenShell sandbox gateway for code execution. The gateway comes in two flavors depending on where cogniverse runs.

### Host Mode (local dev with k3d)

`cogniverse up` auto-installs the OpenShell CLI if missing, starts the gateway on the host, and syncs its mTLS certs into the cluster as k8s Secrets.

```bash
cogniverse up
```

What happens under the hood:

1. The `openshell` CLI is downloaded to `~/.local/bin` if not already installed.
2. `openshell gateway start` launches the `ghcr.io/nvidia/openshell/cluster:0.0.13` Docker container on the host. This image bundles a mini-k3s cluster that in turn runs the gateway as a pod inside itself. Port `19091` on the host is forwarded into the container.
3. The gateway generates mTLS certs at `~/.config/openshell/gateways/<name>/mtls/`.
4. `cogniverse_cli.sandbox.sync_gateway_certs_to_cluster()` reads those certs and creates k8s resources in the `cogniverse` namespace:
   - Secret `openshell-mtls` — contains `ca.crt`, `tls.crt`, `tls.key`
   - ConfigMap `openshell-metadata` — the gateway's `metadata.json` with the endpoint rewritten to `https://host.docker.internal:19091` so the pod can reach the host
   - ConfigMap `openshell-active` — points to `cogniverse` as the active gateway name
5. The runtime pod mounts all three at `/home/cogniverse/.config/openshell/gateways/cogniverse/`.
6. When the coding agent needs to execute code, it calls `SandboxClient.from_active_cluster()` which reads the mounted metadata and certs, opens a gRPC connection to `host.docker.internal:19091` with mTLS, and creates a sandbox.

Host mode is a single-machine setup — one host, one gateway, one developer. The sandboxes run inside the inner k3s cluster that the openshell image bundles.

**Cert rotation:** OpenShell regenerates certs if the gateway is destroyed and restarted. Run `cogniverse sandbox sync` to copy the new certs into the cluster, then restart the runtime pod.

### Sandbox session pool (D.5)

`SandboxManager.exec_in_sandbox` now reuses one OpenShell session per
`agent_type` across calls. The pool is enabled by default and prunes
sessions that have been idle longer than `max_idle_seconds`. Behaviour:

- **First call for an agent**: create + wait_ready (lifecycle spans fire), exec, session retained.
- **Subsequent calls for the same agent**: exec on the cached session (no create / no wait_ready).
- **Different agent**: separate session.
- **Pool full**: oldest idle session evicted before creating a new one.
- **Idle eviction**: `pool.evict_idle()` (or runtime shutdown's `close_all`) destroys idle sessions.
- **Callback exception**: the session is dropped from the pool; next checkout creates a fresh one.

| Env var | Default | Effect |
|---|---|---|
| `COGNIVERSE_SANDBOX_POOL_ENABLED` | `1` | Set to `false` to fall back to per-call create+destroy. |
| `COGNIVERSE_SANDBOX_POOL_SIZE` | `8` | Maximum pooled sessions (one per agent_type). |
| `COGNIVERSE_SANDBOX_POOL_IDLE_S` | `60` | Seconds an entry can sit idle before eviction. |

The pool emits the same D.4 telemetry spans (`sandbox.create_session`,
`sandbox.wait_ready`, `sandbox.delete`) on its lifecycle events, so the
trace shape stays observable — they just fire less often when reuse hits.

### Sandbox lifecycle telemetry (D.4)

Every call to `SandboxManager.exec_in_sandbox` emits a parent
`sandbox.exec_in_sandbox` span plus child spans for each lifecycle phase
(`sandbox.create_session`, `sandbox.wait_ready`, `sandbox.exec`,
`sandbox.delete`). The `sandbox.exec` span carries:

| Attribute | Meaning |
|---|---|
| `openshell.agent_type` | Agent name (e.g. `coding_agent`) |
| `openshell.command_first` | First token of the command (audit aid) |
| `openshell.timeout_seconds` | The exec timeout |
| `openshell.exit_code` | Subprocess exit code |
| `openshell.wall_ms` | Wall-clock duration of the exec |
| `openshell.oom` | True when exit_code ∈ {137, 139} or stderr matches OOM markers |
| `openshell.policy_denied` | True when stderr matches `permission denied` / `syscall denied` / `blocked by policy` |
| `openshell.error` | Exception class name (parent span only, on hard failure) |

These spans become children of whichever agent span is active when
`exec_in_sandbox` is called, so Phoenix shows the sandbox call inline
with the rest of the agent's processing trace.

### Application-layer egress enforcement (D.1)

In addition to kernel-layer NetworkPolicy enforcement (in-cluster mode), the
runtime enforces each agent's `network_policies.egress` allow-list at the
httpx transport layer. Agents whose dispatcher path stamps a policy obtain
their httpx client via `SandboxManager.make_http_client(agent_type)` — the
returned client wraps every outbound request in a `PolicyEnforcingTransport`
that raises `EgressDeniedError` for non-allow-listed `(host, port)`.

This is defence-in-depth: kernel policy stops out-of-process bypass; the
transport surfaces the violation in application logs with the offending
endpoint and the operator-actionable allow-list.

| Env var | Default | Effect |
|---|---|---|
| `COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT` | unset | Set to `disabled` to bypass the transport check (useful while iterating on policies in dev). |

Today wired:

| Agent | Status |
|---|---|
| `coding_agent` | Code execution sandboxed via the existing OpenShell SDK exec path. |
| `orchestrator_agent` | A2A sub-agent calls flow through `make_http_client("orchestrator_agent")`. |
| `search_agent` | Policy file in place; outbound httpx client to be migrated through the dispatcher's `make_http_client` per the same pattern as orchestrator. |
| `summarizer_agent` | Policy file in place; LLM endpoint allow-listed via Ollama (port 11434). |
| `routing_agent` | Policy file in place; same pattern as summarizer. |

The remaining agents (search/summarizer/routing) keep the existing httpx
clients today; `make_http_client(<agent>)` is the migration path. Adding
the wrapper to a new agent is a one-line change at the agent's
construction site in `agent_dispatcher.py` (mirror the `OrchestratorAgent`
example).

### Gateway health probe

When sandboxing is not disabled, the runtime starts a background probe that
calls `SandboxClient.health()` every 30 s (configurable via
`COGNIVERSE_SANDBOX_PROBE_INTERVAL`). Each probe emits an OpenTelemetry span
named `openshell.gateway_health` with attributes:

| Attribute | Meaning |
|---|---|
| `openshell.gateway_available` | 1 when the gateway responded; 0 otherwise |
| `openshell.gateway_latency_ms` | Round-trip probe latency |
| `openshell.gateway_error` | Exception class name or `no_client` (only set on failure) |

The Phoenix dashboard reads these spans for the gateway-status tile. The probe
runs as part of the FastAPI lifespan; `stop()` is awaited at shutdown so the
runtime can exit cleanly.

### Sandbox boot policy

The runtime resolves a single `sandbox.policy` knob with three values:

| Value | Behaviour at boot when gateway is unreachable |
|---|---|
| `required` | **Refuse to start** with `SandboxGatewayUnavailableError`. Use for production tenants where egress isolation is a compliance requirement. |
| `optional` | Log a warning and continue without sandbox enforcement. Default; suitable for dev and staging. |
| `disabled` | Do not even attempt to connect; `SandboxManager.available` is permanently False. Use when sandboxing is intentionally off. |

Resolution order (first non-empty wins):

1. `COGNIVERSE_SANDBOX_POLICY` env var — `required` / `optional` / `disabled`.
2. `config["sandbox"]["policy"]` from `configs/config.json` (or per-tenant config).
3. Legacy `COGNIVERSE_SANDBOX_ENABLED` + presence of `OPENSHELL_GATEWAY_ENDPOINT` → maps to `optional` (true) or `disabled` (false). Kept for backwards compatibility; new code should use `policy` directly.

Default when none are set: `optional`.

### In-Cluster Mode (production)

Production clusters (EKS, GKE, AKS, bare-metal k8s) don't have a "host" to run things on. For these, the gateway is deployed as a k8s StatefulSet inside the same cluster as cogniverse.

```bash
helm install cogniverse charts/cogniverse \
  --set runtime.sandbox.enabled=true \
  --set runtime.sandbox.inCluster.enabled=true \
  --set openshell.server.sshHandshakeSecret=$(openssl rand -hex 32)
```

What the Helm chart deploys:

| Resource | Purpose |
|---|---|
| `Job: cogniverse-openshell-cert-gen` | Pre-install hook that generates CA, server, and client mTLS certs using openssl. Stores them as four k8s Secrets: `openshell-server-tls`, `openshell-server-client-ca`, `openshell-client-tls`, `openshell-client-ca`. Runs once per Helm install/upgrade. |
| `StatefulSet: openshell` | Runs `ghcr.io/nvidia/openshell/gateway:0.0.13` as a non-root pod. Uses the in-cluster k8s API to create sandbox pods in the cogniverse namespace. |
| `Service: openshell` | `ClusterIP` on port 8080. In-cluster DNS: `openshell.cogniverse.svc.cluster.local:8080`. |
| `ServiceAccount` + `Role` + `RoleBinding` | Permissions for the gateway to create/delete sandbox pods. |
| `NetworkPolicy` | Restricts sandbox SSH ingress to the gateway only. |

The runtime pod's env var `OPENSHELL_GATEWAY_ENDPOINT` is auto-set to `openshell.<namespace>.svc.cluster.local:8080` and it mounts `openshell-client-tls` + `openshell-client-ca` as volumes. There is no host dependency — nothing runs outside the cluster.

### Differences Between the Two Modes

| | Host mode | In-cluster mode |
|---|---|---|
| Docker image | `openshell/cluster` (k3s-in-Docker wrapper) | `openshell/gateway` (just the gateway) |
| Bootstrap | `openshell gateway start` CLI | Helm subchart + pre-install Job |
| Where it runs | Host as a plain Docker container | K8s StatefulSet in cogniverse namespace |
| Sandbox isolation | Inner k3s cluster inside the container | Sandbox pods alongside cogniverse |
| Certs generated by | `openshell` CLI on first start | openssl pre-install Job in-cluster |
| Runtime endpoint | `host.docker.internal:19091` | `openshell.cogniverse.svc.cluster.local:8080` |
| Portable? | Local dev only | Any k8s cluster |

**Why two modes?** The gateway needs a k8s API to schedule sandboxes into. On a dev machine there's no k8s available to the host, so NVIDIA ships `openshell/cluster` which bundles k3s and runs it inside a Docker container. In production that's redundant — you already have a real k8s cluster, so you run the gateway image directly and it uses the cluster's existing control plane.

## Requirements

- `cogniverse up` must have already provisioned the stack (host mode) or the production Helm release must have `runtime.sandbox.enabled=true` (in-cluster mode).
- The `openshell==0.0.13` Python package is pinned in `cogniverse-runtime` and installed automatically — no manual setup.
- Host mode requires Docker (for the gateway container) and downloads the `openshell` CLI binary to `~/.local/bin` on first `cogniverse up`.

## Troubleshooting

**`Cannot connect to runtime. Run 'cogniverse up' first.'`** — The REPL can't reach `http://localhost:28000`. Verify the runtime is healthy: `cogniverse status`.

**Coding agent returns 500 with a SandboxManager error** — The gateway isn't reachable from the runtime pod. Check `cogniverse sandbox status`. In host mode, ensure `openshell gateway info` reports the gateway is running. In in-cluster mode, check the openshell StatefulSet is ready: `kubectl get statefulset openshell -n cogniverse`.

**Indexed code isn't showing up in search** — Code search uses the `code_lateon_mv` profile. This requires the LateOn-Code query encoder to be registered, which isn't enabled by default. The coding agent falls back to empty context and still generates code without it — search is optional.

**REPL commands not recognized** — The REPL only recognizes commands starting with `/`. Free text is always sent to the agent. Use `/help` to list commands.
