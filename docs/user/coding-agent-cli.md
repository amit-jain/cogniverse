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
| `--type` | `code` | Content type: `code`, `docs`, `video` |
| `--tenant` | `$COGNIVERSE_TENANT_ID` or `default` | Tenant identifier |
| `--profile` | _(auto from type)_ | Override Vespa profile |

**Type → profile mapping:**

| Type | Profile | Chunking | Embeddings |
|---|---|---|---|
| `code` | `code_lateon_mv` | tree-sitter AST by function/class | LateOn-Code multi-vector |
| `docs` | `document_text_semantic` | paragraph / markdown section | nomic-embed-text |
| `video` | `video_colpali_smol500_mv_frame` | keyframe extraction | ColPali |

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
