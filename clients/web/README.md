# cogniverse-web

Browser UI for the Cogniverse runtime, built on CopilotKit 1.77. Every agent in
the runtime's registry (`GET /agents/`) appears as a Dot in the sidebar; picking
one opens a CopilotKit chat with it. Runs stream over the runtime's AG-UI
surface (`POST /ag-ui/{agent}`): the reply streams token by token, status
phases show above the chat, and search hits in the run's final state render as
result cards.

The browser talks only to this package's Node server. The server hosts the
CopilotKit runtime at `/api/copilotkit` and holds one harness key, which it
sends to the Cogniverse runtime on every run; the key decides the tenant. The
operations views reach the runtime's admin, ingestion and event routes through
`/api/runtime/*`, which forwards an allowlist of those routes (`src/server/proxy.ts`)
and streams their server-sent events. There is no user login, so anyone who can
reach the server can use those routes.

## Operations views

The sidebar's Operations section manages the runtime. Each view calls the
runtime's existing routes through `/api/runtime/*`.

| View | What it does | Runtime routes |
|---|---|---|
| Tenants | List, create and delete organizations and tenants; set a tenant's router tier. Deletes need the name typed to confirm. | `/admin/organizations`, `/admin/tenants`, `/admin/router-tiers` |
| Backend profiles | For a chosen tenant: list the profiles created for it, create one (JSON fields for pipeline, strategies, schema config and model-specific parameters), edit its description, pipeline, strategies and model-specific parameters, deploy its schema, and delete it with or without its schema. Shipped profiles are not listed. | `/admin/profiles` |
| Ingestion | Upload a file to a tenant (optionally naming a profile, or forcing a re-ingest of identical bytes) and follow each ingest live through queued, running, and complete or failed, with its result or error. Follow an existing ingest by its ID. | `/ingestion/upload`, `/ingestion/{id}/events`, `/ingestion/{id}/status` |
| Optimization runs | For a chosen tenant: start a run in any mode the runtime accepts, list its runs with mode, trigger, phase and times (polled while any run is unsettled), open a run to see its steps and Argo message, cancel an unsettled run, and retry the failed steps of a failed one. | `/admin/tenant/optimize-modes`, `/admin/tenant/{tenant}/optimize`, `/admin/tenant/{tenant}/optimize/runs`, `/admin/tenant/{tenant}/optimize/runs/{name}`, `.../cancel`, `.../retry` |
| Memory | For a chosen tenant and namespace (the user's memories, an agent's, or a system one): count its live and archived memories, list or semantically search them, add one with a category and JSON metadata, delete one, and clear the namespace after typing its name. System namespaces are read-only. A store outage shows as an error, never as an empty namespace. | `/admin/tenant/{tenant}/memories`, `/admin/tenant/{tenant}/memories/stats`, `/admin/tenant/{tenant}/memories/{id}`, `/agents/` |
| Approvals | A tenant's generated examples awaiting review, each with its schema, confidence, reasoning and data. As a named reviewer, approve an item into the training dataset, or reject it with feedback and edits to its correctable fields, which regenerates it (or, for a workflow record, merges the corrections) for another review. A decision another reviewer already made is refused rather than overwritten. | `/admin/tenant/{tenant}/approvals`, `/admin/tenant/{tenant}/approvals/{batch}/{item}` |
| Annotation queue | The routing decisions queued for human review across tenants: counts by status, then the pending (by priority), assigned (by due time) and expired requests, each with its query, chosen agent, confidence, outcome and reason. As a named reviewer, assign a pending request to yourself and label a pending or assigned one with a reason; the label is stored on the decision's span for the tenant before the request completes. A request someone else already labelled, and a queue or telemetry outage, show the runtime's reason. | `/agents/annotations/queue`, `/agents/annotations/queue/{span}/assign`, `/agents/annotations/queue/{span}/complete`, `/agents/annotations/labels` |
| Workflow reviews | For a chosen tenant and window, the orchestration workflows the orchestrator recorded, newest first, with their query, pattern, agents, time, outcome and latest review. As a named reviewer, rate a workflow's quality and say whether its pattern, agents and execution order were right, with suggestions and notes; the review is stored on the workflow's span. A telemetry outage shows as an error, never as an empty list. | `/admin/tenant/{tenant}/orchestration-workflows`, `/admin/tenant/{tenant}/orchestration-workflows/{span}/annotation` |

## Setup

Use Node.js 22. Copy `.env.example` to `.env` and set:

| Variable | Meaning |
|---|---|
| `COGNIVERSE_RUNTIME_URL` | Runtime base URL, including any ingress prefix |
| `COGNIVERSE_API_KEY` | Harness key the server sends to the runtime |
| `PORT` | Port the server listens on (default `4000`) |
| `HOST` | Interface the server binds (default `127.0.0.1`) |

```bash
cd clients/web
npm ci
npm run dev        # server on PORT, Vite on :5173 proxying /api to it
npm run build && npm start   # one server serving the built client
```

The server turns off CopilotKit's usage telemetry unless
`COPILOTKIT_TELEMETRY_DISABLED=false` is set.

## Tests

```bash
npm run typecheck
npm test
```

Vitest covers the result-card parsing, error, route, JSON-field and event-stream parsing, the server's
configuration and agent listing, and the runtime proxy against local HTTP
sockets.
`tests/runtime/integration/test_web_client_ag_ui.py` installs this lockfile,
runs the server from source against the runtime's routers and drives it with
the published `@ag-ui/client`. `tests/runtime/integration/test_web_ops_*.py`
build the client, serve it, and drive each operations view in Chromium against
the runtime's routers over real Vespa.
