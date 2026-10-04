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

Vitest covers the result-card parsing, error and route parsing, the server's
configuration and agent listing, and the runtime proxy against local HTTP
sockets.
`tests/runtime/integration/test_web_client_ag_ui.py` installs this lockfile,
runs the server from source against the runtime's routers and drives it with
the published `@ag-ui/client`. `tests/runtime/integration/test_web_ops_*.py`
build the client, serve it, and drive each operations view in Chromium against
the runtime's routers over real Vespa.
