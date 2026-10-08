# Web Client

**Location:** `clients/web/`
**Image:** `cogniverse/web`

The web client is the Cogniverse UI: a Node server (Hono, CopilotKit runtime)
that serves a React client built with Vite. It talks to the Cogniverse runtime
only; the browser talks only to the web server.

---

## Features

- **Agent chat** — every agent in the runtime's registry appears in the
  sidebar. A run streams over the runtime's AG-UI surface (`POST /ag-ui/{agent}`)
  and renders its results beside the conversation: search hits as result cards
  with relevance rating, the coding agent's files and run output, and an
  orchestration's per-agent hits. Conversations are threads the runtime stores,
  restored on reload.
- **Operations views** — tenants, backend profiles, configuration (versions,
  export and import), ingestion with live progress, optimization runs, memory,
  approvals, the annotation queue, workflow reviews, profile metrics, RLM A/B,
  analytics, evaluation, the embedding atlas and routing evaluation.

Each view, and the runtime routes it calls, is listed in
[`clients/web/README.md`](../../clients/web/README.md#operations-views).

## Server Routes

| Route | Purpose |
|---|---|
| `/healthz` | Liveness and readiness: `{"status":"ok"}` while the process serves, whatever the runtime's state |
| `/ui-api/agents` | The runtime's agent registry |
| `/ui-api/copilotkit/*` | CopilotKit runtime; runs go to the runtime's `/ag-ui/{agent}` |
| `/ui-api/runtime/*` | Forwards an allowlist of the runtime's admin, ingestion and event routes (`src/server/proxy.ts`), streaming server-sent events |
| everything else | The built client (`index.html` for client-side routes) |

The server's own routes live under `/ui-api`, so they never collide with the
runtime's `/api` ingress prefix.

## Configuration

| Variable | Meaning |
|---|---|
| `COGNIVERSE_RUNTIME_URL` | Runtime base URL, including any ingress prefix |
| `COGNIVERSE_API_KEY` | Harness key the server sends the runtime on every call; it decides the tenant |
| `PORT` | Port the server listens on (default `4000`) |
| `HOST` | Interface the server binds (default `127.0.0.1`; the image sets `0.0.0.0`) |

The runtime accepts the key when it is a minted harness key or one of the
static keys in `config.json` `harness.api_keys`; the shipped config maps
`$COGNIVERSE_HARNESS_API_KEY` to the `default` tenant.

## Deployment

### Image

`clients/web/Dockerfile` builds with `clients/web` as its context: `npm ci`,
`npm run build`, then a Node 22 image running `node dist/server/index.js` as
user `node` on port 4000. `cogniverse up` and the e2e deploy build it with the
other first-party images, tagged from the last commit that touched
`clients/web`, and import it into k3d.

```bash
docker build -f clients/web/Dockerfile -t cogniverse/web:dev clients/web
```

### Helm values

The chart's `web` block (`charts/cogniverse/values.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `web.enabled` | `true` | Render the web Deployment, Service and Secret |
| `web.image` | `cogniverse/web:<appVersion>` | `values.k3s.yaml` uses `<appVersion>-dev`, `pullPolicy: Never` |
| `web.runtimeUrl` | `""` | Runtime base URL; empty uses the release's runtime Service |
| `web.harnessKey` | `cogniverse-web-dev` | Key stored in the `<release>-web` Secret; `values.prod.yaml` empties it, so a prod install fails until it or `existingSecret` is set |
| `web.existingSecret` | `""` | Secret holding the key under `harness-api-key`, used instead of `harnessKey` |
| `web.env` | `{}` | Extra environment (`name: value`) |
| `web.envFrom` | `[]` | Secret or ConfigMap sources, e.g. for user authentication |
| `web.service` | `ClusterIP`, port `4000`, nodePort `28400` | `values.k3s.yaml` sets `type: NodePort` |
| `web.livenessProbe`, `web.readinessProbe` | `GET /healthz` | |
| `web.resources`, `nodeSelector`, `tolerations`, `affinity` | | |

The chart gives the runtime the same key as `COGNIVERSE_HARNESS_API_KEY`, so
the runtime accepts the web server's calls for the `default` tenant.

```bash
kubectl create secret generic cogniverse-web-key -n cogniverse \
  --from-literal=harness-api-key="$(openssl rand -hex 32)"
helm upgrade --install cogniverse ./charts/cogniverse -f charts/cogniverse/values.prod.yaml \
  --set web.existingSecret=cogniverse-web-key
```

### Ingress

Every values file routes `/api` to the runtime and `/` to the web client on
port 4000. The nginx ingress turns off proxy buffering so server-sent events
(chat replies, ingest progress) reach the browser as they are written.

### Local k3d

`cogniverse up` deploys the web client at http://localhost:28400.
`cogniverse status` probes `http://localhost:28400/healthz` and
`cogniverse logs web` tails the server. A k3d cluster created before port
28400 was published needs it added:

```bash
k3d cluster edit cogniverse --port-add 28400:28400@loadbalancer
```

The Streamlit [dashboard](dashboard.md) is deployed only when
`dashboard.enabled: true` (default `false`).

## Local Development

Node.js 22 is required.

```bash
cd clients/web
cp .env.example .env   # set COGNIVERSE_RUNTIME_URL and COGNIVERSE_API_KEY
npm ci
npm run dev                  # server on PORT, Vite on :5173 proxying /ui-api to it
npm run build && npm start   # one server serving the built client
```

## Tests

`npm run typecheck`, `npm test` and `npm run build` in `clients/web`; the
browser-driven suites are `tests/runtime/integration/test_web_*.py`. See
[`clients/web/README.md`](../../clients/web/README.md#tests).
