# cogniverse-web

Browser UI for the Cogniverse runtime, built on CopilotKit 1.77. Every agent in
the runtime's registry (`GET /agents/`) appears as a Dot in the sidebar; picking
one opens a CopilotKit chat with it. Runs stream over the runtime's AG-UI
surface (`POST /ag-ui/{agent}`): the reply streams token by token and status
phases show above the chat. A run that fails says why in the conversation, and
one that is stopped says it was cancelled.

Each conversation is a thread named in the address (`#/agents/{agent}/{thread}`).
The runtime saves every run's turn under its thread, and opening a thread
restores its turns from the runtime (`GET /ag-ui/threads/{thread}`), so a
conversation survives a reload, a server restart and switching agents: the
browser remembers each agent's last thread, and "New conversation" starts
another. A thread the runtime cannot read shows the reason instead of an empty
conversation.

The run's final state renders beside the chat: search hits as result cards
(video segments with their description and time, documents with their
preview, images and audio with their text, each with its ranking score), the
coding agent's files and the output of running them, and, for an orchestration,
each planned agent's hits under its name. When a search recorded a telemetry
span, each of its cards can be rated Highly Relevant, Somewhat Relevant or Not
Relevant; the rating is stored on the search's span
(`POST /ag-ui/results/relevance`), where the embedding triplet miner reads it,
and a rating that was not stored shows its reason on the card.

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
| Backend profiles | For a chosen tenant: list the profiles created for it, create one, either blank or started from a shipped profile whose every key fills the form (model loader, process type, and JSON fields for pipeline, strategies, schema config, model-specific parameters and extra config such as inference services), edit its description, pipeline, strategies and model-specific parameters, deploy its schema, and delete it with or without its schema. Shipped profiles are not listed. | `/admin/profiles`, `/admin/profile-templates` |
| Configuration | The system config and, for a chosen tenant, its routing, telemetry and durable-execution configs and each agent's config, as forms generated from the runtime's schema for each section, so a new config field appears without client changes. A save applies to the version the form was loaded at: one made over another operator's save is refused with both versions named. Secrets are never shown; a blank secret keeps its value and one marked to clear is removed. Every stored config of the system and of the tenant is listed with its versions, any of which can be restored as the next version. A tenant's configs export to a JSON download (secrets included) and import from one, whole or not at all. The config store's backend and counts are shown. | `/admin/config/sections`, `/admin/config/entries`, `/admin/config/history`, `/admin/config/rollback`, `/admin/config/export`, `/admin/config/import`, `/admin/config/stats` |
| Ingestion | Upload a file to a tenant (optionally naming a profile, or forcing a re-ingest of identical bytes) and follow each ingest live through queued, running, and complete, failed or cancelled, with its result, error or cancellation reason. A completion that fed no documents shows as a failure. Follow an existing ingest by its ID. | `/ingestion/upload`, `/ingestion/{id}/events`, `/ingestion/{id}/status` |
| Optimization runs | For a chosen tenant: start a run in any mode the runtime accepts over a chosen lookback (for the synthetic mode, generating training data for the optimizers chosen, which then wait in Approvals), list its runs with mode, trigger, phase and times (polled while any run is unsettled), open a run to see its steps and Argo message, cancel an unsettled run, and retry the failed steps of a failed one. | `/admin/tenant/optimize-modes`, `/admin/tenant/{tenant}/optimize`, `/admin/tenant/{tenant}/optimize/runs`, `/admin/tenant/{tenant}/optimize/runs/{name}`, `.../cancel`, `.../retry` |
| Memory | For a chosen tenant and namespace (the user's memories, an agent's, or a system one): count its live and archived memories, list or semantically search them, add one with a category and JSON metadata, delete one, and clear the namespace after typing its name. System namespaces are read-only. A store outage shows as an error, never as an empty namespace. | `/admin/tenant/{tenant}/memories`, `/admin/tenant/{tenant}/memories/stats`, `/admin/tenant/{tenant}/memories/{id}`, `/agents/` |
| Approvals | A tenant's generated examples awaiting review, each with its schema, confidence, reasoning, the entity self-consistency check's agreement per sampled mention, and data. As a named reviewer, approve an item into the training dataset, or reject it with feedback and edits to its correctable fields, which regenerates it (or, for a workflow record, merges the corrections) for another review. A decision another reviewer already made is refused rather than overwritten. | `/admin/tenant/{tenant}/approvals`, `/admin/tenant/{tenant}/approvals/{batch}/{item}` |
| Annotation queue | The routing decisions queued for human review across tenants: counts by status, then the pending (by priority), assigned (by due time) and expired requests, each with its query, chosen agent, confidence, outcome and reason. As a named reviewer, assign a pending request to yourself and label a pending or assigned one with a reason; the label is stored on the decision's span for the tenant before the request completes. A request someone else already labelled, and a queue or telemetry outage, show the runtime's reason. | `/agents/annotations/queue`, `/agents/annotations/queue/{span}/assign`, `/agents/annotations/queue/{span}/complete`, `/agents/annotations/labels` |
| Workflow reviews | For a chosen tenant, window and cap (1–500 workflows), the orchestration workflows the orchestrator recorded, newest first and counted, with their query, pattern, agents, time, outcome and latest review. As a named reviewer, rate a workflow's quality and say whether its pattern was optimal (yes, no with a suggested pattern, or unsure, stored as not optimal), whether its agents and execution order were right, with suggestions, what went well and wrong, and notes; the review is stored on the workflow's span and shows in the list at once. A review that is not stored shows the runtime's reason with its error code, failure type and status. A telemetry outage shows as an error, never as an empty list. | `/admin/tenant/{tenant}/orchestration-workflows`, `/admin/tenant/{tenant}/orchestration-workflows/{span}/annotation` |
| Profile metrics | For a chosen tenant and window, its profile selections per modality: count, P50, P95 and P99 latency and success rate, with bars for selections and P95 latency per modality. A telemetry outage shows as an error, never as an empty window. | `/admin/tenant/{tenant}/telemetry/profile-selection` |
| RLM A/B | For a chosen tenant and window, its RLM A/B comparisons: average latency, token and judge-score change with RLM and how often RLM fell back, the same per queries dataset with a latency bar per dataset, and each comparison newest first. | `/admin/tenant/{tenant}/telemetry/rlm-ab` |
| Analytics | For a chosen tenant and window, its traces filtered by operation, profiles and strategies: counts, success rate and latency, then an overview (latency percentiles and traces per operation), latency and trace counts over time, latency histograms and spread grouped by operation, profile, strategy or outcome, a mean-latency heatmap over two chosen fields, the traces outside Tukey's outlier bounds, and a searchable, sortable list of every trace. Charts are Plotly, loaded the first time one is shown. Root causes runs over the filtered traces: their failed ones and, optionally, the successful ones slower than a chosen percentile, giving the hypotheses (with evidence, affected traces and a suggested action) and the recommendations with the components they affect. A telemetry outage shows as an error, never as no traces. | `/admin/tenant/{tenant}/telemetry/traces`, `/admin/tenant/{tenant}/telemetry/root-causes` |
| Evaluation | For a chosen tenant and window, its searches of its golden set's queries, scored against the golden set: MRR, nDCG@10, recall@1 and @5, precision@5 and success (first result expected) per profile and strategy, a Plotly success matrix, each query's expected and retrieved sources marked hit or miss for a chosen profile and strategy, and the golden queries nobody searched. A tenant without a golden set sees how to upload one; a telemetry outage shows as an error. | `/admin/tenant/{tenant}/evaluation/golden` |
| Embedding atlas | For a chosen tenant and one of its profiles (its own or shipped), a Plotly map of up to the chosen number of its documents placed by their embeddings on the set's two principal axes, with the share of variance each axis shows, and the mapped documents listed by title with their coordinates and text, searchable by title or text. | `/admin/tenant/{tenant}/embeddings/atlas`, `/admin/profiles`, `/admin/profile-templates` |
| Routing evaluation | For a chosen tenant and window (a preset or any 1–720 hours), its routing decisions and the telemetry project they come from: counts by outcome, accuracy, confidence calibration and mean, P50 and P95 latency, a per-agent table with precision, recall and F1 shaded red to green and a grouped bar chart of them, and Plotly charts of confidence by outcome, confidence calibration (ten bins from the lowest confidence to the highest, sized by count, against the diagonal), decisions per hour by agent and success rate per hour. Each decision shows its label, who gave it, the LLM's confidence, whether it awaits review and the reasoning. A reviewer approves an LLM label or relabels a decision, starting from its current label, reasoning and suggested agent; the decision list filters to LLM labels awaiting review. The Labelling panel counts the labels stored in the last 30 days, starts an `llm-annotate` run over the window ("Label with the LLM", followed in Optimization runs), and finds the decisions needing review for a confidence threshold and cap, filtered by priority and by whether the LLM labelled them ("Showing X of Y"), each labelled in place. A telemetry outage shows as an error saying whether the store did not answer or refused the query. A view that fails to render shows the error and its traceback. | `/admin/tenant/{tenant}/routing-decisions`, `/admin/tenant/{tenant}/routing-decisions/{span}/(approve\|label)`, `/admin/tenant/{tenant}/routing-decisions/(annotation-candidates\|label-statistics)`, `/agents/annotations/labels`, `/admin/tenant/{tenant}/optimize` |

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

On `SIGTERM` or `SIGINT` the server stops accepting connections and exits once
its requests finish; a request still running after 5 seconds (an ingest event
stream, a runtime that does not answer) is cut.

## Tests

```bash
npm run typecheck
npm test
```

Vitest covers the result parsing and rendering for each agent's payload, the
run notices, error, route, JSON-field and event-stream parsing, the server's
configuration, agent listing and thread restore, and the runtime proxy against
local HTTP sockets.
`tests/runtime/integration/test_web_client_ag_ui.py` installs this lockfile,
runs the server from source against the runtime's routers and drives it with
the published `@ag-ui/client`. `tests/runtime/integration/test_web_agent_workspace.py`
drives the agent workspace in Chromium (run notices, threads restored from the
runtime, each agent's results), and `tests/runtime/integration/test_web_ops_*.py`
drive each operations view, against the runtime's routers over real Vespa.
