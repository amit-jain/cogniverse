# cogniverse-web

Browser UI for the Cogniverse runtime, built on CopilotKit 1.77. Every agent in
the runtime's registry (`GET /agents/`) appears as a Dot in the sidebar, marked
online or offline (the runtime's `/health` answers healthy or degraded and its
registry has the agent, checked every 30 seconds); picking one opens a
CopilotKit chat with it. The gateway agent opens when the address names no
agent; an address naming an agent the runtime does not serve opens the gateway
with a warning naming it.

Everything acts for one tenant at a time: the active tenant, chosen in the
sidebar or in any operations view's Tenant form, is shared by every view and
agent and kept across reloads. A choice is checked with the runtime's tenant
registry (`GET /admin/tenants/{id}`): a registered tenant becomes active in
its canonical `org:tenant` form, an unknown one is refused with how to
register it, a malformed one (an empty part or more than one `:`, which the
runtime answers with 400) is refused with the runtime's reason, and one the
registry cannot answer for is taken with a warning
that names why (a tenant confirmed earlier keeps being used through such an
outage). Changing the active tenant starts every view and conversation over
for the new tenant, and a run's results show only while the tenant they were
produced for is active. Runs stream over the runtime's AG-UI
surface (`POST /ag-ui/{agent}`): the reply streams token by token and status
phases show above the chat, with the themes and draft summary an agent reports
while it works. A run that fails says why in the conversation, and one that is
stopped says it was cancelled; the runtime saves a cancelled run's message with
a cancelled marker, so after a reload the conversation shows the message and
"The run was cancelled.". An empty message cannot be sent. A search reply names
each hit by its title (and a video segment's time range), never by its backend
document ID; an entity extraction, query enhancement or profile selection reply
is one sentence, with its full result in a panel beside the chat.

The session bar shows the conversation's ID and turn count and the search
settings every run sends: results per search (`top_k`, 1-20, sent as
`forwardedProps.cogniverse.top_k`) and a minimum score below which hits are
hidden. Settings are remembered in the browser.

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
each planned agent's hits under its name, with the orchestration's execution
summary, and an answer agent's key points. An entity extraction shows its
entities with their types and the relationships between them; a query
enhancement the query as asked, the query searched ("Unchanged" when it is the
same), its expansion terms, synonyms, variants, which path produced it and
why; a profile selection the profile chosen, its confidence, intent, modality,
complexity, reasoning and the runners-up with their scores. Each search says what it found for
the question ("Found 2 results for 'q'." or "No results for 'q'."), its result
count, the run's latency, the profile or ensemble profiles searched and the
search mode, and warns about each ensemble profile that did not run. A card
shows the hit's video and document IDs. When a search recorded a telemetry
span, each of its cards can be rated Highly Relevant, Somewhat Relevant or Not
Relevant; the rating is stored on the search's span
(`POST /ag-ui/results/relevance`), where the embedding triplet miner reads it,
and a rating that was not stored shows its reason on the card.

Below the results, each for the conversation's tenant: "Summarize results" runs
the summarizer agent grounded in the hits on screen (`POST
/ag-ui/summarizer_agent` with `forwardedProps.cogniverse.search_results`) and
shows its streamed summary and key points; the count of ratings saved in the
conversation, with "Export annotations" downloading them as JSON under the
conversation's `tenant_id` and thread; "Evaluate this conversation", which
stores an outcome (success, partial, failure) and a 0-1 quality on each search
span of the conversation (`POST /ag-ui/threads/{thread}/evaluation`); and the
conversation's history, each run's question, result count and time. The
browser keeps each conversation's run records and ratings by tenant and thread.

The browser talks only to this package's Node server. The server hosts the
CopilotKit runtime at `/ui-api/copilotkit` and acts for the tenant each request
names in its `x-cogniverse-tenant` header. For a tenant's first run it mints a
harness key named `cogniverse-web <host>` through the runtime's `POST
/admin/harness/keys` (refusing a tenant the registry says is unknown), and
sends every run, thread restore and relevance rating of that tenant with that
key, so the runtime resolves the tenant from the key and one browser can never
read another tenant's runs. Concurrent first runs of a tenant share one mint;
a key the runtime rejects (revoked when its tenant is deleted, say) is
replaced by a new one and the request retried once; a running conversation
can only be joined for the tenant that started it; and the server revokes the
keys it minted when it stops. A runtime that cannot issue a key fails the run
with its reason. The operations views reach the runtime's admin, ingestion and
event routes through `/ui-api/runtime/*`, which forwards an allowlist of those
routes (`src/server/proxy.ts`) and streams their server-sent events; those
routes take their tenant from their path. `GET /ui-api/agents/status` reports the
runtime's health and each agent's. There is no user login, so anyone who can
reach the server can use those routes and act for any registered tenant.

## Operations views

The sidebar's Operations section manages the runtime. Each view says in one
sentence under its heading what it is for, and calls the runtime's existing
routes through `/ui-api/runtime/*`.

| View | What it does | Runtime routes |
|---|---|---|
| Tenants | List, create and delete organizations and tenants, with the organization, tenant and per-organization counts; refresh an organization's tenants; set a tenant's router tier. A new tenant gets the base schemas checked from the runtime's list, its defaults checked to start with. Deletes need the name typed to confirm; deleting the organization shown leaves none shown. | `/admin/organizations`, `/admin/tenants`, `/admin/router-tiers`, `/admin/base-schemas` |
| Backend profiles | For a chosen tenant: list and count the profiles created for it, create one, either blank (its pipeline, strategies and schema config filled with the frame-based ColPali layout) or started from a shipped profile whose every key fills the form, choosing type, embedding type, model loader and process type from the values the runtime lists, with JSON fields for pipeline, strategies, schema config, model-specific parameters and extra config such as inference services; edit its description, pipeline, strategies and model-specific parameters, deploy (or force-redeploy) its schema, and delete it with or without its schema. A create whose requested deploy did not happen says so with the reason; a failed deploy or delete shows the request it made beside the runtime's answer. Shipped profiles are not listed. | `/admin/profiles`, `/admin/profile-templates` |
| Configuration | The system config and, for a chosen tenant, its routing, telemetry and durable-execution configs and each agent's config, as forms generated from the runtime's schema for each section, so a new config field appears without client changes. A save applies to the version the form was loaded at: one made over another operator's save is refused with both versions named. Reload discards unsaved edits, and an agent's config stays open after it is saved. Secrets are never shown; a blank secret keeps its value and one marked to clear is removed. Fields with a fixed set of values are dropdowns, a port outside 1 to 65535 is refused before saving, and an agent's optimizer is set or cleared with a checkbox and edited field by field. Every stored config of the system and of the tenant is listed with its versions, each with when it was created and updated, any of which can be restored as the next version. A tenant's configs export to a JSON download (secrets included, every version when asked) and import from one, whole or not at all, after a preview listing the configs the file holds. The config store's implementation, health, backend and counts are shown. | `/admin/config/sections`, `/admin/config/entries`, `/admin/config/history`, `/admin/config/rollback`, `/admin/config/export`, `/admin/config/import`, `/admin/config/stats`, `/admin/config/health` |
| Ingestion | Upload a file to a tenant: the form lists the tenant's profiles that ingest an uploaded file (the default one chosen) and the backend, offers only the files the chosen profiles read, and refuses a file a chosen profile cannot read before anything is sent; a runtime that does not answer is reported and nothing can be uploaded. The file goes to each chosen profile (optionally forcing a re-ingest of identical bytes), and each ingest is followed live through queued, running, and complete, failed or cancelled, with its result, error or cancellation reason; an upload answer without an ingest ID is an error. A completion that fed no documents shows as a failure, and an ingest with no terminal state after 900 s is given up. Each upload's batch line says when all its profiles ingested, or which failed. Each tenant has its own followed ingests and batches, which survive a reload. Follow an existing ingest by its ID. | `/ingestion/profiles`, `/ingestion/upload`, `/ingestion/{id}/events`, `/ingestion/{id}/status` |
| Optimization runs | For a chosen tenant: download an optimizer's example template, choose training-example JSON files to preview and check in the browser, and upload a valid one as a named operator, which approves its examples into the tenant's training dataset for that optimizer's runs. Start a run in any mode the runtime accepts over a chosen lookback (for the synthetic mode, generating training data for the optimizers chosen, which then wait in Approvals), with the last run started linked. List its runs with mode, trigger, phase and times (polled while any run is unsettled), open a run to see its steps, Argo message and what it waits for, cancel an unsettled run, and retry the failed steps of a failed one; a run Argo deleted after its time-to-live says so. Generate the optimization report with `detailed_report_agent`, following its progress, and download it as JSON; the panel says whether that agent is registered. | `/admin/tenant/training-example-templates`, `/admin/tenant/{tenant}/training-examples`, `/admin/tenant/optimize-modes`, `/admin/tenant/{tenant}/optimize`, `/admin/tenant/{tenant}/optimize/runs`, `/admin/tenant/{tenant}/optimize/runs/{name}`, `.../cancel`, `.../retry`, `/admin/tenant/{tenant}/optimize/report`, `/agents/` |
| Optimization framework | For a chosen tenant, eight sections. **Overview**: annotation count, the golden dataset size built this session, run count, last run's age and phase, the workflow and recent history. **Search annotations**: fetch the tenant's recorded searches (the Search agent's included, under the query the user typed) over 1–168 hours, ten per page, each with its query, top five results, profile, strategy and latency, and rate it with thumbs, 1–5 stars or a 0–1 relevance score plus notes. **Golden dataset**: build ground truth from searches rated at or above a minimum over 1–90 days, see a sample and download it as JSON. **Synthetic data**: generate examples for an optimizer (count, backend sample size, sampling strategy, max profiles, human review with the auto-approval threshold and expected review rate) as an Argo run, follow it, and read its outcome: approval counts, the profiles sampled and why, the schema and generation time, sample examples, the first five pending items with their query, reasoning, entities, confidence, retries, band and generation details (decided in Approvals), and download it as JSON; earlier synthetic runs reopen. **Module optimization**: start a routing, workflow or unified run with max iterations, lookback and whether approved synthetic data trains it, or else a golden dataset picked from the tenant's datasets, uploaded as CSV or named by hand, and follow its phase. **Reranking**: annotations collected against a minimum. **Profile selection**: span analysis (profile usage, quality metrics, per-profile quality), train the XGBoost profile recommender (accuracies, samples, feature importance) and ask it for a query's profile and features. **Metrics**: over 7, 30 or 90 days, routing accuracy, decisions, mean decision latency (— when no decision is timed), calibration and per-agent precision, recall and F1, evaluation activity and training runs per day. Telemetry outages show as errors, never as empty data. | `/admin/tenant/{tenant}/search-annotations`, `.../search-annotations/count`, `.../search-annotations/{span}`, `.../golden-dataset`, `.../synthetic/settings`, `.../optimize`, `.../optimize/runs`, `.../optimize/runs/{name}`, `.../optimize/runs/{name}/synthetic`, `.../datasets`, `.../profile-selection/(analysis\|train\|model\|predict)`, `.../optimization-metrics` |
| Memory | For a chosen tenant and namespace (the user's memories, an agent's, or a system one): its store's health (checked on opening and on demand), user and agent IDs, and live and archived counts; list or semantically search up to the chosen number of memories ("Show all" drops the search and lists with the chosen number, or the one in use while the box holds no valid number), a search showing each one's similarity score, with created and updated times and each memory's every field and metadata on opening it; add one with a category and JSON metadata, the saved record shown; delete one, and clear the namespace after typing its name. System namespaces are read-only. A store outage shows as an error, never as an empty namespace. | `/admin/tenant/{tenant}/memories`, `/admin/tenant/{tenant}/memories/stats`, `/admin/tenant/{tenant}/memories/health`, `/admin/tenant/{tenant}/memories/{id}`, `/agents/` |
| Approvals | Four sections for a chosen tenant. **Pending**: the generated examples awaiting review, each with its schema, confidence, retry count, query, reasoning, entities, the entity self-consistency check's agreement per sampled mention, its generation metadata and its JSON. As a named reviewer, approve an item into the training dataset, or open a rejection (which can be cancelled) with feedback and an editor per correctable field; a rejection regenerates the item (or, for a workflow record, merges the corrections) for another review. A decision another reviewer already made is refused rather than overwritten. **Approved**: every approved and auto-approved item with its reviewer and time. **Rejected**: every rejected item with its feedback, corrections, reviewer and replacement; one nothing replaced can be regenerated. **Statistics**: totals per status, the approval rate and the mean confidence per status. | `/admin/tenant/{tenant}/approvals`, `/admin/tenant/{tenant}/approvals/{batch}/{item}`, `.../regenerate`, `/admin/tenant/{tenant}/approvals/history`, `/admin/tenant/{tenant}/approvals/stats` |
| Annotation queue | The routing decisions queued for human review across tenants: counts by status, then the pending (by priority), assigned (by due time) and expired requests, each with the time it was queued, its query, chosen agent, confidence, outcome and reason. As a named reviewer, assign a pending request to yourself and label a pending or assigned one with a reason; the label is stored on the decision's span for the tenant before the request completes. A request someone else already labelled, and a queue or telemetry outage, show the runtime's reason. | `/agents/annotations/queue`, `/agents/annotations/queue/{span}/assign`, `/agents/annotations/queue/{span}/complete`, `/agents/annotations/labels` |
| Workflow reviews | For a chosen tenant, window and cap (1–500 workflows), the orchestration workflows the orchestrator recorded, newest first and counted, with their query, pattern, agents, time, outcome and latest review. As a named reviewer, rate a workflow's quality and say whether its pattern was optimal (yes, no with a suggested pattern, or unsure, stored as not optimal), whether its agents and execution order were right, with suggestions, what went well and wrong, and notes; the review is stored on the workflow's span and shows in the list at once. A review that is not stored shows the runtime's reason with its error code, failure type and status. A telemetry outage shows as an error, never as an empty list. | `/admin/tenant/{tenant}/orchestration-workflows`, `/admin/tenant/{tenant}/orchestration-workflows/{span}/annotation` |
| Profile metrics | For a chosen tenant and lookback (1–720 hours), its profile selections per modality: a card per modality (count and P95), count, P50, P95 and P99 latency and success rate, a pie of queries per modality and bars for selections and P95 latency. An empty window names the telemetry project and the agent to drive; selections without a modality are told apart; a slow store shows as a warning, an outage or a missing telemetry provider as an error. Answers are kept for 30 s; Refresh reads again. | `/admin/tenant/{tenant}/telemetry/profile-selection` |
| RLM A/B | For a chosen tenant and lookback (0.1–720 hours), its RLM A/B comparisons: average latency, token and judge-score change with RLM and how often RLM fell back, the same per queries dataset with a latency bar per dataset, and each comparison newest first with its A/B id. An empty window gives the full `cogniverse-optim --mode ab-compare --tenant-id … --queries-dataset …` command; an outage or a missing telemetry provider shows as an error. | `/admin/tenant/{tenant}/telemetry/rlm-ab` |
| Analytics | For a chosen tenant, its traces in a window (the last 15 minutes, hour, 6 hours, day or week, or a custom UTC range of up to 30 days) filtered by an operation regular expression, profiles and strategies: counts, success rate against the 95% target and latency, refreshed on demand or automatically every 5-300 seconds, with when it was last refreshed. A read that takes over 5.5 seconds says the figures shown may be stale until it answers; a telemetry outage shows as an error with a retry hint, never as no traces. Then an overview (latency percentiles, traces per operation as bars and a donut, and a per-operation table), latency and trace counts over 1, 5, 15 or 60 minute buckets, latency histogram, box, violin and cumulative distribution (with P50, P90, P95 and P99) grouped by operation, profile, strategy or outcome, a mean-latency heatmap over two chosen fields, latency outliers (Tukey's bounds, with the bound and P50, P95 and P99 drawn) or hourly error-rate outliers, and every trace, searchable by trace ID and/or operation, sortable by time, duration, operation or outcome, 20 to a page (a new search, scope or order starts at the first page). "Show raw data" lists every field of every trace, and the window's traces download as JSON (with their statistics and the root causes found), CSV or an HTML report. Root causes runs over the filtered traces, previewing the slow threshold as the percentile changes: their failed ones and, optionally, the successful ones slower than a chosen percentile, giving the totals, the hypotheses (with evidence, affected traces and the Phoenix query and link to find them, and a suggested action), the recommendations with the components they affect, the failures by error kind, operation, profile, strategy, hour and burst, and the slow traces by operation, profile and strategy with how much slower they are. A found analysis stays while the traces it covers do. Charts are Plotly, loaded the first time one is shown, and each section explains how to read it. | `/admin/tenant/{tenant}/telemetry/traces`, `/admin/tenant/{tenant}/telemetry/root-causes`, `/admin/tenant/{tenant}/telemetry/phoenix` |
| Evaluation | For a chosen tenant, two sources. **Golden set**: its searches of its golden set's queries over a typed lookback (1–2160 hours), from search requests and the Search agent alike, matched by the query the user typed. **Phoenix datasets**: the datasets the tenant owns, newest first, chosen by name and example count, with their example count, creation date and a View in Phoenix link (when the runtime knows the Phoenix UI address), each scored the same way. Both show MRR, nDCG@10, recall@1 and @5, precision@5 and success per profile and strategy, a Plotly success matrix, and per-profile tabs with nested per-strategy tabs holding the strategy's MRR, recall@1 and @5 and each query's expected and retrieved sources with colour-coded scores (from 0.7 good, from 0.3 fair). A tenant without a golden set sees how to upload one; an outage shows as an error. Answers are kept for a minute; Refresh reads again. | `/admin/tenant/{tenant}/evaluation/golden`, `/admin/tenant/{tenant}/evaluation/datasets`, `/admin/tenant/{tenant}/evaluation/dataset` |
| Embedding atlas | For a chosen tenant and one of its profiles (its own or shipped), a map of up to the chosen number of its documents by their embeddings, in one of two projections, or an uploaded embedding export file. **PCA, read live**: the set's two principal axes with the share of variance each shows. **UMAP with clusters**: a cached UMAP layout with automatic clusters named by their terms (each name unique on the map, file extensions and id-like words left out), typed queries placed on it as stars with their three most similar documents, an optional density view, hover naming each point's title, cluster and text, lasso or box selection that filters the table and the per-cluster, per-kind and text-length charts, and Recompute to lay the documents out again. **Exported file**: a parquet file written by `scripts/export_backend_embeddings.py` (up to 64 MiB) is uploaded and shown with the same map, clusters, lasso, charts and query analysis: its x/y places when every row has one, otherwise UMAP over its `embedding` column; its `is_query` rows are the queries, each with its three most similar documents by the file's `query_similarity_<row>` column or by cosine similarity. An empty file, one with neither x/y nor embeddings, or one that is not parquet is refused with the reason. The mapped documents are listed with their coordinates and text, searchable by title or text. | `/admin/tenant/{tenant}/embeddings/atlas`, `/admin/tenant/{tenant}/embeddings/atlas/umap`, `/admin/tenant/{tenant}/embeddings/atlas/export`, `/admin/profiles`, `/admin/profile-templates` |
| Routing evaluation | For a chosen tenant and window (a preset or any 1–720 hours), its routing decisions and the telemetry project they come from: counts by outcome, accuracy, confidence calibration and mean, P50 and P95 latency, a per-agent table with precision, recall and F1 shaded red to green and a grouped bar chart of them, and Plotly charts of confidence by outcome, confidence calibration (ten bins from the lowest confidence to the highest, sized by count, against the diagonal), decisions per hour by agent and success rate per hour. Each decision shows its label, who gave it, the LLM's confidence, whether it awaits review and the reasoning. A reviewer approves an LLM label the LLM was sure of (one it flagged for review is relabelled instead; the runtime refuses its approval with 409) or relabels a decision, starting from its current label, reasoning and suggested agent; the decision list filters to LLM labels awaiting review. The Labelling panel counts the labels stored in the last 30 days, starts an `llm-annotate` run over the window ("Label with the LLM", followed in Optimization runs), and finds the decisions needing review for a confidence threshold and cap, filtered by priority and by whether the LLM labelled them ("Showing X of Y"), each labelled in place. A telemetry outage shows as an error saying whether the store did not answer or refused the query. A view that fails to render shows the error and its traceback. | `/admin/tenant/{tenant}/routing-decisions`, `/admin/tenant/{tenant}/routing-decisions/{span}/(approve\|label)`, `/admin/tenant/{tenant}/routing-decisions/(annotation-candidates\|label-statistics)`, `/agents/annotations/labels`, `/admin/tenant/{tenant}/optimize` |

## Setup

Use Node.js 22. Copy `.env.example` to `.env` and set:

| Variable | Meaning |
|---|---|
| `COGNIVERSE_RUNTIME_URL` | Runtime base URL, including any ingress prefix; the server needs its `/admin` routes, `/admin/harness/keys` among them |
| `PORT` | Port the server listens on (default `4000`) |
| `HOST` | Interface the server binds (default `127.0.0.1`) |

The Analytics view's trace links and the Evaluation view's dataset links point
at the runtime's `PHOENIX_UI_URL` (chart value `phoenix.uiUrl`, the Phoenix UI
address browsers reach); without it they are left out.

```bash
cd clients/web
npm ci
npm run dev        # server on PORT, Vite on :5173 proxying /ui-api to it
npm run build && npm start   # one server serving the built client
```

The server turns off CopilotKit's usage telemetry unless
`COPILOTKIT_TELEMETRY_DISABLED=false` is set.

On `SIGTERM` or `SIGINT` the server stops accepting connections, revokes the
harness keys it minted, and exits once its requests finish; a request still
running after 5 seconds (an ingest event stream, a runtime that does not
answer) is cut.

## Deploy

`Dockerfile` builds the image (`docker build -f clients/web/Dockerfile -t cogniverse/web:dev clients/web`):
the built client served by the Node server on port 4000, with `/healthz` for
probes. The Helm chart deploys it as the `web` component behind the ingress's
`/` path; `cogniverse up` builds and imports it and serves it at
http://localhost:28400. The client needs no secure context, so it also works
over plain http on another hostname. Chart values and deployment are in
[docs/modules/web-client.md](../../docs/modules/web-client.md).

## Tests

```bash
npm run typecheck
npm test
```

Vitest covers the result parsing and rendering for each agent's payload, search
facts, key points and orchestration summaries, the entity, query enhancement
and profile selection panels, the conversation records,
export and settings, the summarize stream, the default agent, the run notices,
error, route, JSON-field and event-stream parsing, the active tenant and its
registration check, the Analytics view's figures and exports, the server's
configuration, agent listing and status, per-tenant keys, thread restore
(a cancelled run's marker as its notice) and ownership, and the runtime proxy against local HTTP sockets.
`tests/runtime/integration/test_web_client_ag_ui.py` installs this lockfile,
runs the server from source against the runtime's routers and drives it with
the published `@ag-ui/client`. `tests/runtime/integration/test_web_agent_workspace.py`
drives the agent workspace in Chromium (run notices, threads restored from the
runtime, each agent's results, progress, navigation),
`tests/runtime/integration/test_web_search_session.py` a search conversation
against the dispatcher's own search agent and real Phoenix (settings, results,
ratings export, summarize, evaluation), `tests/runtime/integration/test_web_ops_shell.py`
the active tenant, the tenant gate, the agents' status and two sessions on
different tenants against the real tenant registry and key store,
`tests/runtime/integration/test_web_ops_analytics.py` the Analytics view against
real Phoenix, and `tests/runtime/integration/test_web_ops_*.py` each other
operations view, against the runtime's routers over real Vespa.
