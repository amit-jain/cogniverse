"""The web client's server drives Cogniverse agents through CopilotKit.

The pinned ``clients/web`` lockfile is installed and its Node server runs from
source against the runtime's ``/ag-ui`` and ``/agents`` routers on a real
uvicorn socket. A driver uses the published ``@ag-ui/client`` the browser
bundle uses, POSTing runs to the CopilotKit runtime the server hosts, so each
run crosses every hop a browser run crosses: CopilotKit runtime -> the
server's ``HttpAgent`` with the harness key it minted for the run's tenant
through the runtime's ``/admin/harness/keys`` -> the AG-UI router, which
resolves the tenant from that key -> the real dispatcher -> the agent.

The driver parses the final state with the client's own ``resultsOf``, so the
result cards are pinned against what the runtime actually sends. A recorder
stands in for CopilotKit's telemetry sink and must receive nothing. Node and
npm are required; their absence is a failure, not a skip.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import admin, ag_ui, agents, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.memory_store import InMemoryConfigStore
from tests.utils.node_env import node_env
from tests.utils.web_client import (
    free_port,
    install_web_client,
    recording_telemetry_sink,
    serve_app,
    serve_web,
    web_server_process,
)
from tests.utils.web_ops import harness_key_admin

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]


TENANT = "acme:web"
OTHER_TENANT = "beta:web"
QUERY = "Which clips show the tower at night?"
STATUS_PHASE = "retrieval"
STATUS_MESSAGE = "Searching 2 profiles"
TOOL_CALL_ID = "call_web_write_file"
TOOL_OUTPUT = "saved to notes.txt"
SEARCH_RESULTS = [
    {
        "id": "v7_seg_3",
        "document_id": "id:video:video::v7_seg_3",
        "score": 0.91,
        "metadata": {
            "video_id": "v7",
            "video_title": "Tower at night",
            "audio_transcript": "the tower lights up",
        },
        "temporal_info": {"start_time": 42.0, "end_time": 48.5},
    },
    {
        "id": "v2_seg_0",
        "document_id": "id:video:video::v2_seg_0",
        "score": 0.64,
        "metadata": {"video_id": "v2", "description": "a skyline at dusk"},
    },
]

DRIVER_TS = """
import { HttpAgent } from '@ag-ui/client';
import { resultsOf } from './src/client/ResultCards.tsx';

const WEB = process.env.WEB_URL;

async function run(agentId, messages, tools = [], thread = agentId, tenant = process.env.TENANT) {
  const agent = new HttpAgent({
    agentId,
    url: `${WEB}/ui-api/copilotkit/agent/${agentId}/run`,
    threadId: `thread-${thread}`,
    headers: tenant ? { 'x-cogniverse-tenant': tenant } : {},
  });
  agent.setMessages(messages);
  const events = [];
  const custom = [];
  await agent.runAgent(
    { runId: `run-${agentId}-${messages.length}`, tools },
    {
      onEvent: ({ event }) => {
        events.push(event.type);
        if (event.type === 'CUSTOM') custom.push({ name: event.name, value: event.value });
      },
    },
  );
  return { events, custom, messages: agent.messages, state: agent.state };
}

const reply = (outcome) =>
  outcome.messages.filter((m) => m.role === 'assistant').map((m) => m.content);

if (process.env.SCENARIO === 'concurrent') {
  const tenants = process.env.TENANTS.split(',');
  const queries = Array.from({ length: Number(process.env.RUNS) }, (_, i) => `query ${i}`);
  const outcomes = await Promise.all(
    queries.map((query, i) =>
      run(
        process.env.AGENT,
        [{ id: 'u1', role: 'user', content: query }],
        [],
        `c${i}`,
        tenants[i % tenants.length],
      ),
    ),
  );
  console.log(
    JSON.stringify(
      outcomes.map((outcome) => ({
        reply: reply(outcome),
        tenant: outcome.state.tenant_id,
        cards: resultsOf(outcome.state).map((card) => card.id),
      })),
    ),
  );
  process.exit(0);
}

if (process.env.SCENARIO === 'sequence') {
  const outcomes = [];
  for (const query of process.env.QUERIES.split('|')) {
    const events = [];
    let error = null;
    try {
      const agent = new HttpAgent({
        agentId: 'search_agent',
        url: `${WEB}/ui-api/copilotkit/agent/search_agent/run`,
        threadId: `seq-${outcomes.length}`,
        headers: { 'x-cogniverse-tenant': process.env.TENANT },
      });
      agent.setMessages([{ id: 'u1', role: 'user', content: query }]);
      await agent.runAgent({}, { onEvent: ({ event }) => events.push(event) });
      outcomes.push({
        reply: agent.messages.filter((m) => m.role === 'assistant').map((m) => m.content),
        error: events.find((event) => event.type === 'RUN_ERROR')?.message ?? null,
      });
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
      outcomes.push({ reply: [], error });
    }
  }
  console.log(JSON.stringify(outcomes));
  process.exit(0);
}

if (process.env.SCENARIO === 'fault') {
  const events = [];
  let error = null;
  try {
    const agent = new HttpAgent({
      agentId: 'search_agent',
      url: `${WEB}/ui-api/copilotkit/agent/search_agent/run`,
      headers: process.env.TENANT ? { 'x-cogniverse-tenant': process.env.TENANT } : {},
    });
    agent.setMessages([{ id: 'u1', role: 'user', content: process.env.QUERY }]);
    await agent.runAgent({}, { onEvent: ({ event }) => events.push(event) });
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  const listed = await fetch(`${WEB}/ui-api/agents`);
  console.log(
    JSON.stringify({ events, error, listedStatus: listed.status, listed: await listed.json() }),
  );
  process.exit(0);
}

const listed = await (await fetch(`${WEB}/ui-api/agents`)).json();
const info = await (await fetch(`${WEB}/ui-api/copilotkit/info`)).json();

const search = await run('search_agent', [
  { id: 'u1', role: 'user', content: process.env.QUERY },
]);

const writeFile = {
  name: 'write_file',
  description: 'Write text to a file in the workspace.',
  parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
};
const user = { id: 'u1', role: 'user', content: process.env.QUERY };
const suspended = await run('tool_agent', [user], [writeFile]);
const call = suspended.messages.at(-1).toolCalls[0];
const resumed = await run(
  'tool_agent',
  [
    ...suspended.messages,
    { id: 't1', role: 'tool', toolCallId: call.id, content: process.env.TOOL_OUTPUT },
  ],
  [writeFile],
);

console.log(
  JSON.stringify({
    listed,
    infoAgents: Object.keys(info.agents ?? {}).sort(),
    search: {
      events: search.events,
      custom: search.custom,
      reply: reply(search),
      agent: search.state.agent,
      tenant: search.state.tenant_id,
      cards: resultsOf(search.state),
    },
    suspended: {
      events: suspended.events,
      toolCall: { id: call.id, name: call.function.name, arguments: call.function.arguments },
    },
    resumed: {
      events: resumed.events,
      reply: resumed.messages.at(-1).content,
      roles: resumed.messages.map((m) => m.role),
    },
  }),
);
"""


class WebDeps(AgentDeps):
    pass


class WebInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class WebSearchOutput(AgentOutput):
    summary: str = ""
    results: list = []


class WebSearchAgent(AgentBase[WebInput, WebSearchOutput, WebDeps]):
    """A status phase, then the summary tokens, then public-shaped hits."""

    async def _process_impl(self, input: WebInput) -> WebSearchOutput:
        self.emit_progress(STATUS_PHASE, STATUS_MESSAGE)
        summary = f"[{input.tenant_id}] Two clips match: {input.query}"
        accumulated = ""
        for start in range(0, len(summary), 9):
            chunk = summary[start : start + 9]
            accumulated += chunk
            self.emit_progress(
                "token",
                chunk,
                data={"accumulated": accumulated, "output_field": "summary"},
            )
            await asyncio.sleep(0)
        return WebSearchOutput(summary=summary, results=list(SEARCH_RESULTS))


class WebToolOutput(AgentOutput):
    answer: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class WebToolAgent(AgentBase[WebInput, WebToolOutput, WebDeps]):
    """Suspends on the browser's first tool, then answers from its result."""

    async def _process_impl(self, input: WebInput) -> WebToolOutput:
        if not input.tool_results:
            return WebToolOutput(
                pending_tool_calls=[
                    {
                        "id": TOOL_CALL_ID,
                        "name": input.external_tools[0]["function"]["name"],
                        "arguments": {"text": input.query},
                    }
                ],
                continuation_state={"plan": f"plan-for-{input.tenant_id}"},
            )
        return WebToolOutput(
            answer=(
                f"tool said: {input.tool_results[0]['content']}; "
                f"tenant={input.tenant_id}; "
                f"resumed={input.continuation_state.get('plan')}"
            )
        )


# Runs of the barrier agent wait inside the agent until ``expected`` of them
# have entered, so they are all in flight at once.
_barrier = {"expected": 0, "entered": 0}
_barrier_lock = threading.Lock()


class WebBarrierAgent(AgentBase[WebInput, WebSearchOutput, WebDeps]):
    """Holds its turn until every run of the scenario has entered, then
    answers with a hit named after its tenant."""

    async def _process_impl(self, input: WebInput) -> WebSearchOutput:
        with _barrier_lock:
            _barrier["entered"] += 1
        deadline = time.monotonic() + 60
        while _barrier["entered"] < _barrier["expected"]:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"only {_barrier['entered']} of {_barrier['expected']} runs entered"
                )
            await asyncio.sleep(0.01)
        return WebSearchOutput(
            summary=f"[{input.tenant_id}] {input.query}",
            results=[
                {
                    "id": f"{input.tenant_id}-hit",
                    "document_id": f"id:video:video::{input.tenant_id}-hit",
                    "score": 1.0,
                    "metadata": {"video_title": input.tenant_id},
                }
            ],
        )


_AGENT_CLASSES = {
    "search_agent": f"{__name__}:WebSearchAgent",
    "tool_agent": f"{__name__}:WebToolAgent",
    "barrier_agent": f"{__name__}:WebBarrierAgent",
}
_TOKEN_STREAMING = {"search_agent": True, "tool_agent": False, "barrier_agent": False}
AGENTS = list(_AGENT_CLASSES)


@pytest.fixture(scope="module")
def web_client_dir(tmp_path_factory):
    """The client's sources with its lockfile installed beside them."""
    root = install_web_client(tmp_path_factory.mktemp("web_client"))
    (root / "driver.ts").write_text(DRIVER_TS)
    return root


@pytest.fixture(scope="module")
def live_runtime(session_state_lifespan):
    """The /ag-ui and /agents routers with the two agents on a real socket."""
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    for agent_name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=agent_name,
                url="http://localhost:8000",
                capabilities=["web"],
                streams_answer_tokens=_TOKEN_STREAMING[agent_name],
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    # No conversation memory is configured: each run's turn takes its place in
    # the ledger and is not stored.
    dispatcher._conversation_store_factory = lambda tenant_id: None

    app = FastAPI(lifespan=session_state_lifespan)
    app.state.dispatcher = dispatcher
    app.include_router(ag_ui.router, prefix="/ag-ui")
    app.include_router(agents.router, prefix="/agents")
    agents.set_agent_registry(registry)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({})

    with harness_key_admin(app, config_manager) as keys, serve_app(app) as url:
        yield SimpleNamespace(url=url, keys=keys)

    openai_compat.set_dispatcher_provider(None)
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


@pytest.fixture(scope="module")
def session_state_lifespan(workflow_state_redis_url):
    """A lifespan opening the continuation store and the app's dispatcher's
    conversation ledger on the server's own loop."""

    @asynccontextmanager
    async def lifespan(app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        prefix = f"test:web:{uuid.uuid4().hex}"
        openai_compat.set_continuation_store(
            ContinuationStore(redis, key_prefix=prefix)
        )
        app.state.dispatcher.set_conversation_ledger(
            ConversationLedger(
                redis,
                save_lease_s=CONVERSATION_SAVE_LEASE_S,
                failure_capacity=CONVERSATION_PERSIST_FAILURE_CAPACITY,
                key_prefix=f"{prefix}:conversation",
            )
        )
        try:
            yield
        finally:
            app.state.dispatcher.set_conversation_ledger(None)
            openai_compat.set_continuation_store(None)
            await redis.aclose()

    return lifespan


@pytest.fixture()
def telemetry_sink():
    with recording_telemetry_sink() as sink:
        yield sink


@pytest.fixture()
def web_server(web_client_dir, live_runtime, telemetry_sink):
    with serve_web(
        web_client_dir, live_runtime.url, telemetry_url=telemetry_sink[0]
    ) as url:
        yield url


def _drive(client_dir: Path, web_url: str, **scenario: str):
    """Run the driver against the web server and return its JSON report; runs
    name ``TENANT`` unless the scenario says otherwise."""
    node = shutil.which("node")
    proc = subprocess.run(
        [node, "--import", "tsx", str(client_dir / "driver.ts")],
        cwd=client_dir,
        capture_output=True,
        text=True,
        timeout=180,
        env=node_env(
            node, WEB_URL=web_url, QUERY=QUERY, **{"TENANT": TENANT, **scenario}
        ),
    )
    assert proc.returncode == 0, (
        f"web driver failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _live_keys(runtime, tenant):
    """The tenant's keys the runtime has not revoked, as (tenant, name)."""
    return [
        (key["tenant_id"], key["name"])
        for key in runtime.keys.list(tenant)["keys"]
        if not key["revoked"]
    ]


def test_a_browser_run_reaches_the_agent_through_copilotkit(
    web_client_dir, live_runtime, web_server, telemetry_sink
):
    result = _drive(web_client_dir, web_server, TOOL_OUTPUT=TOOL_OUTPUT)

    # The Dots come from the runtime's registry, in registry order, and the
    # CopilotKit runtime serves exactly those agents.
    assert result["listed"] == {"agents": AGENTS}
    assert result["infoAgents"] == sorted(AGENTS)

    search = result["search"]
    summary = f"[{TENANT}] Two clips match: {QUERY}"
    assert search["events"] == [
        "RUN_STARTED",
        "STEP_STARTED",
        "CUSTOM",
        "STEP_FINISHED",
        "STEP_STARTED",
        "CUSTOM",
        "TEXT_MESSAGE_START",
        *["TEXT_MESSAGE_CONTENT"] * len(range(0, len(summary), 9)),
        "TEXT_MESSAGE_END",
        "STATE_SNAPSHOT",
        "STEP_FINISHED",
        "RUN_FINISHED",
    ]
    assert search["custom"] == [
        {
            "name": ag_ui.STATUS_EVENT,
            "value": {"phase": "starting", "message": "Running search_agent"},
        },
        {
            "name": ag_ui.STATUS_EVENT,
            "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
        },
    ]
    # The key the server minted for the run's tenant resolved to it, and the
    # result names the tenant it was produced for.
    assert search["reply"] == [summary]
    assert search["agent"] == "search_agent"
    assert search["tenant"] == TENANT
    assert search["cards"] == [
        {
            "id": "v7_seg_3",
            "ratingId": "id:video:video::v7_seg_3",
            "score": 0.91,
            "title": "Tower at night",
            "snippet": "the tower lights up",
            "start": 42.0,
            "end": 48.5,
            "videoId": "v7",
            "documentId": "id:video:video::v7_seg_3",
        },
        {
            "id": "v2_seg_0",
            "ratingId": "id:video:video::v2_seg_0",
            "score": 0.64,
            "snippet": "a skyline at dusk",
            "videoId": "v2",
            "documentId": "id:video:video::v2_seg_0",
        },
    ]

    suspended = result["suspended"]
    assert suspended["events"] == [
        "RUN_STARTED",
        "STEP_STARTED",
        "CUSTOM",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "STEP_FINISHED",
        "RUN_FINISHED",
    ]
    assert suspended["toolCall"] == {
        "id": TOOL_CALL_ID,
        "name": "write_file",
        "arguments": json.dumps({"text": QUERY}),
    }

    resumed = result["resumed"]
    assert resumed["reply"] == (
        f"tool said: {TOOL_OUTPUT}; tenant={TENANT}; resumed=plan-for-{TENANT}"
    )
    assert resumed["roles"] == ["user", "assistant", "tool", "assistant"]

    # One key, minted under the server's name, served all three runs (the
    # keys of servers that ran before are revoked).
    assert _live_keys(live_runtime, TENANT) == [
        (TENANT, f"cogniverse-web {socket.gethostname()}")
    ]

    # CopilotKit's own usage reporting stays off in a self-hosted deployment.
    assert telemetry_sink[1] == []


def test_concurrent_runs_each_get_their_own_reply(web_client_dir, web_server):
    """Runs in flight together through one server, each on its own thread,
    come back with their own query and nothing of another run's."""
    runs = 6
    outcomes = _drive(
        web_client_dir,
        web_server,
        SCENARIO="concurrent",
        AGENT="search_agent",
        TENANTS=TENANT,
        RUNS=str(runs),
    )
    assert [outcome["reply"] for outcome in outcomes] == [
        [f"[{TENANT}] Two clips match: query {i}"] for i in range(runs)
    ]


def test_two_tenants_in_flight_together_see_only_their_own_results(
    web_client_dir, live_runtime, web_server
):
    """Runs for two tenants, all inside the agent at once, each come back
    with their own tenant's reply and hits; the first runs of each tenant
    race for its key and share one mint."""
    runs = 8
    _barrier.update(expected=runs, entered=0)
    outcomes = _drive(
        web_client_dir,
        web_server,
        SCENARIO="concurrent",
        AGENT="barrier_agent",
        TENANTS=f"{TENANT},{OTHER_TENANT}",
        RUNS=str(runs),
    )
    assert _barrier["entered"] == runs
    tenants = [(TENANT, OTHER_TENANT)[i % 2] for i in range(runs)]
    assert outcomes == [
        {
            "reply": [f"[{tenant}] query {i}"],
            "tenant": tenant,
            "cards": [f"{tenant}-hit"],
        }
        for i, tenant in enumerate(tenants)
    ]
    name = f"cogniverse-web {socket.gethostname()}"
    assert [_live_keys(live_runtime, t) for t in (TENANT, OTHER_TENANT)] == [
        [(TENANT, name)],
        [(OTHER_TENANT, name)],
    ]


def test_a_revoked_key_is_replaced_without_failing_the_run(
    web_client_dir, live_runtime, web_server
):
    """The runtime revokes a tenant's keys (as deleting the tenant does); the
    next run is answered 401, and the server mints a new key and runs it."""
    first = _drive(web_client_dir, web_server, SCENARIO="sequence", QUERIES="first")
    before = {key["key_hash"] for key in live_runtime.keys.list(TENANT)["keys"]}
    revoked = live_runtime.keys.revoke_tenant(TENANT)
    second = _drive(web_client_dir, web_server, SCENARIO="sequence", QUERIES="second")
    assert (first, revoked, second) == (
        [{"reply": [f"[{TENANT}] Two clips match: first"], "error": None}],
        1,
        [{"reply": [f"[{TENANT}] Two clips match: second"], "error": None}],
    )
    live = [
        key["key_hash"]
        for key in live_runtime.keys.list(TENANT)["keys"]
        if not key["revoked"]
    ]
    assert (len(live), live[0] in before) == (1, False)


def test_the_server_revokes_its_keys_when_it_stops(
    web_client_dir, live_runtime, telemetry_sink
):
    tenant = "gamma:web"
    with serve_web(
        web_client_dir, live_runtime.url, telemetry_url=telemetry_sink[0]
    ) as web_url:
        ran = _drive(
            web_client_dir, web_url, SCENARIO="sequence", QUERIES="hi", TENANT=tenant
        )
        held = _live_keys(live_runtime, tenant)
    assert (ran, held) == (
        [{"reply": [f"[{tenant}] Two clips match: hi"], "error": None}],
        [(tenant, f"cogniverse-web {socket.gethostname()}")],
    )
    assert _live_keys(live_runtime, tenant) == []


KEY_TTL_S = 6


def _keyed_ping(url: str, headers: dict) -> int:
    """POST an empty relevance rating; the runtime resolves the bearer key
    before reading the body, so a valid key answers 400 and a refused one
    401."""
    response = httpx.post(
        f"{url}/results/relevance",
        content="{}",
        headers={"content-type": "application/json", **headers},
        timeout=30,
    )
    return response.status_code


def _wait_until(moment: float) -> None:
    time.sleep(max(0.0, moment - time.monotonic()))


def test_a_killed_servers_key_expires_while_a_running_server_renews_its_own(
    web_client_dir, live_runtime, telemetry_sink
):
    """Two servers act for one tenant with short-lived keys; one is killed
    with SIGKILL, so it never revokes its key. Its key stops authenticating
    once its ttl passes and lists as revoked, while the other server replaces
    its key at half the ttl, the superseded key staying valid until it
    expires, and keeps serving runs."""
    tenant = "delta:web"
    presented: list[str] = []

    def recording(key: str) -> str:
        presented.append(key)
        return live_runtime.keys.resolve(key)

    def taken() -> list[str]:
        keys = list(dict.fromkeys(presented))
        presented.clear()
        return keys

    def listed() -> dict:
        return {
            key["key_hash"]: (
                key["revoked"],
                (
                    datetime.fromisoformat(key["expires_at"])
                    - datetime.fromisoformat(key["created_at"])
                ).total_seconds(),
            )
            for key in live_runtime.keys.list(tenant)["keys"]
        }

    def digest(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def at_runtime(key: str) -> int:
        return _keyed_ping(
            f"{live_runtime.url}/ag-ui", {"authorization": f"Bearer {key}"}
        )

    env = {"COGNIVERSE_WEB_HARNESS_KEY_TTL_S": str(KEY_TTL_S)}
    openai_compat.set_key_resolver(recording)
    try:
        with (
            web_server_process(
                web_client_dir,
                live_runtime.url,
                telemetry_url=telemetry_sink[0],
                env=env,
            ) as survivor,
            web_server_process(
                web_client_dir,
                live_runtime.url,
                telemetry_url=telemetry_sink[0],
                env=env,
            ) as doomed,
        ):
            through = {"x-cogniverse-tenant": tenant}
            doomed_minted = time.monotonic()
            assert _keyed_ping(f"{doomed.url}/ui-api/runtime/ag-ui", through) == 400
            [doomed_key] = taken()
            survivor_minted = time.monotonic()
            assert _keyed_ping(f"{survivor.url}/ui-api/runtime/ag-ui", through) == 400
            renewable = time.monotonic() + KEY_TTL_S / 2 + 0.3
            [first_key] = taken()
            doomed.kill()
            assert listed() == {
                digest(doomed_key): (False, KEY_TTL_S),
                digest(first_key): (False, KEY_TTL_S),
            }

            # Past half the ttl the survivor mints a replacement; the key it
            # replaces is neither revoked nor expired yet.
            _wait_until(renewable)
            assert _keyed_ping(f"{survivor.url}/ui-api/runtime/ag-ui", through) == 400
            [second_key] = taken()
            assert second_key != first_key
            assert at_runtime(first_key) == 400
            assert time.monotonic() < survivor_minted + KEY_TTL_S, (
                "the checks before the first key's expiry ran past it"
            )
            assert listed() == {
                digest(doomed_key): (False, KEY_TTL_S),
                digest(first_key): (False, KEY_TTL_S),
                digest(second_key): (False, KEY_TTL_S),
            }

            # Once their ttl has passed, the killed server's key and the
            # superseded one are refused and listed as revoked.
            _wait_until(max(doomed_minted, survivor_minted) + KEY_TTL_S + 0.3)
            assert [at_runtime(doomed_key), at_runtime(first_key)] == [401, 401]
            assert listed() == {
                digest(doomed_key): (True, KEY_TTL_S),
                digest(first_key): (True, KEY_TTL_S),
                digest(second_key): (False, KEY_TTL_S),
            }

            presented.clear()
            ran = _drive(
                web_client_dir,
                survivor.url,
                SCENARIO="sequence",
                QUERIES="still here",
                TENANT=tenant,
            )
            assert ran == [
                {"reply": [f"[{tenant}] Two clips match: still here"], "error": None}
            ]
            # The run went with the survivor's current key, never one that
            # expired.
            used = taken()
            live = {h for h, (revoked, _) in listed().items() if not revoked}
            assert (used != [], {digest(key) for key in used} - live) == (True, set())
            assert {doomed_key, first_key} & set(used) == set()
    finally:
        openai_compat.set_key_resolver(live_runtime.keys.resolve)
    # The survivor revoked the keys it still held when it stopped.
    assert _live_keys(live_runtime, tenant) == []


def test_runtime_auth_unavailable_fails_the_run_with_the_reason(
    web_client_dir, live_runtime, telemetry_sink
):
    """While the runtime cannot issue a key, a run fails naming why; a run
    that names no tenant is refused; once keys issue again runs go through."""
    refused = admin._harness_key_store_unavailable(
        ConfigStoreUnavailableError("config store did not answer")
    )
    with InterceptFaultProxy(live_runtime.url) as proxy:
        proxy.intercept = lambda method, path, body: (
            (refused.status_code, {"detail": refused.detail})
            if (method, path) == ("POST", "/admin/harness/keys")
            else None
        )
        with serve_web(
            web_client_dir, proxy.url, telemetry_url=telemetry_sink[0]
        ) as web_url:
            down = _drive(web_client_dir, web_url, SCENARIO="fault")
            anonymous = _drive(web_client_dir, web_url, SCENARIO="fault", TENANT="")
            proxy.intercept = None
            recovered = _drive(
                web_client_dir, web_url, SCENARIO="sequence", QUERIES="back"
            )
    reason = (
        f"The runtime did not issue a harness key for tenant {TENANT} "
        f"(HTTP 503: {refused.detail['message']})."
    )
    assert [down["events"], anonymous["events"]] == [
        [
            {
                "type": "RUN_ERROR",
                "code": "INCOMPLETE_STREAM",
                "message": "HTTP 503: "
                + json.dumps({"error": reason}, separators=(",", ":")),
            }
        ],
        [
            {
                "type": "RUN_ERROR",
                "code": "INCOMPLETE_STREAM",
                "message": "HTTP 400: "
                + json.dumps(
                    {"error": "Choose a tenant before talking to an agent."},
                    separators=(",", ":"),
                ),
            }
        ],
    ]
    assert recovered == [
        {"reply": [f"[{TENANT}] Two clips match: back"], "error": None}
    ]


def test_a_down_runtime_fails_the_run_and_the_agent_list(
    web_client_dir, telemetry_sink
):
    """Neither the Dots nor a run read as empty when the runtime is down:
    both fail naming the runtime that did not answer."""
    dead_runtime = f"http://127.0.0.1:{free_port()}"
    with serve_web(
        web_client_dir, dead_runtime, telemetry_url=telemetry_sink[0]
    ) as web_url:
        result = _drive(web_client_dir, web_url, SCENARIO="fault")
    reason = f"The Cogniverse runtime at {dead_runtime} did not answer (TypeError)."
    assert result == {
        "events": [],
        "error": "HTTP 500: "
        + json.dumps(
            {"error": "Failed to run agent", "message": reason}, separators=(",", ":")
        ),
        "listedStatus": 502,
        "listed": {"error": reason},
    }


def test_the_server_announces_itself_only_once_it_is_listening(
    web_client_dir, telemetry_sink
):
    """Callers connect as soon as the server prints its listening line, so the
    line is printed only once the port is bound: a server that cannot bind
    exits with the bind error and never prints it."""
    node = shutil.which("node")
    assert node is not None, "node is required to run the web client"
    taken = socket.create_server(("127.0.0.1", 0))
    port = taken.getsockname()[1]
    try:
        run = subprocess.run(
            [node, "--import", "tsx", "src/server/index.ts"],
            cwd=web_client_dir,
            env=node_env(
                node,
                COGNIVERSE_RUNTIME_URL=f"http://127.0.0.1:{free_port()}",
                PORT=str(port),
                COPILOTKIT_TELEMETRY_URL=telemetry_sink[0],
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        taken.close()
    assert (run.returncode, run.stdout) == (1, "")
    assert (
        f"Error: listen EADDRINUSE: address already in use 127.0.0.1:{port}\n"
        in run.stderr
    ), run.stderr
    assert telemetry_sink[1] == []


def test_the_server_stops_after_a_grace_period_with_a_request_in_flight(
    web_client_dir, telemetry_sink
):
    """A request the runtime never answers is cut once the shutdown grace
    period (5 s) has passed."""
    hung = socket.create_server(("127.0.0.1", 0))
    accepted = []
    acceptor = threading.Thread(
        target=lambda: accepted.append(hung.accept()), daemon=True
    )
    acceptor.start()
    outcome = []
    try:
        with serve_web(
            web_client_dir,
            f"http://127.0.0.1:{hung.getsockname()[1]}",
            telemetry_url=telemetry_sink[0],
        ) as web_url:
            connection = http.client.HTTPConnection(
                web_url.removeprefix("http://"), timeout=30
            )

            def call():
                try:
                    connection.request("GET", "/ui-api/runtime/agents/")
                    outcome.append(connection.getresponse().status)
                except (http.client.HTTPException, OSError) as exc:
                    outcome.append(type(exc).__name__)

            caller = threading.Thread(target=call, daemon=True)
            caller.start()
            acceptor.join(timeout=20)
            assert len(accepted) == 1, "the request never reached the runtime"
            stopping = time.monotonic()
        stopped_after = time.monotonic() - stopping
        caller.join(timeout=10)
    finally:
        for connection_, _ in accepted:
            connection_.close()
        hung.close()
    assert 5 <= stopped_after < 8
    assert outcome == ["RemoteDisconnected"]
