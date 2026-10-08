"""The web client's server drives Cogniverse agents through CopilotKit.

The pinned ``clients/web`` lockfile is installed and its Node server runs from
source against the runtime's ``/ag-ui`` and ``/agents`` routers on a real
uvicorn socket. A driver uses the published ``@ag-ui/client`` the browser
bundle uses, POSTing runs to the CopilotKit runtime the server hosts, so each
run crosses every hop a browser run crosses: CopilotKit runtime -> the
server's ``HttpAgent`` with the harness key -> the AG-UI router -> the real
dispatcher -> the agent.

The driver parses the final state with the client's own ``resultsOf``, so the
result cards are pinned against what the runtime actually sends. A recorder
stands in for CopilotKit's telemetry sink and must receive nothing. Node and
npm are required; their absence is a failure, not a skip.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

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
from cogniverse_runtime.routers import ag_ui, agents, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.memory_store import InMemoryConfigStore
from tests.utils.node_env import node_env
from tests.utils.web_client import (
    free_port,
    install_web_client,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]


TENANT = "acme:web"
KEY = "web-client-harness-key"
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

async function run(agentId, messages, tools = [], thread = agentId) {
  const agent = new HttpAgent({
    agentId,
    url: `${WEB}/api/copilotkit/agent/${agentId}/run`,
    threadId: `thread-${thread}`,
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
  const queries = Array.from({ length: Number(process.env.RUNS) }, (_, i) => `query ${i}`);
  const outcomes = await Promise.all(
    queries.map((query, i) =>
      run('search_agent', [{ id: 'u1', role: 'user', content: query }], [], `c${i}`),
    ),
  );
  console.log(JSON.stringify(outcomes.map(reply)));
  process.exit(0);
}

if (process.env.SCENARIO === 'fault') {
  const events = [];
  let error = null;
  try {
    const agent = new HttpAgent({
      agentId: 'search_agent',
      url: `${WEB}/api/copilotkit/agent/search_agent/run`,
    });
    agent.setMessages([{ id: 'u1', role: 'user', content: process.env.QUERY }]);
    await agent.runAgent({}, { onEvent: ({ event }) => events.push(event) });
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  const listed = await fetch(`${WEB}/api/agents`);
  console.log(
    JSON.stringify({ events, error, listedStatus: listed.status, listed: await listed.json() }),
  );
  process.exit(0);
}

const listed = await (await fetch(`${WEB}/api/agents`)).json();
const info = await (await fetch(`${WEB}/api/copilotkit/info`)).json();

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


_AGENT_CLASSES = {
    "search_agent": f"{__name__}:WebSearchAgent",
    "tool_agent": f"{__name__}:WebToolAgent",
}
_TOKEN_STREAMING = {"search_agent": True, "tool_agent": False}


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
    openai_compat.set_api_keys({KEY: TENANT})
    openai_compat.set_key_resolver(None)

    with serve_app(app) as url:
        yield url

    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
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
        web_client_dir, live_runtime, KEY, telemetry_url=telemetry_sink[0]
    ) as url:
        yield url


def _drive(client_dir: Path, web_url: str, **scenario: str):
    """Run the driver against the web server and return its JSON report."""
    node = shutil.which("node")
    proc = subprocess.run(
        [node, "--import", "tsx", str(client_dir / "driver.ts")],
        cwd=client_dir,
        capture_output=True,
        text=True,
        timeout=180,
        env=node_env(node, WEB_URL=web_url, QUERY=QUERY, **scenario),
    )
    assert proc.returncode == 0, (
        f"web driver failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_browser_run_reaches_the_agent_through_copilotkit(
    web_client_dir, web_server, telemetry_sink
):
    result = _drive(web_client_dir, web_server, TOOL_OUTPUT=TOOL_OUTPUT)

    # The Dots come from the runtime's registry, in registry order, and the
    # CopilotKit runtime serves exactly those agents.
    assert result["listed"] == {"agents": ["search_agent", "tool_agent"]}
    assert result["infoAgents"] == ["search_agent", "tool_agent"]

    search = result["search"]
    summary = f"[{TENANT}] Two clips match: {QUERY}"
    assert search["events"] == [
        "RUN_STARTED",
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
            "value": {"phase": STATUS_PHASE, "message": STATUS_MESSAGE},
        }
    ]
    # The harness key the server holds resolved to the tenant.
    assert search["reply"] == [summary]
    assert search["agent"] == "search_agent"
    assert search["cards"] == [
        {
            "id": "v7_seg_3",
            "ratingId": "id:video:video::v7_seg_3",
            "score": 0.91,
            "title": "Tower at night",
            "snippet": "the tower lights up",
            "start": 42.0,
            "end": 48.5,
        },
        {
            "id": "v2_seg_0",
            "ratingId": "id:video:video::v2_seg_0",
            "score": 0.64,
            "snippet": "a skyline at dusk",
        },
    ]

    suspended = result["suspended"]
    assert suspended["events"] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
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

    # CopilotKit's own usage reporting stays off in a self-hosted deployment.
    assert telemetry_sink[1] == []


def test_concurrent_runs_each_get_their_own_reply(web_client_dir, web_server):
    """Runs in flight together through one server, each on its own thread,
    come back with their own query and nothing of another run's."""
    runs = 6
    replies = _drive(web_client_dir, web_server, SCENARIO="concurrent", RUNS=str(runs))
    assert replies == [[f"[{TENANT}] Two clips match: query {i}"] for i in range(runs)]


def test_a_rejected_harness_key_fails_the_run_with_the_reason(
    web_client_dir, live_runtime, telemetry_sink
):
    """The runtime's 401 reaches the browser as the run's error, word for
    word; the unauthenticated agent list still answers."""
    with serve_web(
        web_client_dir,
        live_runtime,
        "not-a-harness-key",
        telemetry_url=telemetry_sink[0],
    ) as web_url:
        result = _drive(web_client_dir, web_url, SCENARIO="fault")
    assert result == {
        "events": [
            {
                "type": "RUN_ERROR",
                "code": "INCOMPLETE_STREAM",
                "message": "HTTP 401: "
                + json.dumps(
                    {
                        "error": {
                            "message": openai_compat.UNAUTHORIZED["message"],
                            "type": "invalid_request_error",
                            "code": openai_compat.UNAUTHORIZED["code"],
                        }
                    },
                    separators=(",", ":"),
                ),
            }
        ],
        "error": None,
        "listedStatus": 200,
        "listed": {"agents": ["search_agent", "tool_agent"]},
    }


def test_a_down_runtime_fails_the_run_and_the_agent_list(
    web_client_dir, telemetry_sink
):
    """Neither the Dots nor a run read as empty when the runtime is down:
    both fail naming the runtime that did not answer."""
    dead_runtime = f"http://127.0.0.1:{free_port()}"
    with serve_web(
        web_client_dir, dead_runtime, KEY, telemetry_url=telemetry_sink[0]
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
                COGNIVERSE_API_KEY=KEY,
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
            KEY,
            telemetry_url=telemetry_sink[0],
        ) as web_url:
            connection = http.client.HTTPConnection(
                web_url.removeprefix("http://"), timeout=30
            )

            def call():
                try:
                    connection.request("GET", "/api/runtime/agents/")
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
