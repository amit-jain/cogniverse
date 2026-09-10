"""Pi's own agent loop completes a tool round trip against /v1.

The harness face is Pi, and Pi speaks its own ``openai-completions``
implementation rather than the ``openai`` SDK. This module runs the real
published packages the pinned ``clients/pi-cogniverse`` lockfile resolves
(``@earendil-works/pi-ai`` + ``pi-agent-core``) against the /v1 router on a
real uvicorn socket: Pi streams the completion, parses the suspended
tool_calls turn, executes a local ``write_file`` tool in a scratch
workspace, replays the transcript with the result and receives the answer.

No LM: the agent answers from the turn's own input, so the final text pins
the whole loop — the tool result Pi sent back, the tenant, the replayed
history length and the continuation fast path. Node and npm are required;
their absence is a failure, not a skip.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from cogniverse_core.agents.base import ConfigManagerAware
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CLIENT_DIR = REPO_ROOT / "clients" / "pi-cogniverse"
# The pi packages declare engines.node >= 22.19.0.
MIN_NODE_MAJOR = 22

TENANT = "test:unit"
KEY = "pi-loop-key"
MODEL = "cogniverse/pi-echo"
TOOL_CALL_ID = "call_pi_write_file"
PROMPT = "write the report"
TOOL_OUTPUT = "file written ok"
SYSTEM_PROMPT = "You drive cogniverse."

# One deterministic string for the whole round trip: the tool result Pi
# replayed, the tenant the key resolved to, the one history entry Pi sends
# (its system prompt), and the continuation state popped on resume.
FINAL_ANSWER = f"tool said: {TOOL_OUTPUT}; tenant={TENANT}; history=1; resumed=True"

DRIVER_MJS = """
import {
  createModels,
  createProvider,
  envApiKeyAuth,
  Type,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { Agent } from "@earendil-works/pi-agent-core";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const BASE = process.env.COGNIVERSE_BASE_URL;
const WORKSPACE = process.env.PI_WORKSPACE;

const model = {
  id: process.env.PI_MODEL,
  name: "Cogniverse",
  api: "openai-completions",
  provider: "cogniverse",
  baseUrl: BASE,
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 16384,
  maxTokens: 4096,
};

const provider = createProvider({
  id: "cogniverse",
  auth: { apiKey: envApiKeyAuth("API Key", ["COGNIVERSE_API_KEY"]) },
  models: [model],
  baseUrl: BASE,
  api: openAICompletionsApi(),
});
const models = createModels();
models.setProvider(provider);

const events = [];
const writeTool = {
  name: "write_file",
  label: "Write File",
  description: "Write text to a file in the workspace",
  parameters: Type.Object({ text: Type.String() }),
  execute: async (_toolCallId, params) => {
    await writeFile(path.join(WORKSPACE, "note.txt"), params.text, "utf8");
    return { content: [{ type: "text", text: process.env.PI_TOOL_OUTPUT }], details: {} };
  },
};

const agent = new Agent({
  initialState: {
    systemPrompt: process.env.PI_SYSTEM_PROMPT,
    model,
    tools: [writeTool],
    messages: [],
  },
  streamFn: (m, ctx, opts) => models.streamSimple(m, ctx, opts),
});
agent.subscribe((event) => {
  if (event.type === "tool_execution_start") {
    events.push(`${event.type}:${event.toolName ?? event.toolCall?.name ?? ""}`);
  } else {
    events.push(event.type);
  }
});

await agent.prompt(process.env.PI_PROMPT);
await agent.waitForIdle();

const messages = agent.state.messages;
const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
const text = (lastAssistant?.content ?? [])
  .filter((b) => b.type === "text")
  .map((b) => b.text)
  .join("");
let note = null;
try {
  note = await readFile(path.join(WORKSPACE, "note.txt"), "utf8");
} catch {}

console.log(
  JSON.stringify({
    text,
    note,
    stopReason: lastAssistant?.stopReason ?? null,
    roles: messages.map((m) => m.role),
    events,
    errorMessage: lastAssistant?.errorMessage ?? null,
  }),
);
"""


class PiEchoDeps(BaseModel):
    """The one Deps shape for this module."""


class PiEchoInput(BaseModel):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class PiEchoOutput(BaseModel):
    status: str = "success"
    answer: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class PiEchoAgent(ConfigManagerAware):
    """Suspends on the client's first advertised tool, then answers from the
    result the client replayed."""

    def __init__(self, deps: PiEchoDeps):
        self.deps = deps

    async def process(self, input: PiEchoInput) -> PiEchoOutput:
        if input.tool_results:
            results = "; ".join(result["content"] for result in input.tool_results)
            return PiEchoOutput(
                answer=(
                    f"tool said: {results}; tenant={input.tenant_id}; "
                    f"history={len(input.conversation_history)}; "
                    f"resumed={bool(input.continuation_state)}"
                )
            )
        return PiEchoOutput(
            status="external_tool_calls",
            pending_tool_calls=[
                {
                    "id": TOOL_CALL_ID,
                    "name": input.external_tools[0]["function"]["name"],
                    "arguments": {"text": input.query},
                }
            ],
            continuation_state={"plan": f"plan-for-{input.tenant_id}"},
        )


_AGENT_CLASSES = {"pi_echo_agent": f"{__name__}:PiEchoAgent"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _node_env(node: str, **extra: str) -> dict:
    """The subprocess environment, named entry by entry.

    Inheriting os.environ would let an ambient COGNIVERSE_BASE_URL or
    COGNIVERSE_API_KEY decide what the driver talks to. HOME is named
    explicitly because npm resolves its cache under it.
    """
    env = {
        # npm runs package install scripts through ``sh``, so the system bin
        # directories belong on PATH alongside the node the test resolved.
        "PATH": f"{Path(node).parent.as_posix()}:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
    }
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def pi_driver_dir(tmp_path_factory):
    """The pinned Pi packages installed from the client's own lockfile.

    The driver is written INSIDE the installed pi-coding-agent package so
    node resolves ``@earendil-works/pi-ai`` and ``pi-agent-core`` from that
    package's nested node_modules, which is where the lockfile places them.
    """
    node = shutil.which("node")
    npm = shutil.which("npm")
    assert node is not None and npm is not None, (
        "node and npm are required to drive the Pi agent loop; the runtime CI "
        "job installs node 22 and both are base tools on a dev host"
    )
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, timeout=30
    ).stdout.strip()
    assert int(version.lstrip("v").split(".")[0]) >= MIN_NODE_MAJOR, (
        f"the pinned pi packages require node >= {MIN_NODE_MAJOR}, got {version}"
    )

    root = tmp_path_factory.mktemp("pi_packages")
    for name in ("package.json", "package-lock.json"):
        shutil.copy(CLIENT_DIR / name, root / name)
    install = subprocess.run(
        [npm, "ci", "--no-fund", "--no-audit"],
        cwd=root,
        env=_node_env(node),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert install.returncode == 0, (
        f"npm ci from the client lockfile failed:\n{install.stdout}\n{install.stderr}"
    )
    driver_dir = root / "node_modules" / "@earendil-works" / "pi-coding-agent"
    assert driver_dir.is_dir(), (
        f"the lockfile did not install pi-coding-agent at {driver_dir}"
    )
    (driver_dir / "driver.mjs").write_text(DRIVER_MJS)
    return driver_dir


@pytest.fixture(scope="module")
def live_v1():
    """The /v1 app with the deterministic agent on a real socket."""
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="pi_echo_agent",
            url="http://localhost:8000",
            capabilities=["pi_echo"],
        )
    )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )

    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY: TENANT})
    openai_compat.set_model_map({MODEL: "pi_echo_agent"})
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"

    yield f"http://127.0.0.1:{port}/v1"

    server.should_exit = True
    thread.join(timeout=20)
    assert not thread.is_alive()
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.clear_continuations()
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


def test_pi_completes_a_tool_round_trip(pi_driver_dir, live_v1, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    node = shutil.which("node")

    proc = subprocess.run(
        [node, str(pi_driver_dir / "driver.mjs")],
        cwd=pi_driver_dir,
        capture_output=True,
        text=True,
        timeout=180,
        env=_node_env(
            node,
            COGNIVERSE_BASE_URL=live_v1,
            COGNIVERSE_API_KEY=KEY,
            PI_MODEL=MODEL,
            PI_PROMPT=PROMPT,
            PI_SYSTEM_PROMPT=SYSTEM_PROMPT,
            PI_TOOL_OUTPUT=TOOL_OUTPUT,
            PI_WORKSPACE=str(workspace),
        ),
    )
    assert proc.returncode == 0, (
        f"pi driver failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["errorMessage"] is None
    assert result["stopReason"] == "stop"

    # Pi ran the local tool with the arguments the turn produced.
    assert (workspace / "note.txt").read_text() == PROMPT
    assert result["note"] == PROMPT

    # Pi's own transcript of the turn.
    assert result["roles"] == ["user", "assistant", "toolResult", "assistant"]
    # Pi's whole event record for the turn. The message_update runs are the
    # SSE content deltas the router emits for each streamed message, so the
    # sequence pins both loops end to end: suspend, local execution, resume.
    assert result["events"] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "tool_execution_start:write_file",
        "tool_execution_end",
        "message_start",
        "message_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]

    assert result["text"] == FINAL_ANSWER
