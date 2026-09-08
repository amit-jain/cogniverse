"""The OpenAI-compatible /v1 surface over the real dispatcher.

Real FastAPI router -> real AgentDispatcher -> real AgentRegistry ->
deterministic agents registered through the generic ``AGENT_CLASSES`` path.
No boundary is mocked: the key store is a real ``HarnessKeyStore`` over a real
``ConfigStore``, the outage cases use a real ``VespaConfigStore`` pointed at a
dead port, and the client-disconnect case runs against a real uvicorn socket.

No LM is involved: every agent here answers from its input, so the wire shape,
the tenant boundary, the message grammar and the cancellation behaviour are
pinned exactly. A turn that reaches a live LM is exercised elsewhere; this
module makes no claim about it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.harness_keys import HarnessKeyStore
from cogniverse_runtime.harness_turn import derive_request_seed
from cogniverse_runtime.routers import openai_compat
from cogniverse_vespa.config.config_store import (
    _CONFIG_STORE_DOCUMENT_READ_TIMEOUT_SECONDS as CONFIG_STORE_DOCUMENT_TIMEOUT,
)
from cogniverse_vespa.config.config_store import (
    _CONFIG_STORE_READ_MAX_ATTEMPTS as CONFIG_STORE_READ_MAX_ATTEMPTS,
)
from cogniverse_vespa.config.config_store import (
    _config_store_visit_backoff_seconds as _read_backoff_seconds,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT_A_RAW = "acme"
TENANT_A = "acme:acme"
TENANT_B = "beta:prod"
KEY_A = "harness-key-tenant-a"
KEY_B = "harness-key-tenant-b"

QUERY = "Summarize the Eiffel Tower article in one sentence."
IMAGE_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
TOOL_CALL_ID = "call_fixed_write_file"
DEAD_BACKEND_PORT = 29071
# What the config store's own retry structure allows a single-document
# read to take: one per-attempt timeout per attempt, plus the capped
# backoff between them.
KEY_READ_BUDGET_SECONDS = CONFIG_STORE_READ_MAX_ATTEMPTS * (
    CONFIG_STORE_DOCUMENT_TIMEOUT
) + sum(
    _read_backoff_seconds(attempt)
    for attempt in range(1, CONFIG_STORE_READ_MAX_ATTEMPTS)
)

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the client workspace",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]


class HarnessEchoDeps(BaseModel):
    """The one Deps shape for this module.

    ``AgentDispatcher`` resolves the generic Deps/Input classes by scanning the
    agent's module for the first name ending in ``Deps``/``Input``, so every
    agent registered from one module shares them; each declares its own Output.
    """


class HarnessEchoInput(BaseModel):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    attachments: list = []
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    request_id: str = ""
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class ContextEchoOutput(BaseModel):
    answer: str = ""


class ContextEchoAgent:
    """Answers with the dispatch context it was handed, as sorted JSON.

    Everything the router is supposed to place on the context is therefore
    pinned by one string equality on the completion's content.
    """

    def __init__(self, deps: HarnessEchoDeps):
        self.deps = deps

    async def process(self, input: HarnessEchoInput) -> ContextEchoOutput:
        return ContextEchoOutput(
            answer=json.dumps(
                {
                    "query": input.query,
                    "tenant_id": input.tenant_id,
                    "history": input.conversation_history,
                    "attachments": input.attachments,
                    "temperature": input.temperature,
                    "max_tokens": input.max_tokens,
                    "request_id": input.request_id,
                    "tool_names": [
                        tool["function"]["name"] for tool in input.external_tools
                    ],
                },
                sort_keys=True,
            )
        )


class ToolEchoOutput(BaseModel):
    status: str = "success"
    answer: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class ToolEchoAgent:
    """Deterministic dual-loop agent with a fixed call id.

    The fixed id lets a second tenant replay the first tenant's exact
    transcript, which is what the continuation cache must not honour.
    """

    def __init__(self, deps: HarnessEchoDeps):
        self.deps = deps

    async def process(self, input: HarnessEchoInput) -> ToolEchoOutput:
        if not input.tool_results:
            return ToolEchoOutput(
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
        return ToolEchoOutput(
            answer=json.dumps(
                {
                    "tenant_id": input.tenant_id,
                    "resumed_state": input.continuation_state,
                    "rounds": len(input.tool_exchange),
                    "results": [r["content"] for r in input.tool_results],
                },
                sort_keys=True,
            )
        )


slow_agent_events: List[str] = []


class SlowEchoOutput(BaseModel):
    answer: str = ""


class SlowEchoAgent:
    """A 20 s turn that records whether it was cancelled."""

    def __init__(self, deps: HarnessEchoDeps):
        self.deps = deps

    async def process(self, input: HarnessEchoInput) -> SlowEchoOutput:
        slow_agent_events.append("started")
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            slow_agent_events.append("cancelled")
            raise
        slow_agent_events.append("completed")
        return SlowEchoOutput(answer="slow answer")


class FailingEchoOutput(BaseModel):
    answer: str = ""


class FailingEchoAgent:
    def __init__(self, deps: HarnessEchoDeps):
        self.deps = deps

    async def process(self, input: HarnessEchoInput) -> FailingEchoOutput:
        raise RuntimeError(f"boom-{input.query}")


_AGENT_CLASSES = {
    "context_echo_agent": f"{__name__}:ContextEchoAgent",
    "tool_echo_agent": f"{__name__}:ToolEchoAgent",
    "slow_echo_agent": f"{__name__}:SlowEchoAgent",
    "failing_echo_agent": f"{__name__}:FailingEchoAgent",
}

MODEL_MAP = {
    "cogniverse": "context_echo_agent",
    "cogniverse/tools": "tool_echo_agent",
    "cogniverse/slow": "slow_echo_agent",
    "cogniverse/failing": "failing_echo_agent",
}


@pytest.fixture(scope="module")
def dispatcher():
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    for agent_name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=agent_name,
                url="http://localhost:8000",
                capabilities=[agent_name.removesuffix("_agent")],
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    yield AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    return app


@pytest.fixture()
def compat_app(dispatcher):
    """The real router mounted at /v1, wired to the real dispatcher.

    Module-level DI is reset after each test so nothing leaks between them.
    """
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A_RAW, KEY_B: TENANT_B})
    openai_compat.set_model_map(MODEL_MAP)
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()
    slow_agent_events.clear()
    yield _build_app()
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()


@pytest.fixture()
async def client(compat_app):
    transport = httpx.ASGITransport(app=compat_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=60.0
    ) as http_client:
        yield http_client


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _body(model: str = "cogniverse", **overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": QUERY}],
        "stream": False,
    }
    body.update(overrides)
    return body


def _content(response: httpx.Response) -> Dict[str, Any]:
    payload = response.json()
    return json.loads(payload["choices"][0]["message"]["content"])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestTenantBoundary:
    """4a — one canonicalization, and an outage that reads as an outage."""

    def test_static_map_tenant_is_canonicalized_once(self):
        openai_compat.set_api_keys({KEY_A: "acme", KEY_B: "beta:prod"})
        try:
            assert openai_compat.resolve_tenant(KEY_A) == "acme:acme"
            assert openai_compat.resolve_tenant(KEY_B) == "beta:prod"
            assert openai_compat.resolve_tenant("no-such-key") is None
        finally:
            openai_compat.set_api_keys({})

    async def test_dispatch_context_carries_the_canonical_tenant(self, client):
        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(KEY_A)
        )

        assert response.status_code == 200
        assert _content(response) == {
            "query": QUERY,
            "tenant_id": TENANT_A,
            "history": [],
            "attachments": [],
            "temperature": None,
            "max_tokens": None,
            "request_id": derive_request_seed(QUERY, []),
            "tool_names": [],
        }

    async def test_unknown_key_is_401_with_the_openai_error_body(self, client):
        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth("not-a-key")
        )

        assert response.status_code == 401
        assert response.json() == {
            "error": {
                "message": "Invalid or missing API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }
        assert response.headers["WWW-Authenticate"] == "Bearer"

    async def test_missing_bearer_header_is_401(self, client):
        response = await client.post("/v1/chat/completions", json=_body())

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"

    async def test_revoked_store_key_is_401_not_503(self, client):
        store = InMemoryConfigStore()
        store.initialize()
        key_store = HarnessKeyStore(store)
        minted = key_store.create(TENANT_B, "revoked-key")
        key_store.revoke(minted["key_hash"])
        openai_compat.set_key_resolver(key_store.resolve)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(minted["key"])
        )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"

    async def test_store_key_reaches_dispatch_canonicalized(self, client):
        store = InMemoryConfigStore()
        store.initialize()
        key_store = HarnessKeyStore(store)
        minted = key_store.create(TENANT_A_RAW, "live-key")
        openai_compat.set_key_resolver(key_store.resolve)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(minted["key"])
        )

        assert response.status_code == 200
        assert _content(response)["tenant_id"] == TENANT_A

    async def test_key_store_outage_is_503_naming_the_dead_backend(self, client):
        from cogniverse_vespa.config.config_store import VespaConfigStore

        dead_store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=DEAD_BACKEND_PORT
        )
        openai_compat.set_key_resolver(HarnessKeyStore(dead_store).resolve)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth("any-unknown-key")
        )

        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "service_unavailable"
        assert error["type"] == "server_error"
        assert error["message"].startswith(
            "Harness key store unavailable: Failed to read Vespa config document "
            f"after {CONFIG_STORE_READ_MAX_ATTEMPTS} attempts over "
        )
        assert f"port={DEAD_BACKEND_PORT}" in error["message"]
        assert "Connection refused" in error["message"]

    async def test_a_hung_key_store_is_503_not_a_hang(self, client):
        """A backend that accepts and never answers still ends as a 503.

        This is what a paused container looks like on the wire: the socket
        connects, the request is never answered. The read must still end in
        the outage contract rather than holding the request open forever.
        """
        from cogniverse_vespa.config.config_store import VespaConfigStore

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        accepted: List[socket.socket] = []

        def accept_forever():
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                accepted.append(connection)

        accepter = threading.Thread(target=accept_forever, daemon=True)
        accepter.start()
        try:
            hung_store = VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=listener.getsockname()[1]
            )
            openai_compat.set_key_resolver(HarnessKeyStore(hung_store).resolve)

            started = time.perf_counter()
            response = await client.post(
                "/v1/chat/completions",
                json=_body(),
                headers=_auth("any-unknown-key"),
            )
            elapsed = time.perf_counter() - started
        finally:
            listener.close()
            for connection in accepted:
                connection.close()
            accepter.join(timeout=5)

        assert response.status_code == 503, f"after {elapsed:.1f}s"
        assert (
            KEY_READ_BUDGET_SECONDS
            <= elapsed
            < (KEY_READ_BUDGET_SECONDS + CONFIG_STORE_DOCUMENT_TIMEOUT)
        ), (
            f"hung key-store read took {elapsed:.1f}s; the store's retry "
            f"structure spends {KEY_READ_BUDGET_SECONDS:.2f}s and must finish "
            f"within one further {CONFIG_STORE_DOCUMENT_TIMEOUT}s attempt"
        )
        error = response.json()["error"]
        assert error["code"] == "service_unavailable"
        assert error["message"].startswith("Harness key store unavailable: ")

    async def test_models_route_reports_the_outage_as_503(self, client):
        from cogniverse_vespa.config.config_store import VespaConfigStore

        dead_store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=DEAD_BACKEND_PORT
        )
        openai_compat.set_key_resolver(HarnessKeyStore(dead_store).resolve)

        response = await client.get("/v1/models", headers=_auth("any-unknown-key"))

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"

    async def test_models_route_lists_the_configured_map_in_order(self, client):
        response = await client.get("/v1/models", headers=_auth(KEY_A))

        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        assert [entry["id"] for entry in payload["data"]] == list(MODEL_MAP)
        assert {entry["owned_by"] for entry in payload["data"]} == {"cogniverse"}

    async def test_unwired_dispatcher_provider_is_503(self, client):
        openai_compat.set_dispatcher_provider(None)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(KEY_A)
        )

        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "message": "Runtime initialising; dispatcher not wired.",
                "type": "server_error",
                "code": "service_unavailable",
            }
        }

    async def test_provider_returning_none_is_503(self, client):
        openai_compat.set_dispatcher_provider(lambda: None)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(KEY_A)
        )

        assert response.status_code == 503
        assert response.json()["error"]["message"] == (
            "Runtime initialising; dispatcher not built yet."
        )

    async def test_provider_raising_is_503_carrying_the_cause(self, client):
        def _boom():
            raise RuntimeError("AgentDispatcher not initialized")

        openai_compat.set_dispatcher_provider(_boom)

        response = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(KEY_A)
        )

        assert response.status_code == 503
        assert response.json()["error"]["message"] == (
            "Dispatcher unavailable: AgentDispatcher not initialized"
        )

    async def test_unknown_model_is_404(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/nope"),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 404
        assert response.json()["error"] == {
            "message": (
                "Model 'cogniverse/nope' does not exist or you do not have "
                "access to it."
            ),
            "type": "invalid_request_error",
            "code": "model_not_found",
        }

    async def test_agent_failure_is_500_carrying_the_cause(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/failing"),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 500
        error = response.json()["error"]
        assert error["code"] == "internal_error"
        assert error["type"] == "server_error"
        assert f"boom-{QUERY}" in error["message"]


class TestContinuationIsolation:
    """R3 — a suspended turn belongs to one tenant and one agent."""

    def test_key_separates_tenant_and_agent(self):
        openai_compat.clear_continuations()
        openai_compat.put_continuation(
            TENANT_A, "tool_echo_agent", "seed", ["c1"], {"plan": "a"}
        )

        assert (
            openai_compat.pop_continuation(TENANT_B, "tool_echo_agent", "seed", ["c1"])
            is None
        )
        assert (
            openai_compat.pop_continuation(TENANT_A, "other_agent", "seed", ["c1"])
            is None
        )
        assert openai_compat.continuation_count() == 1
        assert openai_compat.pop_continuation(
            TENANT_A, "tool_echo_agent", "seed", ["c1"]
        ) == {"plan": "a"}
        assert openai_compat.continuation_count() == 0

    async def test_replay_by_another_tenant_misses_and_leaves_the_owner_intact(
        self, client
    ):
        suspend = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
        )
        assert suspend.status_code == 200
        suspended = suspend.json()["choices"][0]
        assert suspended["finish_reason"] == "tool_calls"
        assert suspended["message"]["tool_calls"] == [
            {
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"text": QUERY}),
                },
            }
        ]
        assert openai_compat.continuation_count() == 1

        resume_messages = [
            {"role": "user", "content": QUERY},
            {
                "role": "assistant",
                "tool_calls": suspended["message"]["tool_calls"],
            },
            {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "wrote it"},
        ]

        thief = await client.post(
            "/v1/chat/completions",
            json=_body(
                model="cogniverse/tools", tools=TOOL_DEFS, messages=resume_messages
            ),
            headers=_auth(KEY_B),
        )
        assert thief.status_code == 200
        assert _content(thief) == {
            "tenant_id": TENANT_B,
            "resumed_state": {},
            "rounds": 1,
            "results": ["wrote it"],
        }

        owner = await client.post(
            "/v1/chat/completions",
            json=_body(
                model="cogniverse/tools", tools=TOOL_DEFS, messages=resume_messages
            ),
            headers=_auth(KEY_A),
        )
        assert owner.status_code == 200
        assert _content(owner) == {
            "tenant_id": TENANT_A,
            "resumed_state": {"plan": f"plan-for-{TENANT_A}"},
            "rounds": 1,
            "results": ["wrote it"],
        }
        assert openai_compat.continuation_count() == 0

    async def test_two_spellings_of_one_tenant_share_one_namespace(self, client):
        """The router canonicalizes the key's tenant before it keys anything.

        Two static entries naming the same tenant in the simple and the
        ``org:tenant`` form must resume each other's suspended turn; without
        canonicalization at the key they are two namespaces.
        """
        openai_compat.set_api_keys({KEY_A: TENANT_A_RAW, "colon-spelled-key": TENANT_A})

        suspend = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
        )
        assert suspend.status_code == 200
        tool_calls = suspend.json()["choices"][0]["message"]["tool_calls"]

        resume = await client.post(
            "/v1/chat/completions",
            json=_body(
                model="cogniverse/tools",
                tools=TOOL_DEFS,
                messages=[
                    {"role": "user", "content": QUERY},
                    {"role": "assistant", "tool_calls": tool_calls},
                    {
                        "role": "tool",
                        "tool_call_id": TOOL_CALL_ID,
                        "content": "wrote it",
                    },
                ],
            ),
            headers=_auth("colon-spelled-key"),
        )

        assert resume.status_code == 200
        assert _content(resume) == {
            "tenant_id": TENANT_A,
            "resumed_state": {"plan": f"plan-for-{TENANT_A}"},
            "rounds": 1,
            "results": ["wrote it"],
        }


class TestMessageGrammar:
    """4b — every Class D input from the audit, pinned."""

    async def test_image_only_user_message_is_accepted(self, client):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": IMAGE_URI}}],
            }
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        assert _content(response) == {
            "query": "",
            "tenant_id": TENANT_A,
            "history": [],
            "attachments": [IMAGE_URI],
            "temperature": None,
            "max_tokens": None,
            "request_id": derive_request_seed("", []),
            "tool_names": [],
        }

    async def test_caption_plus_image_keeps_both(self, client):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": IMAGE_URI},
                ],
            }
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        body = _content(response)
        assert body["query"] == "what is this?"
        assert body["attachments"] == [IMAGE_URI]

    @pytest.mark.parametrize(
        "text_value,type_name",
        [(None, "NoneType"), (123, "int"), (["a", "b"], "list"), ({}, "dict")],
    )
    async def test_non_string_text_part_is_400_naming_its_position(
        self, client, text_value, type_name
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "keep"},
                    {"type": "text", "text": text_value},
                ],
            }
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": (
                f"messages[0] part 1 has text of type {type_name}; "
                "text parts require a string"
            ),
            "type": "invalid_request_error",
            "code": "invalid_request",
        }

    async def test_unsupported_part_type_is_400_naming_its_position(self, client):
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "x"}}],
            }
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages[0] part 0 has unsupported type 'input_audio'; only text "
            "and image parts are accepted"
        )

    async def test_image_part_without_a_url_is_400(self, client):
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages[0] part 0 is an image part with no url string; expected "
            "image_url as a string or as an object with a 'url' string"
        )

    async def test_content_of_the_wrong_shape_is_400(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=[{"role": "user", "content": 42}]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages[0] content must be a string or an array of parts, got int"
        )

    async def test_transcript_without_a_user_message_is_400(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=[{"role": "system", "content": "be nice"}]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages must contain at least one user message"
        )

    async def test_empty_user_turn_is_400(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=[{"role": "user", "content": "   "}]),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages[0] is the turn's user message and carries neither text "
            "nor an image part"
        )

    @pytest.mark.parametrize(
        "tool_calls,expected",
        [
            (
                "write_file",
                "messages[1] must carry tool_calls as a non-empty array, got str",
            ),
            (
                {"id": "c1"},
                "messages[1] must carry tool_calls as a non-empty array, got dict",
            ),
            (
                [],
                "messages[1] must carry tool_calls as a non-empty array, got list",
            ),
            (
                ["c1"],
                "messages[1] tool_calls[0] must be an object, got str",
            ),
            (
                [{"function": {"name": "write_file"}}],
                "messages[1] tool_calls[0] requires a non-empty string id",
            ),
            (
                [{"id": "c1"}],
                "messages[1] tool_calls[0] requires a function object, got NoneType",
            ),
            (
                [{"id": "c1", "function": {"arguments": "{}"}}],
                "messages[1] tool_calls[0] requires a non-empty string function.name",
            ),
            (
                [{"id": "c1", "function": {"name": "w", "arguments": {"a": 1}}}],
                "messages[1] tool_calls[0] function.arguments must be a JSON "
                "string, got dict",
            ),
        ],
    )
    async def test_malformed_tool_calls_are_400(self, client, tool_calls, expected):
        messages = [
            {"role": "user", "content": QUERY},
            {"role": "assistant", "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == expected

    async def test_tool_arguments_that_are_not_json_are_400(self, client):
        messages = [
            {"role": "user", "content": QUERY},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "w", "arguments": "{oops"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"].startswith(
            "messages[1] tool_calls[0] function.arguments is not valid JSON: "
        )

    def _round(self, calls, results):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": QUERY}]
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": call, "function": {"name": "write_file"}} for call in calls
                ],
            }
        )
        messages.extend(
            {"role": "tool", "tool_call_id": result, "content": f"out-{result}"}
            for result in results
        )
        return messages

    @pytest.mark.parametrize(
        "calls,results,expected",
        [
            (
                ["c1"],
                ["c2"],
                "tool round 0 has a tool result for unmatched id 'c2'; the "
                "round called ['c1']",
            ),
            (
                ["c1", "c1"],
                ["c1"],
                "tool round 0 repeats tool_call id(s) ['c1']",
            ),
            (
                ["c1"],
                ["c1", "c1"],
                "tool round 0 repeats a tool result for id 'c1'",
            ),
            (
                ["c1", "c2"],
                ["c1"],
                "tool round 0 has no tool result for id(s) ['c2']",
            ),
        ],
    )
    async def test_unmatched_tool_results_are_400_naming_the_ids(
        self, client, calls, results, expected
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", messages=self._round(calls, results)),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == expected

    async def test_a_call_id_reused_across_rounds_is_400(self, client):
        messages = [
            {"role": "user", "content": QUERY},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "write_file"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "write_file"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "two"},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "tool_call id(s) ['c1'] reused across rounds"
        )

    async def test_tool_result_before_any_assistant_call_is_400(self, client):
        messages = [
            {"role": "user", "content": QUERY},
            {"role": "tool", "tool_call_id": "c1", "content": "orphan"},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(model="cogniverse/tools", messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "messages[1] is a tool result but no assistant tool_calls message "
            "precedes it in this turn"
        )

    async def test_history_precedes_the_last_user_message(self, client):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": QUERY},
        ]

        response = await client.post(
            "/v1/chat/completions",
            json=_body(messages=messages),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        body = _content(response)
        assert body["query"] == QUERY
        assert body["history"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        assert body["request_id"] == derive_request_seed(
            QUERY, [{"role": "user", "content": "first question"}]
        )


class TestSamplingParameters:
    """R23 — the knobs the client sets reach the dispatch context."""

    async def test_temperature_and_max_tokens_reach_the_context(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(temperature=0.25, max_tokens=321),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        body = _content(response)
        assert body["temperature"] == 0.25
        assert body["max_tokens"] == 321

    async def test_max_completion_tokens_is_the_same_ceiling(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(max_completion_tokens=64),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        assert _content(response)["max_tokens"] == 64

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"temperature": 2.5}, "temperature must be between 0 and 2.0, got 2.5"),
            (
                {"temperature": -0.1},
                "temperature must be between 0 and 2.0, got -0.1",
            ),
            ({"max_tokens": 0}, "max_tokens must be at least 1, got 0"),
            (
                {"max_completion_tokens": -5},
                "max_completion_tokens must be at least 1, got -5",
            ),
            (
                {"max_tokens": 10, "max_completion_tokens": 20},
                "max_tokens and max_completion_tokens disagree: "
                "{'max_tokens': 10, 'max_completion_tokens': 20}",
            ),
        ],
    )
    async def test_out_of_range_sampling_is_400(self, client, overrides, expected):
        response = await client.post(
            "/v1/chat/completions", json=_body(**overrides), headers=_auth(KEY_A)
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == expected

    async def test_agreeing_ceilings_are_accepted_once(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json=_body(max_tokens=99, max_completion_tokens=99),
            headers=_auth(KEY_A),
        )

        assert response.status_code == 200
        assert _content(response)["max_tokens"] == 99


class TestStreamedChunking:
    """The stream path chunks the finished answer; concatenation is identity."""

    def test_split_is_a_partition_of_the_text(self):
        text = "".join(f"{index:04d}-" for index in range(400))

        parts = openai_compat.split_answer_chunks(text)

        assert len(parts) == 8
        assert [len(part) for part in parts] == [256] * 7 + [208]
        assert "".join(parts) == text

    async def test_streamed_turn_reassembles_to_the_non_streamed_answer(self, client):
        plain = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(KEY_A)
        )
        assert plain.status_code == 200
        expected = plain.json()["choices"][0]["message"]["content"]

        streamed = await client.post(
            "/v1/chat/completions", json=_body(stream=True), headers=_auth(KEY_A)
        )

        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")
        lines = [
            line[len("data: ") :]
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert lines[-1] == "[DONE]"
        chunks = [json.loads(line) for line in lines[:-1]]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert (
            "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            == expected
        )
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["total_tokens"] == (
            chunks[-1]["usage"]["prompt_tokens"]
            + chunks[-1]["usage"]["completion_tokens"]
        )
        assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}


class TestEventLoopIsNotStalled:
    """The key read is a blocking backend call; it must leave the loop free."""

    async def test_twenty_concurrent_turns_keep_the_loop_responsive(self, client):
        class SlowReadStore(InMemoryConfigStore):
            """A real store whose reads take as long as a backend round-trip."""

            def get_immutable_config(self, *args, **kwargs):
                time.sleep(0.03)
                return super().get_immutable_config(*args, **kwargs)

        store = SlowReadStore()
        store.initialize()
        key_store = HarnessKeyStore(store)
        minted = key_store.create(TENANT_A_RAW, "loop-gap-key")
        openai_compat.set_key_resolver(key_store.resolve)

        warmup = await client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(minted["key"])
        )
        assert warmup.status_code == 200

        gaps: List[float] = []

        async def ticker():
            previous = time.perf_counter()
            while True:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - previous)
                previous = now

        tick = asyncio.create_task(ticker())
        try:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json=_body(),
                        headers=_auth(minted["key"]),
                    )
                    for _ in range(20)
                )
            )
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)

        assert [response.status_code for response in responses] == [200] * 20
        assert {_content(response)["tenant_id"] for response in responses} == {TENANT_A}
        assert max(gaps) < 0.05, f"max loop gap {max(gaps):.4f}s over {len(gaps)} ticks"


class TestClientDisconnectCancelsTheTurn:
    """R7 — a hang-up on a non-streamed turn stops the work behind it."""

    @pytest.fixture()
    def live_server(self, compat_app):
        port = _free_port()
        config = uvicorn.Config(
            compat_app, host="127.0.0.1", port=port, log_level="warning"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=20)
        assert not thread.is_alive()

    def test_hangup_cancels_the_in_flight_turn(self, live_server):
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                f"{live_server}/v1/chat/completions",
                json=_body(model="cogniverse/slow"),
                headers=_auth(KEY_A),
                timeout=1.5,
            )

        hangup = time.perf_counter()
        deadline = hangup + 2.0
        while time.perf_counter() < deadline and "cancelled" not in slow_agent_events:
            time.sleep(0.02)
        observed = time.perf_counter() - hangup

        assert slow_agent_events == ["started", "cancelled"], (
            f"agent events {slow_agent_events} after {observed:.3f}s"
        )
        assert observed < 2.0
        assert openai_compat.in_flight_count() == 0

    def test_a_completed_turn_still_answers_on_the_same_socket(self, live_server):
        response = httpx.post(
            f"{live_server}/v1/chat/completions",
            json=_body(),
            headers=_auth(KEY_A),
            timeout=30.0,
        )

        assert response.status_code == 200
        assert (
            json.loads(response.json()["choices"][0]["message"]["content"])["tenant_id"]
            == TENANT_A
        )
        assert openai_compat.in_flight_count() == 0
