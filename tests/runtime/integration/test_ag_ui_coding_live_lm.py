"""A coding run with a frontend tool, driven end to end by the real LM.

Real router -> real ``AgentDispatcher`` -> the real ``CodingAgent`` on the
Modal-served student, with the suspended turn kept in the real Redis
continuation store. The client offers one tool, ``write_file``; the agent must
call it (the run suspends on it) and, once the client sends the tool's result,
finish with an answer. A run that ends on an empty-summary error instead is
the failure this guards (the workspace step's blank ``summary`` failed the
parse).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import dspy
import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.dspy import LenientJSONAdapter
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import ag_ui, openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_lm,
    pytest.mark.no_shared_vespa,
    pytest.mark.local_only,
]

TENANT = "agcodelive:main"
KEY = "ag-ui-coding-live-key"
TEXT = "hello from the workspace"
TASK = f"Use the write_file tool to save exactly this text: {TEXT}"
WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write text to the file hello.txt in the user's workspace.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}
TOOL_RESULT = "wrote hello.txt (24 bytes)"


@pytest.fixture
def dispatcher(ensure_host_ollama):
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="coding_agent",
            url="http://localhost:8000",
            capabilities=["coding", "code_generation"],
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    dispatcher._conversation_store_factory = lambda tenant_id: None
    return dispatcher


@pytest.fixture
async def client(dispatcher, continuation_store, conversation_ledger):
    dispatcher.set_conversation_ledger(conversation_ledger)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY: TENANT})
    openai_compat.set_key_resolver(None)
    openai_compat.set_continuation_store(continuation_store)
    app = FastAPI()
    app.include_router(ag_ui.router, prefix="/ag-ui")
    try:
        # The adapter the runtime lifespan binds for every agent.
        with dspy.context(adapter=LenientJSONAdapter()):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
                timeout=600.0,
            ) as http_client:
                yield http_client
    finally:
        dispatcher.set_conversation_ledger(None)
        openai_compat.set_dispatcher_provider(None)
        openai_compat.set_api_keys({})
        openai_compat.set_continuation_store(None)


def _run(messages: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    return {
        "threadId": "coding-live-thread",
        "runId": run_id,
        "state": {},
        "messages": messages,
        "tools": [WRITE_FILE_TOOL],
        "context": [],
        "forwardedProps": {},
    }


def _events(raw: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]


async def test_the_run_calls_the_frontend_tool_then_answers(client):
    user = {"id": "u1", "role": "user", "content": TASK}
    first = await client.post(
        "/ag-ui/coding_agent",
        json=_run([user], "run-1"),
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert first.status_code == 200, first.text
    events = _events(first.text)

    # The run suspends on the tool: its call is the run's last act.
    assert [event["type"] for event in events[-5:]] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "STEP_FINISHED",
        "RUN_FINISHED",
    ], events
    start, args = events[-5], events[-4]
    assert start["toolCallName"] == "write_file"
    assert json.loads(args["delta"]) == {"text": TEXT}
    assert events[-1]["outcome"] == {
        "type": "success",
        "pendingToolCallIds": [start["toolCallId"]],
    }
    assert [event for event in events if event["type"] == "RUN_ERROR"] == []

    resumed = await client.post(
        "/ag-ui/coding_agent",
        json=_run(
            [
                user,
                {
                    "id": start["parentMessageId"],
                    "role": "assistant",
                    "toolCalls": [
                        {
                            "id": start["toolCallId"],
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": args["delta"],
                            },
                        }
                    ],
                },
                {
                    "id": "t1",
                    "role": "tool",
                    "toolCallId": start["toolCallId"],
                    "content": TOOL_RESULT,
                },
            ],
            "run-2",
        ),
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert resumed.status_code == 200, resumed.text
    after = _events(resumed.text)

    # The resumed turn finishes with an answer: no further tool call, no error.
    assert after[-1]["type"] == "RUN_FINISHED", after
    assert after[-1]["outcome"] == {"type": "success"}
    snapshot = next(e for e in after if e["type"] == "STATE_SNAPSHOT")["snapshot"]
    result = snapshot["result"]
    assert result["status"] == "success"
    assert result["result"]["pending_tool_calls"] == []
    assert result["result"]["success"] is True
    reply = "".join(e["delta"] for e in after if e["type"] == "TEXT_MESSAGE_CONTENT")
    summary = result["result"]["summary"]
    assert summary.strip() == summary and summary != ""
    assert reply == summary
