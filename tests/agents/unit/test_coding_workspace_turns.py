"""Workspace decisions preserve tool identity and bounded failure semantics."""

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path

import dspy
import pytest

from cogniverse_agents import coding_agent as coding


def _memory_config_manager():
    """The injected ConfigManager the agent constructor requires."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a workspace file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def request(**kwargs):
    return coding.CodingInput(
        task="read x",
        tenant_id="workspace:workspace",
        external_tools=TOOLS,
        continuation_state={"plan": "read plan", "step": 0},
        **kwargs,
    )


def envelope(**kwargs):
    return {
        "plan": "read plan",
        "code_changes": [],
        "execution_results": [],
        "summary": "",
        "iterations_used": 0,
        "files_modified": [],
        "rlm_synthesis": None,
        "rlm_telemetry": None,
        "pending_tool_calls": [],
        "continuation_state": {},
        "success": True,
        "error": None,
        **kwargs,
    }


def controlled_agent(monkeypatch, decisions):
    agent = coding.CodingAgent(
        coding.CodingDeps(tenant_id="workspace:workspace"),
        config_manager=_memory_config_manager(),
    )
    calls = []

    async def decide(module, **kwargs):
        calls.append(kwargs)
        decision = decisions[min(len(calls) - 1, len(decisions) - 1)]
        if isinstance(decision, BaseException):
            raise decision
        return dspy.Prediction(**decision)

    monkeypatch.setattr(agent, "call_dspy", decide)
    return agent, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "reason"),
    [
        ("unknown", "{}", "unknown tool 'unknown'"),
        ("read_file", "{broken", "arguments for 'read_file' must be valid JSON"),
        ("read_file", "{}", "missing required argument 'path' for 'read_file'"),
        ("read_file", "[]", "arguments for 'read_file' must be a JSON object"),
    ],
)
async def test_malformed_action_is_failed_after_exact_retry_budget(
    monkeypatch, name, arguments, reason
):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": name, "tool_args_json": arguments, "summary": "read the file"}],
    )
    result = await agent._process_workspace(request())
    assert result.model_dump() == envelope(
        success=False,
        iterations_used=3,
        error=f"Invalid workspace action after 3 attempt(s): {reason}",
    )
    assert len(calls) == 3
    assert [call["remaining_steps"] for call in calls] == [8, 7, 6]
    assert (
        calls[1]["observations"]
        == f"no workspace actions taken yet\nstep 1 failed: {reason}"
    )


@pytest.mark.asyncio
async def test_invalid_action_retries_then_suspends_exact_tool(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [
            {"tool_name": "unknown", "tool_args_json": "{}", "summary": "not complete"},
            {
                "tool_name": "read_file",
                "tool_args_json": '{"path":"x.py"}',
                "summary": "read file",
            },
        ],
    )
    monkeypatch.setattr(coding.uuid, "uuid4", lambda: uuid.UUID(int=1))
    result = await agent._process_impl(request())
    assert result.model_dump() == envelope(
        pending_tool_calls=[
            {
                "id": "call_000000000000",
                "name": "read_file",
                "arguments": {"path": "x.py"},
            }
        ],
        continuation_state={"mode": "workspace", "plan": "read plan", "step": 2},
        iterations_used=2,
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_finish_is_the_only_completed_action(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [
            {
                "tool_name": "finish",
                "tool_args_json": "{}",
                "summary": "x.py contains 42",
            }
        ],
    )
    result = await agent._process_workspace(request())
    assert result.model_dump() == envelope(
        summary="x.py contains 42", iterations_used=1
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_exported_cap_stops_without_another_decision(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "wrong"}],
    )
    inp = request(max_iterations=100)
    inp.continuation_state["step"] = 8
    result = await agent._process_workspace(inp)
    assert coding.WORKSPACE_MAX_ROUNDS == 8
    assert result.model_dump() == envelope(
        success=False, iterations_used=8, error="Workspace round limit reached (8)"
    )
    assert calls == []


@pytest.mark.asyncio
async def test_retry_cannot_exceed_remaining_rounds(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "unknown", "tool_args_json": "{}", "summary": "wrong"}],
    )
    inp = request(max_iterations=100)
    inp.continuation_state["step"] = 7
    result = await agent._process_workspace(inp)
    assert result.model_dump() == envelope(
        success=False,
        iterations_used=8,
        error="Invalid workspace action after 1 attempt(s): unknown tool 'unknown'",
    )
    assert len(calls) == 1


def replay_round():
    return {
        "tool_calls": [
            {
                "id": "read",
                "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
            },
            {
                "id": "write",
                "function": {"name": "write_file", "arguments": '{"path":"y.py"}'},
            },
        ],
        "results": [
            {"tool_call_id": "write", "content": "wrote y.py"},
            {"tool_call_id": "read", "content": "x.py is 42"},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [True, False])
async def test_replayed_results_are_attributed_by_id(monkeypatch, history):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "read and wrote"}],
    )
    exchange = replay_round()
    inp = request(
        **(
            {"tool_exchange": [exchange]}
            if history
            else {
                "assistant_tool_calls": exchange["tool_calls"],
                "tool_results": exchange["results"],
            }
        )
    )
    result = await agent._process_workspace(inp)
    assert (
        calls[0]["observations"]
        == 'step 1: read_file({"path":"x.py"}) -> x.py is 42\nstep 1: write_file({"path":"y.py"}) -> wrote y.py'
    )
    assert result.model_dump() == envelope(summary="read and wrote", iterations_used=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("duplicate_result", "Duplicate tool result id 'write'"),
        ("unmatched_result", "Unmatched tool result id 'foreign'"),
        ("missing_result", "Missing tool results for: read"),
        ("duplicate_call", "Duplicate tool call id 'read'"),
    ],
)
async def test_invalid_replay_is_rejected_before_decision(monkeypatch, fault, error):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "wrong"}],
    )
    exchange = replay_round()
    if fault == "duplicate_result":
        exchange["results"].append(exchange["results"][0])
    elif fault == "unmatched_result":
        exchange["results"][0]["tool_call_id"] = "foreign"
    elif fault == "missing_result":
        exchange["results"].pop()
    else:
        exchange["tool_calls"].append(exchange["tool_calls"][0])
    with pytest.raises(ValueError, match=f"^{error}$"):
        await agent._process_workspace(request(tool_exchange=[exchange]))
    assert calls == []


@pytest.mark.asyncio
async def test_workspace_memory_reads_run_off_loop_with_tenant_isolation(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "done"}],
    )
    entered = threading.Barrier(3, timeout=3)
    release = threading.Event()
    seen = []

    def memory_read(prompt, query):
        tenant = agent._current_memory_tenant_id()
        entered.wait()
        if not release.wait(timeout=3):
            raise RuntimeError("event loop did not release memory reads")
        return f"{tenant}: {prompt}"

    monkeypatch.setattr(agent, "inject_context_into_prompt", memory_read)
    a, b = request(), request()
    a.tenant_id, b.tenant_id = "one:one", "two:two"
    tasks = [asyncio.create_task(agent._process_workspace(inp)) for inp in (a, b)]
    try:
        await asyncio.to_thread(entered.wait)
        seen.append("loop advanced while both memory reads blocked")
    finally:
        release.set()
    results = await asyncio.gather(*tasks)
    assert seen == ["loop advanced while both memory reads blocked"]
    assert sorted(call["task"] for call in calls) == [
        "one:one: read x",
        "two:two: read x",
    ]
    assert [result.summary for result in results] == ["done", "done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_staging_directory_removed_when_generation_fails(
    monkeypatch, tmp_path, cancel
):
    agent = coding.CodingAgent(
        coding.CodingDeps(tenant_id="workspace:workspace"),
        config_manager=_memory_config_manager(),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def staging(**kwargs):
        path = real_mkdtemp(dir=tmp_path, **kwargs)
        created.append(path)
        return path

    async def decision(module, **kwargs):
        if kwargs["output_field"] == "plan":
            return dspy.Prediction(plan="read plan")
        started.set()
        await release.wait()
        raise RuntimeError("generation failed after staging creation")

    monkeypatch.setattr(coding.tempfile, "mkdtemp", staging)
    monkeypatch.setattr(agent, "call_dspy", decision)
    before = sorted(tmp_path.iterdir())
    task = asyncio.create_task(
        agent._process_impl(
            coding.CodingInput(task="write x", tenant_id="workspace:workspace")
        )
    )
    await started.wait()
    assert len(created) == 1
    assert [Path(path).is_dir() for path in created] == [True]
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        with pytest.raises(
            RuntimeError, match="^generation failed after staging creation$"
        ):
            await task
    assert sorted(tmp_path.iterdir()) == before


@pytest.mark.asyncio
async def test_workspace_lm_connection_failure_is_not_completion():
    agent = coding.CodingAgent(
        coding.CodingDeps(tenant_id="workspace:workspace"),
        config_manager=_memory_config_manager(),
    )
    lm = dspy.LM(
        "openai/unavailable",
        api_base="http://127.0.0.1:29071/v1",
        api_key="unused",
        num_retries=0,
        timeout=1,
        cache=False,
    )
    with dspy.context(lm=lm), pytest.raises(Exception, match="Connection error"):
        await agent._process_workspace(request())


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_limit", [None, 2])
async def test_workspace_default_budget_uses_exported_cap(monkeypatch, explicit_limit):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "last step"}],
    )
    inp = request(**({"max_iterations": explicit_limit} if explicit_limit else {}))
    limit = explicit_limit or 8
    inp.continuation_state["step"] = limit - 1
    result = await agent._process_workspace(inp)
    assert result.model_dump() == envelope(summary="last step", iterations_used=limit)
    assert [call["remaining_steps"] for call in calls] == [1]
    inp.continuation_state["step"] = limit
    stopped = await agent._process_workspace(inp)
    assert stopped.model_dump() == envelope(
        success=False,
        iterations_used=limit,
        error=f"Workspace round limit reached ({limit})",
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_negative_workspace_step_is_rejected(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "wrong"}],
    )
    inp = request()
    inp.continuation_state["step"] = -5
    with pytest.raises(
        ValueError, match="^Workspace step must be a non-negative integer$"
    ):
        await agent._process_workspace(inp)
    assert calls == []


@pytest.mark.asyncio
async def test_workspace_step_cannot_undercount_replayed_rounds(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "wrong"}],
    )
    inp = request(max_iterations=2, tool_exchange=[replay_round(), replay_round()])
    result = await agent._process_workspace(inp)
    assert result.model_dump() == envelope(
        success=False, iterations_used=2, error="Workspace round limit reached (2)"
    )
    assert calls == []


@pytest.mark.asyncio
async def test_latest_results_reordered_do_not_duplicate_recorded_round(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [
            {
                "tool_name": "finish",
                "tool_args_json": "{}",
                "summary": "one round replayed",
            }
        ],
    )
    exchange = replay_round()
    inp = request(
        tool_exchange=[exchange],
        assistant_tool_calls=list(reversed(exchange["tool_calls"])),
        tool_results=list(reversed(exchange["results"])),
    )
    result = await agent._process_workspace(inp)
    assert (
        calls[0]["observations"]
        == 'step 1: read_file({"path":"x.py"}) -> x.py is 42\nstep 1: write_file({"path":"y.py"}) -> wrote y.py'
    )
    assert result.model_dump() == envelope(
        summary="one round replayed", iterations_used=2
    )


@pytest.mark.asyncio
async def test_latest_results_must_agree_with_recorded_round(monkeypatch):
    agent, calls = controlled_agent(
        monkeypatch,
        [{"tool_name": "finish", "tool_args_json": "{}", "summary": "wrong"}],
    )
    exchange = replay_round()
    inp = request(
        tool_exchange=[exchange],
        assistant_tool_calls=exchange["tool_calls"],
        tool_results=[
            {"tool_call_id": "read", "content": "tampered"},
            exchange["results"][0],
        ],
    )
    with pytest.raises(
        ValueError, match="^Latest tool replay conflicts with recorded round$"
    ):
        await agent._process_workspace(inp)
    assert calls == []


def test_sandbox_output_shape_matches_e2e_contract():
    from tests.e2e.test_coding_cli_e2e import _assert_coding_output_shape

    result = coding.CodingOutput().model_dump()
    _assert_coding_output_shape(result)
    for field in result:
        missing = dict(result)
        del missing[field]
        with pytest.raises(AssertionError):
            _assert_coding_output_shape(missing)
    with pytest.raises(AssertionError):
        _assert_coding_output_shape({**result, "unexpected": True})
    for field, value in {
        "pending_tool_calls": [{"id": "unexpected"}],
        "continuation_state": {"step": 1},
        "success": False,
        "error": "unexpected failure",
    }.items():
        with pytest.raises(AssertionError):
            _assert_coding_output_shape({**result, field: value})
