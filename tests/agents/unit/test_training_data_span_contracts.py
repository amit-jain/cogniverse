"""Contract test for training-data telemetry span emitters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cogniverse_foundation.telemetry import config as telemetry_config

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

ROOT = Path(__file__).resolve().parents[3]
OPTIMIZATION_CLI = ROOT / "libs/runtime/cogniverse_runtime/optimization_cli.py"
ORCHESTRATION_EVALUATOR = (
    ROOT / "libs/agents/cogniverse_agents/routing/orchestration_evaluator.py"
)
AGENT_ROOT = ROOT / "libs/agents/cogniverse_agents"


def _source_segment(text: str, node: ast.AST) -> str:
    lines = text.splitlines(keepends=True)
    end_lineno = getattr(node, "end_lineno", None) or node.lineno
    return "".join(lines[node.lineno - 1 : end_lineno])


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _training_span_constant_names_from_cli() -> set[str]:
    text = OPTIMIZATION_CLI.read_text()
    module = ast.parse(text, filename=str(OPTIMIZATION_CLI))
    names: set[str] = set()

    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source = _source_segment(text, node)
        if "_query_spans_by_name(" not in source:
            continue
        if not any(
            marker in source
            for marker in (
                "ArtifactManager(",
                "teleprompter.compile(",
                "WorkflowIntelligence(",
            )
        ):
            continue

        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_query_spans_by_name"
            ):
                continue
            span_node = None
            if len(call.args) >= 4:
                span_node = call.args[3]
            else:
                for kw in call.keywords:
                    if kw.arg == "span_name":
                        span_node = kw.value
                        break
            if isinstance(span_node, ast.Name) and span_node.id.startswith(
                "SPAN_NAME_"
            ):
                names.add(span_node.id)
            elif isinstance(span_node, ast.Attribute) and span_node.attr.startswith(
                "SPAN_NAME_"
            ):
                names.add(span_node.attr)

    return names


def _training_span_constant_names_from_orchestration_reader() -> set[str]:
    text = ORCHESTRATION_EVALUATOR.read_text()
    module = ast.parse(text, filename=str(ORCHESTRATION_EVALUATOR))
    names: set[str] = set()

    for call in ast.walk(module):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get_all_spans"
        ):
            continue

        for kw in call.keywords:
            if kw.arg != "filters" or not isinstance(kw.value, ast.Dict):
                continue
            for key_node, value_node in zip(kw.value.keys, kw.value.values):
                if _string_constant(key_node) != "name":
                    continue
                if isinstance(value_node, ast.Name) and value_node.id.startswith(
                    "SPAN_NAME_"
                ):
                    names.add(value_node.id)
                elif isinstance(
                    value_node, ast.Attribute
                ) and value_node.attr.startswith("SPAN_NAME_"):
                    names.add(value_node.attr)

    return names


def _training_span_names() -> list[str]:
    constant_names = (
        _training_span_constant_names_from_cli()
        | _training_span_constant_names_from_orchestration_reader()
    )
    return [getattr(telemetry_config, name) for name in sorted(constant_names)]


def _call_uses_span_name(call: ast.Call, span_name: str) -> bool:
    for arg in call.args:
        if _string_constant(arg) == span_name:
            return True
    for kw in call.keywords:
        if kw.arg == "name" and _string_constant(kw.value) == span_name:
            return True
    return False


def _find_emitter(span_name: str) -> tuple[Path, ast.AST, str]:
    for path in sorted(AGENT_ROOT.rglob("*.py")):
        if path.name.startswith("test_") or path.name == "__init__.py":
            continue
        text = path.read_text()
        if span_name not in text or "telemetry_manager.span" not in text:
            continue

        module = ast.parse(text, filename=str(path))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            source = _source_segment(text, node)
            if (
                span_name not in source
                or "telemetry_manager.span" not in source
                or "record_span_io(" not in source
            ):
                continue
            return path, node, source

    raise AssertionError(f"Could not locate an emitter for {span_name}")


def _logger_calls(node: ast.AST, level: str) -> list[ast.Call]:
    return [
        inner
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == level
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id == "logger"
    ]


def _first_string_arg(call: ast.Call) -> str:
    return _string_constant(call.args[0]) or "" if call.args else ""


def _assert_emitter_contract(
    span_name: str, path: Path, node: ast.AST, source: str
) -> None:
    where = f"{span_name} emitter {path}:{node.lineno}"

    assert isinstance(node, ast.AsyncFunctionDef), (
        f"{where} must be `async def` and be awaited inline on the request path"
    )

    span_calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "span"
        and _call_uses_span_name(call, span_name)
    ]
    assert span_calls, f"{where} does not call span()"

    for call in span_calls:
        assert not any(kw.arg == "require_export" for kw in call.keywords), (
            f"{where} passes require_export: synchronous export is not allowed "
            "on the request path; spans go on the batch queue"
        )
    assert not any(
        isinstance(inner, ast.Attribute) and inner.attr == "required_span"
        for inner in ast.walk(node)
    ), (
        f"{where} uses required_span: synchronous export is not allowed on the request path"
    )

    assert not any(
        isinstance(inner, ast.Raise)
        and isinstance(inner.exc, ast.Call)
        and isinstance(inner.exc.func, ast.Name)
        and inner.exc.func.id == "RuntimeError"
        for inner in ast.walk(node)
    ), f"{where} raises RuntimeError: telemetry loss must never fail a request"

    assert not _logger_calls(node, "debug"), (
        f"{where} swallows a telemetry failure at logger.debug; loss must be a WARNING"
    )

    warnings = [_first_string_arg(call) for call in _logger_calls(node, "warning")]
    assert any("has no telemetry_manager" in message for message in warnings), (
        f"{where} must WARN (not return silently) when no telemetry_manager is attached"
    )
    assert any(message.startswith("Failed to emit ") for message in warnings), (
        f"{where} must WARN (not raise, not swallow) when the span cannot be enqueued"
    )


def _assert_call_site_awaits_inline(
    span_name: str, path: Path, emitter_name: str
) -> None:
    module = ast.parse(path.read_text(), filename=str(path))
    awaited_inline = any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == emitter_name
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "self"
        for node in ast.walk(module)
    )
    assert awaited_inline, (
        f"{span_name} emitter {emitter_name} in {path} is not awaited inline "
        "(`await self.<emitter>(...)`)"
    )
    threaded = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        and any(
            isinstance(arg, ast.Attribute) and arg.attr == emitter_name
            for arg in node.args
        )
    ]
    assert not threaded, (
        f"{span_name} emitter {emitter_name} in {path} is handed to to_thread at "
        f"line {threaded[0].lineno}: enqueue is non-blocking, a thread hop only "
        "adds a context switch and a pool slot"
    )


def test_training_data_span_emitters_are_async_queued_and_warn_on_loss() -> None:
    """Every training-data span emitter is async, enqueue-only, and warns on loss."""
    span_names = _training_span_names()
    assert span_names, "Optimizer readers did not yield any training-data span names"

    for span_name in span_names:
        path, node, source = _find_emitter(span_name)
        _assert_emitter_contract(span_name, path, node, source)
        _assert_call_site_awaits_inline(span_name, path, node.name)
