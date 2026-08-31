"""Unit tests for the DSPy span lookup diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import phoenix.client as phoenix_client
import pytest

import tests.e2e.test_inbound_dspy_span_e2e as e2e


@dataclass
class _FakePhoenixSpans:
    chain_frame: pd.DataFrame
    lm_frame: pd.DataFrame
    chain_error: Exception | None = None
    lm_error: Exception | None = None
    chain_calls: int = 0
    lm_calls: int = 0

    def get_spans_dataframe(self, **kwargs):
        condition = getattr(getattr(kwargs["query"], "_filter", None), "condition", "")
        if "ChainOfThought.forward" in condition:
            self.chain_calls += 1
            if self.chain_error is not None:
                raise self.chain_error
            return self.chain_frame
        if "LM.__call__" in condition:
            self.lm_calls += 1
            if self.lm_error is not None:
                raise self.lm_error
            return self.lm_frame
        if self.lm_error is not None:
            raise self.lm_error
        raise AssertionError(f"unexpected Phoenix query: {kwargs['query']!r}")


class _FakePhoenixClient:
    def __init__(self, base_url: str, spans: _FakePhoenixSpans):
        self.base_url = base_url
        self.spans = spans


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    spans: _FakePhoenixSpans,
    text: str = "needle",
    timeout_s: float = 1.0,
) -> str:
    clock = {"now": 0.0}

    def fake_time() -> float:
        return clock["now"]

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    def fake_client(base_url: str) -> _FakePhoenixClient:
        return _FakePhoenixClient(base_url, spans)

    monkeypatch.setattr(e2e.time, "time", fake_time)
    monkeypatch.setattr(e2e.time, "sleep", fake_sleep)
    monkeypatch.setattr(phoenix_client, "Client", fake_client)

    with pytest.raises(AssertionError) as excinfo:
        e2e._query_dspy_lm_spans_with_text(text, timeout_s=timeout_s)
    return str(excinfo.value)


def test_query_dspy_lm_spans_reports_three_distinct_failure_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = {
        "phoenix": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame([]),
                lm_frame=_frame([]),
                chain_error=TimeoutError("phoenix busy"),
            ),
        ),
        "chain": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame(
                    [
                        {
                            "attributes.input.value": "other query one",
                            "context.trace_id": "trace-a",
                            "context.span_id": "span-a",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "other query two",
                            "context.trace_id": "trace-b",
                            "context.span_id": "span-b",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
                lm_frame=_frame([]),
            ),
        ),
        "lm": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame(
                    [
                        {
                            "attributes.input.value": "needle in the first chain row",
                            "context.trace_id": "trace-c",
                            "context.span_id": "span-c",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "another chain row",
                            "context.trace_id": "trace-d",
                            "context.span_id": "span-d",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
                lm_frame=_frame(
                    [
                        {
                            "attributes.input.value": "lm child prompt one",
                            "context.trace_id": "trace-c",
                            "context.span_id": "lm-span-1",
                            "name": "LM.__call__",
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "lm child prompt two",
                            "context.trace_id": "trace-c",
                            "context.span_id": "lm-span-2",
                            "name": "LM.__call__",
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
            ),
        ),
    }

    assert len(set(messages.values())) == 3
    assert (
        "Phoenix error while querying ChainOfThought.forward spans"
        in messages["phoenix"]
    )
    assert "TimeoutError: phoenix busy" in messages["phoenix"]
    assert "chain_span_count=2" in messages["chain"]
    assert "matching_chain_count=0" in messages["chain"]
    assert "trace_ids=['trace-c']" in messages["lm"]
    assert "lm_child_count=2" in messages["lm"]
    assert "matching_lm_count=0" in messages["lm"]
