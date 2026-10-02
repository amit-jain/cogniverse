"""The output shape of a coding agent turn."""

from __future__ import annotations


def _assert_coding_output_shape(result):
    assert set(result) == {
        "plan",
        "code_changes",
        "execution_results",
        "summary",
        "iterations_used",
        "files_modified",
        "rlm_synthesis",
        "rlm_telemetry",
        "pending_tool_calls",
        "continuation_state",
        "success",
        "error",
    }, result
    assert result["pending_tool_calls"] == []
    assert result["continuation_state"] == {}
    assert result["success"] is True
    assert result["error"] is None
