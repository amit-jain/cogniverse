# Audit Cycle 6 — Cluster: a2a error-text test was hollow

| Site | Class | Problem | Fix |
|------|-------|---------|-----|
| `tests/runtime/unit/test_a2a_server.py::test_message_send_error_returns_error_text` | HOLLOW-TEST | docstring claims "executor returns error as text message" but only asserted `result is not None` — a generic/empty success would pass | parse the response message part and assert the embedded error payload: `status=="error"`, `agent=="bad_agent"`, and the verbatim dispatch error in `error` |

Test-only. The executor surfaces `str(e)` as a JSON text message
(`a2a_executor.py:156-162`); the test now verifies that contract.
