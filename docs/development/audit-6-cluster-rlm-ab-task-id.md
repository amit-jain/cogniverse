# Audit Cycle 6 — Cluster: RLM A/B arm emitted task-less events

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/inference/ab_harness.py` `_run_with_rlm` | E | constructed `RLMInference(event_queue=..., tenant_id=...)` but never passed `task_id` (documented as "required if event_queue provided"); RLM-arm progress events emitted with `task_id=None` → not routed to the run's SSE stream | forward `self._event_queue.task_id` (EventQueue exposes its own task_id) so RLM events reach the same task stream |

## Test (`tests/agents/unit/test_rlm_ab_harness.py::TestRlmArmEventRouting`)

Patches `RLMInference` to capture its kwargs; asserts `task_id` equals the
event queue's `task_id`. Fails on pre-fix (no task_id passed).
