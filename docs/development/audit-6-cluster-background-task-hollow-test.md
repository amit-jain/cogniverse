# Audit Cycle 6 — Cluster: background-task tests rebuilt the pattern inline

| Site | Class | Problem | Fix |
|------|-------|---------|-----|
| `tests/runtime/unit/test_dispatcher_background_tasks.py` | HOLLOW-TEST | the tests re-created the `create_task` + `_background_tasks.add` + `add_done_callback` dance **inline in the test body**, proving asyncio works rather than that `AgentDispatcher` / `PhoenixEvaluationProvider` actually retain their fire-and-forget tasks | extracted `_spawn_background()` on both `AgentDispatcher` and `PhoenixEvaluationProvider` (DRYs the dispatch + annotation call sites), and rewrote the tests to drive the real method |

## Verification

- 5 tests pass driving the real `_spawn_background`.
- Mutation: removing `self._background_tasks.add(task)` from
  `AgentDispatcher._spawn_background` makes the strong-reference test fail
  (it passed vacuously before — the inline copy was self-contained).
