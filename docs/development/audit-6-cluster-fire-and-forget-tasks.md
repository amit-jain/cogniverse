# Audit Cycle 6 — Cluster: fire-and-forget asyncio task references

Review summary for the Class-C "fire-and-forget `create_task` with a dropped
reference" hunt. CPython holds only a weak reference to a running task; if no
strong reference is kept, the task may be garbage-collected before its
coroutine completes, silently dropping the work.

## Repo-wide sweep (`grep -rnP "\bcreate_task\(" libs/`)

| Site | Status |
|------|--------|
| `runtime/openshell_health.py:127` | safe — `self._task = ...` |
| `runtime/openshell_cert_rotator.py:256` | safe — `self._task = ...` |
| `core/events/backends/memory.py:206` | safe — `self._cleanup_task = ...` |
| `runtime/agent_dispatcher.py:584` | safe — `_background_tasks` set + `add_done_callback` |
| `core/memory/lifecycle_scheduler.py:123` | safe — `self._task = ...` |
| `core/agents/base.py:589` | safe — `task` is a live local held through the generator body |
| `telemetry-phoenix/.../evaluation_provider.py:379` | safe — `_background_tasks` set + `add_done_callback` |
| **`agents/inference/instrumented_rlm.py:127`** | **BUG — result of `loop.create_task(...)` discarded** |

Only one genuine offender; the rest already retain a strong reference.

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `agents/inference/instrumented_rlm.py:127` (`_emit_sync`) | C | `loop.create_task(self._event_queue.enqueue(event))` discards the task → the RLM progress/event enqueue can be GC'd before it runs, silently dropping the event from the SSE stream | hold the task in `self._background_tasks` (set added in `__init__`) and `add_done_callback(self._background_tasks.discard)` to release on completion — same pattern as `agent_dispatcher` / `evaluation_provider` |

## Test (`tests/agents/unit/test_instrumented_rlm.py::TestEmitSyncRetainsTask`)

Real `InstrumentedRLM` + a recording async queue:

- after `_emit_sync(...)`, the task is retained (`len(_background_tasks) == 1`);
- after awaiting it, the event was delivered (`delivered == [{"e": 1}]`) and the
  reference was released (`_background_tasks == set()`).

Fails on pre-fix code (`_background_tasks` does not exist → AttributeError).
