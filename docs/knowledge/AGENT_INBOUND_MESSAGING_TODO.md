# TODO: Per-Session Inbound Messaging for Running Cogniverse Agents

## Problem

Cogniverse's A2A protocol is request/response only. Every `POST /agents/{name}/process` starts a fresh agent invocation; there is no inbound channel into a still-running agent. This matters for long-running flows:

- An `OrchestratorAgent.process()` call can sit 30+ seconds inside `_iterative_retrieval_loop` (multiple LM-backed gate calls + KG expansions + sub-agent dispatches).
- `RLMInference` runs Deno-backed recursive LM iterations, sometimes for minutes on large contexts.
- `DeepResearchAgent` synthesises across multiple documents over a long horizon.

While those flows execute, the caller has no way to:
- Inject additional context ("also factor in this constraint").
- Cancel cleanly with a request to return partial state ("stop, give me what you have so far").
- Steer mid-plan ("the LLM is going down the wrong path — try X instead").
- Resume from a checkpoint with new instructions after a pause.

The closest existing primitives are one-way OUTBOUND: `EventQueue` streams progress events from the running agent to the caller. `CancellationToken` lets the caller raise a hard-cancel signal. Neither carries arbitrary message payloads INBOUND.

## Why this matters

Industry-grade agent platforms (LangGraph human-in-the-loop, AutoGen GroupChat, OpenAI Assistants threads) all expose a "send message to running session" surface. Without it, cogniverse loses the steering / interruption capability that long-horizon agentic workflows need — especially for the iterative retrieve→reason→retrieve patterns we just built for BRIGHT/BrowseComp-class queries.

It's also a real product limit for any UI that wants conversational refinement of an in-flight orchestrator decision instead of cancel-and-restart.

## Existing primitives we'd build on (do not rebuild)

| Primitive | File | Role in the new flow |
|---|---|---|
| `EventQueue` | `libs/runtime/cogniverse_runtime/events.py` | Outbound channel pattern — copy the design (per-session registry, async enqueue / dequeue, drained between iterations) |
| `CancellationToken` | same module | Already in-process; the "stop and return partial" message just sets `cancelled=True` |
| `AgentRegistry` | `libs/core/cogniverse_core/registries/agent_registry.py` | Endpoint lookup. The new HTTP route plugs into the same per-tenant registry |
| Workflow checkpointing | `libs/agents/cogniverse_agents/orchestrator/checkpoint_types.py` | Durable execution between iterations — a message can be persisted to the checkpoint and replayed on resume |
| Mem0 + `KnowledgeRegistry` | `libs/core/cogniverse_core/memory/` | Persistent storage for messages so they survive process death |
| Redis | `cogniverse-redis` k8s service | Multi-pod session-to-pod routing table |
| `agents` router | `libs/runtime/cogniverse_runtime/routers/agents.py` | New `/agents/{name}/message` endpoint lives alongside `/process` |

## Proposed design

### Phase 1 — Single-pod inbound queue

**New module: `libs/runtime/cogniverse_runtime/messaging.py`**

```python
@dataclass
class InboundMessage:
    session_id: str
    role: str          # "user" | "system" | "agent"
    content: str       # free-form payload (the agent decides how to interpret)
    tags: list[str]    # e.g. ["interrupt", "constraint", "stop"]
    created_at: str
    deadline_ms: int | None  # honor by; older messages dropped

class InboundQueueRegistry:
    """In-pod registry of (session_id) -> InboundQueue, mirrors EventQueueRegistry."""
    def create_or_get(self, session_id: str) -> InboundQueue: ...
    def drop(self, session_id: str) -> None: ...
```

**New HTTP route in `routers/agents.py`**

```
POST /agents/{name}/message
{
  "session_id": "...",
  "role": "user",
  "content": "...",
  "tags": ["interrupt"]
}
→ 202 Accepted   (message enqueued, will be drained by running agent at next checkpoint)
→ 404 Not Found  (no such session — agent finished or never started)
```

**Agent loop integration** — three concrete bite-sized integrations:

1. **`_iterative_retrieval_loop`**: drain `inbound_queue` between iterations. Append `interrupt`-tagged messages to `missing_aspects` for the next reformulation. `stop`-tagged messages set the cancellation token, loop exits with `exit_reason="user_stop"`.

2. **`InstrumentedRLM`**: extend the existing `_check_cancelled` callback to also drain inbound messages. New messages get injected as extra context on the next REPL iteration.

3. **`OrchestratorAgent.process`**: pre-loop, register the session in the queue registry. Post-loop, drop it.

### Phase 2 — Multi-pod routing (Redis-backed)

When the runtime scales to multiple pods, the inbound HTTP request lands on an arbitrary pod that may not hold the target session. Add:

- **Session → pod map in Redis**: each agent invocation writes `session:<id> -> pod:<hostname>` with a TTL aligned to the timeout. The HTTP handler looks up the right pod, forwards the message OR (cheaper) proxies via Redis Pub/Sub:
- **Redis Pub/Sub channel** `session-messages:<session_id>` — every pod subscribes; the holder of the session drains; others ignore. Eliminates explicit pod routing.

### Phase 3 — Persistence + replay

Messages persisted via Mem0 with `kind="user_interrupt"` and `tenant_id` provenance. On workflow checkpoint resume (Argo replay after pod restart), the new agent invocation drains the persisted messages first, then continues.

## API surface — what the caller sees

```python
# From a test or a UI:
import httpx
async with httpx.AsyncClient() as client:
    # Start a long-running orchestrator call (non-blocking via task)
    task = asyncio.create_task(
        client.post(
            "https://cogniverse/agents/orchestrator/process",
            json={"query": "...", "session_id": "user-42-q-7", "tenant_id": "acme"},
        )
    )

    # Mid-flight, inject additional context
    await client.post(
        "https://cogniverse/agents/orchestrator/message",
        json={
            "session_id": "user-42-q-7",
            "role": "user",
            "content": "also consider sources from 2024 only",
            "tags": ["constraint"],
        },
    )

    # Or ask it to stop early and return partial
    await client.post(
        "https://cogniverse/agents/orchestrator/message",
        json={"session_id": "user-42-q-7", "tags": ["stop"]},
    )

    result = await task
```

## Test plan (real services, byte-equal where determinism allows)

1. **`test_inbound_queue_basic.py`**: send a message during a synthetic 5-iteration orchestrator loop, assert the message reached the agent and modified the next iteration's reformulation.
2. **`test_inbound_stop_partial.py`**: send `tags=["stop"]` at iter 2 of a 5-iter loop, assert `exit_reason == "user_stop"` and partial evidence returned.
3. **`test_inbound_404_after_completion.py`**: send a message after the agent finished, assert 404.
4. **`test_inbound_multi_pod_routing.py`** (Phase 2): two pods, session held on pod-A, message POSTed to pod-B, assert it arrives.
5. **`test_inbound_checkpoint_replay.py`** (Phase 3): kill pod mid-loop, Argo replays the workflow, assert pre-restart messages re-drain.

## Estimated effort

| Phase | LOC | Wall-clock | Risk |
|---|---|---|---|
| 1 — Single-pod queue + HTTP route + iterative loop integration | ~400 | 1-2 weeks | Low — mirrors EventQueue pattern |
| 2 — Multi-pod via Redis Pub/Sub | ~250 | 1 week | Medium — Redis lib already in repo, need session-TTL lifecycle |
| 3 — Mem0 persistence + Argo replay | ~150 | 3-5 days | Medium — checkpoint integration touches durable execution |

Total: ~3 weeks for a complete production-grade inbound channel.

## What this unlocks

- **Conversational refinement** — UI can steer an in-flight orchestrator without cancel/restart.
- **Cooperative cancellation** — graceful "stop and return what you have" instead of brute-force timeout.
- **Multi-user moderation** — a supervisor agent can inject constraints into a worker agent's loop.
- **Adaptive replanning** — when external signals change (new doc ingested, fresh search result), running agents pick it up at their next iteration without restarting.
- **Long-horizon agents** — Deep-Research style flows that span minutes/hours stay steerable throughout.

## Tracking

- Test files: `tests/runtime/integration/test_inbound_messaging_*.py` (not yet created).
- Related: cross-modal precision TODO at `docs/knowledge/CROSS_MODAL_PRECISION_TODO.md`.
- Reference impls to mimic patterns from: LangGraph `interrupt()` API, OpenAI Assistants `runs.steps.list` + `submit_tool_outputs`.
