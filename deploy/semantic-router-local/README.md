# Semantic Router (local stack)

A self-contained stack to exercise cogniverse's opt-in **semantic routing**
against a real [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
in the request path — the real-boundary counterpart to the in-process unit
tests in `tests/foundation/unit/test_semantic_router.py`.

```
cogniverse (create_dspy_lm, extra_headers)
      │  POST /v1/chat/completions   model=auto
      ▼
   Envoy  ──ext_proc(gRPC)──►  Semantic Router  ──picks model + reasoning──┐
      │                                                                     │
      └───────────────── forwards rewritten request ──────────────────────►│
                                                                     stub-upstream
                                                          (reflects back what it received)
```

Both catalog models (`basic-chat`, `pro-reasoning`) point at the **same**
stub backend on purpose: this stack proves the router's *decision* (which
model name and whether reasoning), which the stub reflects back — so no
per-model Envoy clusters or a real GPU/vLLM are needed.

## Run it

Requires Docker + Docker Compose.

```bash
docker compose -f deploy/semantic-router-local/docker-compose.yml up
# first run pulls the SR image + ~1.5GB of classifier models

# in another shell:
SR_ENVOY_URL=http://localhost:8801/v1 \
  uv run pytest tests/foundation/integration/test_semantic_router_sr_e2e.py -v
```

The test skips automatically when `SR_ENVOY_URL` is unreachable, so it is a
no-op in CI without Docker.

## What the e2e asserts

| Test | Proves |
|---|---|
| `test_completion_survives_the_proxy` | A real DSPy/litellm completion round-trips Envoy→SR→backend intact (the exact prompt sentinel comes back). |
| `test_routing_headers_reach_the_backend` | SR forwards cogniverse's `x-authz-user-groups` (tier) and `x-vsr-task` headers to the backend. |
| `test_free_tier_routes_to_basic_model` | `tier=free` selects `basic-chat`, reasoning off. |
| `test_pro_planning_routes_to_reasoning_model` | `tier=pro` + `task=orchestrator_plan` selects `pro-reasoning` with reasoning on — model and reasoning are co-decided. |
| `test_pro_non_planning_task_keeps_reasoning_off` | Same tenant, non-planning task keeps reasoning off — the task label, not just the tier, drives reasoning. |

## Wiring cogniverse to it

Set `semantic_router` in the cogniverse `config.json` `system` config
(disabled by default):

```json
"semantic_router": {
  "enabled": true,
  "semantic_router_url": "http://localhost:8801/v1",
  "tenant_tiers": { "pro-tenant": "pro", "free-tenant": "free" },
  "default_tier": "free",
  "agent_tasks": { "orchestrator_agent": "orchestrator_plan" },
  "default_task": "general"
}
```

`apply_semantic_routing(endpoint, config.semantic_router, tenant_id, agent_name)`
then rewrites `api_base` to Envoy and attaches the tier/task headers before
`create_dspy_lm`. See `docs/operations/models-and-inference.md` →
"Config-driven semantic routing".

## Caveats (read before debugging)

- **SR image tag / config schema.** `docker-compose.yml` pins
  `ghcr.io/vllm-project/semantic-router:latest`; `config.yaml` targets the
  documented v0.2 Athena / v0.3 Themis schema. If SR fails to start, the
  `routing:` block (signals + decisions) is the version-sensitive part —
  diff against `config/config.yaml` in the SR repo at your pinned tag and
  re-port the two models plus the tier/task decisions. `providers` and
  `global` are stable.
- **Tier by header vs. auth.** This stack sends the tier as a plain header
  and has SR branch on it — fine for a trusted local box. In production the
  tier signal should come from a credential SR resolves itself (Authorino /
  ext_authz → `x-authz-user-groups`), not a client-supplied header.
- **`NO_PROXY`.** If your shell exports `HTTPS_PROXY`/`HTTP_PROXY`, set
  `NO_PROXY=localhost,127.0.0.1` so the client reaches Envoy directly (the
  test sets this defensively).
- **Swapping in real vLLM.** Replace `stub-upstream` with a vLLM container
  and point each model's `backend_refs.endpoint` at it; the assertions on
  `served_model`/`reasoning` still hold, and you additionally get real
  generations.
- **Alternative stack.** You can instead run the SR project's own
  `docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up`
  (Envoy + SR + a mock vLLM) and point `SR_ENVOY_URL` at its Envoy; the
  test is decoupled from this compose file via that env var.
```
