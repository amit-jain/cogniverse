# Audit Cycle 6 — Cluster: AgentRegistry eagerly opened an httpx client

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `core/registries/agent_registry.py:50` `__init__` | E | unconditionally constructed `httpx.AsyncClient(timeout=10.0)` on every instantiation — a registry used only for local agent lookup opened (and had to close) a client it never used | make `http_client` a lazy `@property` (constructed on first use); `close()` only `aclose()`s if a client was created |

## Test (`tests/core/unit/test_agent_registry_lazy_client.py`)

- `http_client` is None until accessed, then created and cached.
- `close()` without prior use constructs nothing.

Health-endpoint suite (which builds an AgentRegistry) stays green.
