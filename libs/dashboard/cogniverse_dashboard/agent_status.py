"""Agent availability probes against the unified runtime."""

from __future__ import annotations

from contextlib import nullcontext

import httpx


def probe_agents(
    runtime_url: str,
    agents: list[str],
    *,
    client: httpx.Client | None = None,
    timeout: float = 5.0,
) -> dict[str, dict[str, str]]:
    """Report each agent's availability from the runtime agent registry.

    All agents are served by the unified runtime, so availability is whether
    ``GET /agents/<name>`` resolves: 200 means the registry serves it, 404
    means it is not registered, and any other response or a transport failure
    means it is not usable. Returns ``{"error": ...}`` when the runtime itself
    is unreachable, since no per-agent answer is meaningful then.
    """
    ctx = nullcontext(client) if client is not None else httpx.Client(timeout=timeout)
    with ctx as http:
        try:
            resp = http.get(f"{runtime_url}/health", timeout=timeout)
        except (httpx.HTTPError, OSError) as e:
            return {"error": f"Runtime not reachable: {e}"}
        if resp.status_code != 200:
            return {"error": f"Runtime not reachable (HTTP {resp.status_code})"}

        results: dict[str, dict[str, str]] = {}
        for agent_name in agents:
            url = f"{runtime_url}/agents/{agent_name}"
            entry: dict[str, str] = {"status": "offline", "url": url}
            try:
                agent_resp = http.get(url, timeout=timeout)
            except (httpx.HTTPError, OSError):
                entry["message"] = "Connection failed"
            else:
                if agent_resp.status_code == 200:
                    entry = {"status": "online", "url": url}
                elif agent_resp.status_code == 404:
                    entry["message"] = "Not registered (HTTP 404)"
                else:
                    entry["message"] = f"HTTP {agent_resp.status_code}"
            results[agent_name.replace("_", " ").title()] = entry
    return results
