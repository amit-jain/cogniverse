"""Policy-enforcing httpx transport (D.1).

Wraps a real ``httpx.AsyncBaseTransport`` so every outbound request is
checked against the agent's OpenShell policy ``network_policies.egress``
allow-list before it touches the wire. Requests to non-allow-listed
``(host, port)`` raise ``EgressDeniedError`` when the policy declares
``deny_all_other: true``.

This is application-layer enforcement: it complements (does not replace)
in-cluster k8s ``NetworkPolicy`` enforcement when cogniverse is deployed
via the production Helm chart. Defence in depth — kernel-layer policy stops
out-of-process bypass; this transport stops the agent code itself from
making egress calls the policy disallows, and surfaces a clear error so
operators see the violation in logs rather than silent kernel rejects.

Wiring: agents whose dispatcher path stamps a policy build their httpx
client via ``make_policy_enforcing_client(policy)`` instead of the bare
``httpx.AsyncClient``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class EgressDeniedError(httpx.RequestError):
    """Raised when a request targets a host:port not in the policy allow-list."""

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        host: str,
        port: int,
    ) -> None:
        super().__init__(message, request=request)
        self.host = host
        self.port = port


def _normalise_egress_rules(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ``(host, port, protocol)`` rules from a policy dict.

    Tolerates missing keys — an empty rule list paired with
    ``deny_all_other: true`` blocks everything (a useful posture for
    fully-isolated agents).
    """
    egress = (policy.get("network_policies") or {}).get("egress") or []
    rules: List[Dict[str, Any]] = []
    for rule in egress:
        if not isinstance(rule, dict):
            continue
        host = rule.get("host")
        port = rule.get("port")
        if host is None or port is None:
            continue
        rules.append(
            {
                "host": str(host),
                "port": int(port),
                "protocol": str(rule.get("protocol") or "tcp").lower(),
            }
        )
    return rules


def _request_host_port(request: httpx.Request) -> tuple[str, int]:
    """Resolve the (host, port) the request will actually dial."""
    url = request.url
    host = url.host or ""
    if url.port is not None:
        port = int(url.port)
    elif url.scheme == "https":
        port = 443
    elif url.scheme == "http":
        port = 80
    else:
        # Other schemes (file, data, etc.) — pass through; the underlying
        # transport will handle them.
        port = 0
    return host, port


def _matches_egress(host: str, port: int, rules: List[Dict[str, Any]]) -> bool:
    """True iff ``(host, port)`` matches any rule in the allow-list."""
    for rule in rules:
        if rule["host"] == host and rule["port"] == port:
            return True
    return False


class PolicyEnforcingTransport(httpx.AsyncBaseTransport):
    """httpx transport that vets each request against an OpenShell policy.

    Args:
        policy: The agent's OpenShell policy dict (from
            ``configs/agent_policies/{agent}.yaml``).
        inner: The real transport to forward allowed requests to. Defaults
            to ``httpx.AsyncHTTPTransport()`` when omitted.
    """

    def __init__(
        self,
        policy: Dict[str, Any],
        inner: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._policy = policy or {}
        self._rules = _normalise_egress_rules(self._policy)
        self._deny_all_other = bool(
            (self._policy.get("network_policies") or {}).get("deny_all_other", False)
        )
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host, port = _request_host_port(request)
        if port == 0:
            # Non-TCP scheme (file:, data:) — let the inner transport decide.
            return await self._inner.handle_async_request(request)

        if not self._deny_all_other or _matches_egress(host, port, self._rules):
            return await self._inner.handle_async_request(request)

        rules_repr = ", ".join(
            f"{r['host']}:{r['port']}/{r['protocol']}" for r in self._rules
        )
        msg = (
            f"OpenShell policy denied egress to {host}:{port}. "
            f"Allow-listed: [{rules_repr or 'none'}]. "
            f"Update configs/agent_policies/<agent>.yaml to add this endpoint, "
            f"or remove the deny_all_other flag if egress should be open."
        )
        logger.warning(msg)
        raise EgressDeniedError(msg, request=request, host=host, port=port)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_policy_enforcing_client(
    policy: Dict[str, Any],
    *,
    timeout: Optional[httpx.Timeout] = None,
    inner_transport: Optional[httpx.AsyncBaseTransport] = None,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` whose transport enforces the policy.

    Convenience wrapper so callers don't need to assemble the transport
    themselves. Extra kwargs are forwarded to ``httpx.AsyncClient`` so
    callers can still set headers, cookies, base_url, etc.
    """
    transport = PolicyEnforcingTransport(policy, inner=inner_transport)
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout if timeout is not None else httpx.Timeout(60.0),
        **client_kwargs,
    )
