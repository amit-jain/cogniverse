"""Policy-enforcing httpx transport.

Wraps a real ``httpx.AsyncBaseTransport`` so every outbound request is
checked against the agent's OpenShell policy ``network_policies.egress``
allow-list before it touches the wire. Requests to non-allow-listed
``(host, port)`` raise ``EgressDeniedError`` when the policy declares
``deny_all_other: true``.

Policy rules name services at their ``SystemConfig`` default addresses (the
runtime at ``localhost:8000``, Vespa at ``localhost:8080``). A deployment
serves them elsewhere, so the transport takes endpoint bindings from
``deployed_endpoint_bindings``: a rule for a default address also admits the
address the deployment configures for that service.

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
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

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


Address = Tuple[str, int]


def _url_address(url: str) -> Address:
    """``(host, port)`` a base URL dials, with the scheme's default port."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.port is not None:
        return parsed.hostname or "", int(parsed.port)
    return parsed.hostname or "", 443 if parsed.scheme == "https" else 80


def deployed_endpoint_bindings(system_config: Any) -> Dict[Address, Address]:
    """Map each service's ``SystemConfig`` default address to the address
    ``system_config`` configures for it: the runtime's A2A endpoint
    (``agent_registry_url``) and the search backend."""
    from cogniverse_foundation.config.unified_config import SystemConfig

    default = SystemConfig()
    return {
        _url_address(default.agent_registry_url): _url_address(
            system_config.agent_registry_url
        ),
        _url_address(f"{default.backend_url}:{default.backend_port}"): _url_address(
            f"{system_config.backend_url}:{system_config.backend_port}"
        ),
    }


def _matches_egress(
    host: str,
    port: int,
    rules: List[Dict[str, Any]],
    bindings: Mapping[Address, Address],
) -> bool:
    """True iff ``(host, port)`` is a rule's address or the address bound to it."""
    for rule in rules:
        address = (rule["host"], rule["port"])
        if address == (host, port) or bindings.get(address) == (host, port):
            return True
    return False


class PolicyEnforcingTransport(httpx.AsyncBaseTransport):
    """httpx transport that vets each request against an OpenShell policy.

    Args:
        policy: The agent's OpenShell policy dict (from
            ``configs/agent_policies/{agent}.yaml``).
        inner: The real transport to forward allowed requests to. Defaults
            to ``httpx.AsyncHTTPTransport()`` when omitted.
        endpoint_bindings: Rule address → deployed address, from
            ``deployed_endpoint_bindings``.
    """

    def __init__(
        self,
        policy: Dict[str, Any],
        inner: Optional[httpx.AsyncBaseTransport] = None,
        endpoint_bindings: Optional[Mapping[Address, Address]] = None,
    ) -> None:
        self._policy = policy or {}
        self._rules = _normalise_egress_rules(self._policy)
        self._bindings = dict(endpoint_bindings or {})
        self._deny_all_other = bool(
            (self._policy.get("network_policies") or {}).get("deny_all_other", False)
        )
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host, port = _request_host_port(request)
        if port == 0:
            # Non-TCP scheme (file:, data:) — let the inner transport decide.
            return await self._inner.handle_async_request(request)

        if not self._deny_all_other or _matches_egress(
            host, port, self._rules, self._bindings
        ):
            return await self._inner.handle_async_request(request)

        rules_repr = ", ".join(self._describe_rule(r) for r in self._rules)
        msg = (
            f"OpenShell policy denied egress to {host}:{port}. "
            f"Allow-listed: [{rules_repr or 'none'}]. "
            f"Update configs/agent_policies/<agent>.yaml to add this endpoint, "
            f"or remove the deny_all_other flag if egress should be open."
        )
        logger.warning(msg)
        raise EgressDeniedError(msg, request=request, host=host, port=port)

    def _describe_rule(self, rule: Dict[str, Any]) -> str:
        text = f"{rule['host']}:{rule['port']}/{rule['protocol']}"
        bound = self._bindings.get((rule["host"], rule["port"]))
        return f"{text} (bound to {bound[0]}:{bound[1]})" if bound else text

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_policy_enforcing_client(
    policy: Dict[str, Any],
    *,
    timeout: Optional[httpx.Timeout] = None,
    inner_transport: Optional[httpx.AsyncBaseTransport] = None,
    endpoint_bindings: Optional[Mapping[Address, Address]] = None,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` whose transport enforces the policy.

    Convenience wrapper so callers don't need to assemble the transport
    themselves. Extra kwargs are forwarded to ``httpx.AsyncClient`` so
    callers can still set headers, cookies, base_url, etc.
    """
    transport = PolicyEnforcingTransport(
        policy, inner=inner_transport, endpoint_bindings=endpoint_bindings
    )
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout if timeout is not None else httpx.Timeout(60.0),
        **client_kwargs,
    )
