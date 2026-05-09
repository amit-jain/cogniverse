"""Unit tests for the OpenShell policy-enforcing httpx transport."""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

from cogniverse_runtime.sandbox_http import (
    EgressDeniedError,
    PolicyEnforcingTransport,
    make_policy_enforcing_client,
)

SEARCH_AGENT_POLICY: Dict[str, Any] = {
    "network_policies": {
        "egress": [
            {"host": "localhost", "port": 8080, "protocol": "tcp"},
            {"host": "localhost", "port": 11434, "protocol": "tcp"},
        ],
        "deny_all_other": True,
    }
}

SUMMARIZER_POLICY: Dict[str, Any] = {
    "network_policies": {
        "egress": [{"host": "localhost", "port": 11434, "protocol": "tcp"}],
        "deny_all_other": True,
    }
}

PERMISSIVE_POLICY: Dict[str, Any] = {
    "network_policies": {
        "egress": [],
        "deny_all_other": False,
    }
}

EMPTY_DENY_ALL_POLICY: Dict[str, Any] = {
    "network_policies": {"egress": [], "deny_all_other": True}
}


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Inner transport stub that records requests instead of dialing."""

    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, content=b"ok", request=request)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
class TestAllowList:
    async def test_allowed_host_port_passes_through(self):
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SEARCH_AGENT_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://localhost:8080/search")
            assert resp.status_code == 200
            assert len(inner.requests) == 1

    async def test_second_allowed_endpoint_passes(self):
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SEARCH_AGENT_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            assert resp.status_code == 200

    async def test_explicit_port_in_url_must_match_rule_port(self):
        """A request to localhost:9999 must NOT match a rule for port 8080."""
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SEARCH_AGENT_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(EgressDeniedError) as exc:
                await client.get("http://localhost:9999/anything")
            assert exc.value.host == "localhost"
            assert exc.value.port == 9999

    async def test_different_host_blocked(self):
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SEARCH_AGENT_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(EgressDeniedError):
                await client.get("http://example.com:8080/")
            assert inner.requests == []  # never dialed

    async def test_summarizer_policy_blocks_vespa(self):
        """Tighter policy = stricter allow-list. Summarizer must not reach Vespa."""
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SUMMARIZER_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(EgressDeniedError):
                await client.get("http://localhost:8080/")
            # Ollama still allowed.
            resp = await client.get("http://localhost:11434/")
            assert resp.status_code == 200


@pytest.mark.asyncio
class TestImplicitPorts:
    async def test_https_default_port_443(self):
        inner = _RecordingTransport()
        policy = {
            "network_policies": {
                "egress": [{"host": "api.example.com", "port": 443}],
                "deny_all_other": True,
            }
        }
        transport = PolicyEnforcingTransport(policy, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("https://api.example.com/x")
            assert resp.status_code == 200
            with pytest.raises(EgressDeniedError):
                await client.get("http://api.example.com/x")  # port 80, not 443

    async def test_http_default_port_80(self):
        inner = _RecordingTransport()
        policy = {
            "network_policies": {
                "egress": [{"host": "example.com", "port": 80}],
                "deny_all_other": True,
            }
        }
        transport = PolicyEnforcingTransport(policy, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://example.com/")
            assert resp.status_code == 200


@pytest.mark.asyncio
class TestPolicyShapes:
    async def test_permissive_policy_passes_everything(self):
        """deny_all_other=False bypasses the allow-list entirely."""
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(PERMISSIVE_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://anywhere.example:1234/")
            assert resp.status_code == 200

    async def test_empty_allowlist_with_deny_all_other_blocks_everything(self):
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(EMPTY_DENY_ALL_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(EgressDeniedError):
                await client.get("http://localhost:8080/")
            assert inner.requests == []

    async def test_missing_network_policies_treated_permissively(self):
        """An empty policy dict = no enforcement (effectively permissive)."""
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport({}, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://anywhere:80/")
            assert resp.status_code == 200

    async def test_malformed_egress_entries_ignored(self):
        inner = _RecordingTransport()
        policy = {
            "network_policies": {
                "egress": [
                    {"host": "localhost", "port": 8080},  # well-formed
                    {"host": "missing port"},  # dropped
                    "not a dict",  # dropped
                    {"port": 9999, "host": None},  # dropped
                ],
                "deny_all_other": True,
            }
        }
        transport = PolicyEnforcingTransport(policy, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://localhost:8080/")
            assert resp.status_code == 200
            with pytest.raises(EgressDeniedError):
                await client.get("http://localhost:9999/")


@pytest.mark.asyncio
class TestErrorMessage:
    async def test_error_lists_allowed_endpoints_for_operator(self):
        inner = _RecordingTransport()
        transport = PolicyEnforcingTransport(SEARCH_AGENT_POLICY, inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            try:
                await client.get("http://denied.example:1/")
            except EgressDeniedError as exc:
                msg = str(exc)
                assert "denied.example" in msg
                assert "localhost:8080" in msg  # listed allowed endpoints
                assert "localhost:11434" in msg


@pytest.mark.asyncio
class TestFactory:
    async def test_make_policy_enforcing_client_uses_transport(self):
        client = make_policy_enforcing_client(SEARCH_AGENT_POLICY)
        try:
            assert isinstance(client._transport, PolicyEnforcingTransport)
        finally:
            await client.aclose()

    async def test_factory_forwards_extra_kwargs(self):
        client = make_policy_enforcing_client(
            SEARCH_AGENT_POLICY,
            base_url="http://localhost:8080",
            headers={"X-Test": "yes"},
        )
        try:
            assert client.headers.get("X-Test") == "yes"
            assert str(client.base_url).startswith("http://localhost:8080")
        finally:
            await client.aclose()
