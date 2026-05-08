"""Unit tests for SandboxPolicy resolution + boot-time enforcement.

Integration coverage (real OpenShell gateway) lives in
tests/agents/integration/test_sandbox_integration.py — these unit tests
exercise the policy interpretation, the legacy ``enabled`` alias, and the
fast-fail path when policy=required without a reachable gateway.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cogniverse_runtime.sandbox_manager import (
    SandboxGatewayUnavailableError,
    SandboxManager,
    SandboxPolicy,
)


class TestSandboxPolicyResolution:
    def test_policy_explicit_wins_over_enabled(self):
        policy = SandboxManager._resolve_policy(
            policy=SandboxPolicy.REQUIRED, enabled=False
        )
        assert policy is SandboxPolicy.REQUIRED

    def test_policy_string_value_accepted(self):
        policy = SandboxManager._resolve_policy(policy="required", enabled=None)
        assert policy is SandboxPolicy.REQUIRED

    def test_policy_string_case_insensitive(self):
        assert (
            SandboxManager._resolve_policy(policy="OPTIONAL", enabled=None)
            is SandboxPolicy.OPTIONAL
        )

    def test_legacy_enabled_true_maps_to_optional(self):
        assert (
            SandboxManager._resolve_policy(policy=None, enabled=True)
            is SandboxPolicy.OPTIONAL
        )

    def test_legacy_enabled_false_maps_to_disabled(self):
        assert (
            SandboxManager._resolve_policy(policy=None, enabled=False)
            is SandboxPolicy.DISABLED
        )

    def test_default_when_neither_provided(self):
        assert (
            SandboxManager._resolve_policy(policy=None, enabled=None)
            is SandboxPolicy.OPTIONAL
        )

    def test_invalid_policy_string_raises(self):
        with pytest.raises(ValueError):
            SandboxManager._resolve_policy(policy="bogus", enabled=None)


class TestPolicyDisabledShortCircuits:
    """policy=disabled must skip both policy loading and gateway connect."""

    def test_disabled_does_not_try_to_connect(self):
        with (
            patch.object(SandboxManager, "_connect", autospec=True) as connect,
            patch.object(SandboxManager, "_load_policies", autospec=True) as load,
        ):
            mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
            assert mgr._available is False
            connect.assert_not_called()
            load.assert_not_called()


def _connect_sets(value: bool):
    """Build an autospec-friendly _connect side_effect that flips _available."""

    def _impl(self):
        self._available = value

    return _impl


class TestPolicyOptionalDegrades:
    """policy=optional must connect best-effort and tolerate failure."""

    def test_optional_with_unreachable_gateway_returns_unavailable(self, tmp_path):
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(False),
        ):
            mgr = SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.OPTIONAL,
            )
            # Construction succeeds even with unreachable gateway.
            assert mgr._policy is SandboxPolicy.OPTIONAL
            assert mgr._available is False


class TestMakeHttpClient:
    """D.1 — SandboxManager.make_http_client returns a policy-aware client."""

    def test_returns_enforcing_client_when_policy_registered(self, tmp_path):
        from cogniverse_runtime.sandbox_http import PolicyEnforcingTransport

        # Seed a policy file so SandboxManager._load_policies registers it.
        (tmp_path / "demo_agent.yaml").write_text(
            "network_policies:\n  egress: []\n  deny_all_other: true\n"
        )
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(False),
        ):
            mgr = SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.OPTIONAL,
            )
            client = mgr.make_http_client("demo_agent")
            try:
                assert isinstance(client._transport, PolicyEnforcingTransport)
            finally:
                # Async client; close synchronously by using run_until_complete
                # equivalent. httpx.AsyncClient also exposes a sync close path
                # when never used in an async context — call .aclose() via asyncio.
                import asyncio

                asyncio.run(client.aclose())

    def test_returns_plain_client_when_no_policy_registered(self, tmp_path):
        import httpx as _httpx

        from cogniverse_runtime.sandbox_http import PolicyEnforcingTransport

        # No policy files in the dir.
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(False),
        ):
            mgr = SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.OPTIONAL,
            )
            client = mgr.make_http_client("agent_with_no_policy")
            try:
                assert not isinstance(client._transport, PolicyEnforcingTransport)
                assert isinstance(client, _httpx.AsyncClient)
            finally:
                import asyncio

                asyncio.run(client.aclose())

    def test_disabled_policy_yields_plain_client(self, tmp_path, monkeypatch):
        """policy=disabled returns a bare client even if a policy file exists."""
        import asyncio

        from cogniverse_runtime.sandbox_http import PolicyEnforcingTransport

        (tmp_path / "agent.yaml").write_text(
            "network_policies:\n  egress: []\n  deny_all_other: true\n"
        )
        mgr = SandboxManager(
            policy_dir=tmp_path,
            policy=SandboxPolicy.DISABLED,
        )
        client = mgr.make_http_client("agent")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport)
        finally:
            asyncio.run(client.aclose())

    def test_env_disable_overrides_policy(self, tmp_path, monkeypatch):
        """COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT=disabled bypasses enforcement."""
        import asyncio

        from cogniverse_runtime.sandbox_http import PolicyEnforcingTransport

        (tmp_path / "agent.yaml").write_text(
            "network_policies:\n  egress: []\n  deny_all_other: true\n"
        )
        monkeypatch.setenv("COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT", "disabled")
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(False),
        ):
            mgr = SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.OPTIONAL,
            )
            client = mgr.make_http_client("agent")
            try:
                assert not isinstance(client._transport, PolicyEnforcingTransport)
            finally:
                asyncio.run(client.aclose())


class TestPolicyRequiredRefusesToBoot:
    """policy=required must raise SandboxGatewayUnavailableError if gateway missing."""

    def test_required_without_gateway_raises(self, tmp_path):
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(False),
        ):
            with pytest.raises(SandboxGatewayUnavailableError) as exc:
                SandboxManager(
                    policy_dir=tmp_path,
                    policy=SandboxPolicy.REQUIRED,
                )
            assert "policy=required" in str(exc.value)
            assert "gateway" in str(exc.value).lower()

    def test_required_with_gateway_succeeds(self, tmp_path):
        with patch.object(
            SandboxManager,
            "_connect",
            autospec=True,
            side_effect=_connect_sets(True),
        ):
            mgr = SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.REQUIRED,
            )
            assert mgr._policy is SandboxPolicy.REQUIRED
            assert mgr._available is True
