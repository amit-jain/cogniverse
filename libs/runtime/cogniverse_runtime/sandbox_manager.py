"""
SandboxManager — wraps OpenShell SDK to create/manage per-agent sandboxes.

Each agent type runs inside an OpenShell sandbox with a per-agent YAML
policy controlling network egress, filesystem access, inference routing,
and process constraints.

Requires an OpenShell gateway (K3s cluster). The ``SandboxPolicy`` knob
controls behaviour when the gateway is unreachable: ``required`` refuses
to start, ``optional`` degrades with a warning, ``disabled`` skips entirely.
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class SandboxPolicy(str, enum.Enum):
    """Boot-time policy for the OpenShell sandbox.

    ``required``: the runtime refuses to start unless the gateway is
        reachable. Use for production tenants where egress isolation is a
        compliance requirement.
    ``optional``: best-effort connect; log a warning and continue without
        sandbox enforcement when the gateway is missing. Default for dev.
    ``disabled``: do not even attempt to connect; SandboxManager.available
        is permanently False. Use when sandboxing is intentionally off.
    """

    REQUIRED = "required"
    OPTIONAL = "optional"
    DISABLED = "disabled"


class SandboxGatewayUnavailableError(RuntimeError):
    """Raised at boot when policy=required but the gateway is unreachable."""


class SandboxManager:
    """
    Manages OpenShell sandboxes for per-agent execution isolation.

    Each agent type gets a sandbox with policy loaded from
    configs/openshell/{agent_type}.yaml.
    """

    def __init__(
        self,
        policy_dir: Path = Path("configs/openshell"),
        cluster: str | None = None,
        enabled: bool | None = None,
        policy: SandboxPolicy | str | None = None,
    ):
        """Initialize the sandbox manager.

        Args:
            policy_dir: Directory containing per-agent policy YAML files.
            cluster: OpenShell cluster name (None = active).
            enabled: Deprecated. ``True`` maps to ``policy=optional``,
                ``False`` to ``policy=disabled``. Use ``policy`` for new
                code; this kwarg remains for backwards compatibility with
                existing tests/callers.
            policy: New explicit policy knob; takes precedence over
                ``enabled`` when both are passed.
        """
        self._policy_dir = Path(policy_dir)
        self._cluster = cluster
        self._policy = self._resolve_policy(policy=policy, enabled=enabled)
        # Maintain ``_enabled`` as the legacy "should we try at all" flag —
        # external callers (tests, dispatcher hot-path) read .enabled.
        self._enabled = self._policy is not SandboxPolicy.DISABLED
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._client = None
        self._available = False

        if self._policy is SandboxPolicy.DISABLED:
            logger.info("SandboxManager disabled by configuration (policy=disabled)")
            return

        self._load_policies()
        self._connect()

        if self._policy is SandboxPolicy.REQUIRED and not self._available:
            raise SandboxGatewayUnavailableError(
                "sandbox.policy=required but the OpenShell gateway is "
                "unreachable. Refusing to start. Either install/start the "
                "gateway or set sandbox.policy=optional to degrade with "
                "a warning instead."
            )

    @staticmethod
    def _resolve_policy(
        policy: SandboxPolicy | str | None, enabled: bool | None
    ) -> SandboxPolicy:
        """Translate the (policy, enabled) inputs into a single SandboxPolicy.

        ``policy`` wins when both are provided. When neither is provided,
        default to ``OPTIONAL`` (degrade-with-warning). The legacy ``enabled``
        kwarg keeps existing call sites working: True → optional, False → disabled.
        """
        if policy is not None:
            if isinstance(policy, str):
                policy = SandboxPolicy(policy.lower())
            return policy
        if enabled is None:
            return SandboxPolicy.OPTIONAL
        return SandboxPolicy.OPTIONAL if enabled else SandboxPolicy.DISABLED

    def _load_policies(self) -> None:
        """Load per-agent policy YAML files from the policy directory."""
        if not self._policy_dir.exists():
            logger.warning(f"Policy directory not found: {self._policy_dir}")
            return

        for policy_file in self._policy_dir.glob("*.yaml"):
            agent_type = policy_file.stem
            with open(policy_file) as f:
                self._policies[agent_type] = yaml.safe_load(f) or {}
            logger.info(f"Loaded policy for {agent_type}")

        logger.info(f"Loaded {len(self._policies)} agent policies")

    def _connect(self) -> None:
        """Connect to the OpenShell gateway.

        Prefers `OPENSHELL_GATEWAY_ENDPOINT` env var when set (for
        containerized deployments where the host gateway is reachable
        via host.docker.internal or similar). Falls back to the active
        cluster metadata (~/.config/openshell/gateways/<name>/metadata.json).
        """
        import os

        try:
            from openshell import SandboxClient

            override_endpoint = os.environ.get("OPENSHELL_GATEWAY_ENDPOINT")
            if override_endpoint:
                self._client = SandboxClient(endpoint=override_endpoint)
                logger.info(f"Connected to OpenShell gateway at {override_endpoint}")
            else:
                self._client = SandboxClient.from_active_cluster(cluster=self._cluster)
                logger.info(
                    f"Connected to OpenShell gateway "
                    f"(cluster={self._cluster or 'active'})"
                )
            self._available = True
        except Exception as e:
            logger.warning(
                f"OpenShell gateway unavailable: {e}. "
                "Agents will execute without sandbox isolation."
            )
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the OpenShell gateway is reachable.

        If not currently available, attempts a fresh connection — this
        lets the manager recover from transient failures or from the
        openshell package becoming importable after startup.
        """
        if not self._available and self._enabled:
            self._connect()
        return self._available

    def reconnect(self) -> bool:
        """Force a reconnection attempt. Returns True if available."""
        if self._enabled:
            self._connect()
        return self._available

    def get_policy(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get the policy for an agent type, or None."""
        return self._policies.get(agent_type)

    def reload_policies(self) -> None:
        """Hot-reload policy files from disk."""
        self._policies.clear()
        self._load_policies()

    def create_sandbox(self, agent_type: str) -> Optional[Any]:
        """
        Create a sandbox for the given agent type.

        Returns None if the gateway is unavailable or sandboxing is disabled.
        """
        if not self._available or not self._client:
            return None

        policy = self._policies.get(agent_type)
        if not policy:
            logger.warning(f"No policy for agent type '{agent_type}', using defaults")

        try:
            session = self._client.create_session()
            logger.info(
                f"Created sandbox {session.id} for {agent_type} "
                f"(policy={'custom' if policy else 'default'})"
            )
            return session
        except Exception as e:
            logger.warning(f"Failed to create sandbox for {agent_type}: {e}")
            return None

    def exec_in_sandbox(
        self,
        agent_type: str,
        command: list[str],
        timeout_seconds: int = 60,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a command inside a sandbox for the given agent type.

        Uses the OpenShell Python SDK (SandboxClient.create_session + exec).

        Returns:
            Dict with stdout, stderr, exit_code. None if sandbox unavailable.
        """
        if not self._available or not self._client:
            return None

        session = None
        try:
            session = self._client.create_session()
            self._client.wait_ready(
                session.sandbox.name,
                timeout_seconds=120,
            )
            result = session.exec(
                command,
                timeout_seconds=timeout_seconds,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
        except Exception as e:
            logger.warning(f"Sandbox exec failed for {agent_type}: {e}")
            return {"stdout": "", "stderr": str(e), "exit_code": -1}
        finally:
            if session:
                session.delete()

    def list_sandboxes(self) -> list:
        """List active sandboxes."""
        if not self._available or not self._client:
            return []
        try:
            return self._client.list()
        except Exception as e:
            logger.warning(f"Failed to list sandboxes: {e}")
            return []

    def close(self) -> None:
        """Close the gateway connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._available = False
