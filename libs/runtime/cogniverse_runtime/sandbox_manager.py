"""
SandboxManager — wraps OpenShell SDK to create/manage per-agent sandboxes.

Each agent type runs inside an OpenShell sandbox with a per-agent YAML
policy controlling network egress, filesystem access, inference routing,
and process constraints.

Requires an OpenShell gateway (K3s cluster). Gracefully degrades when
the gateway is unavailable (logs a warning, dispatches without sandbox).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


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
        enabled: bool = True,
    ):
        self._policy_dir = Path(policy_dir)
        self._cluster = cluster
        self._enabled = enabled
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._client = None
        self._available = False

        if not enabled:
            logger.info("SandboxManager disabled by configuration")
            return

        self._load_policies()
        self._connect()

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
        """Connect to the OpenShell gateway."""
        try:
            from openshell import SandboxClient

            self._client = SandboxClient.from_active_cluster(cluster=self._cluster)
            health = self._client.health()
            self._available = True
            logger.info(
                f"Connected to OpenShell gateway "
                f"(cluster={self._cluster or 'active'}, health={health})"
            )
        except Exception as e:
            logger.warning(
                f"OpenShell gateway unavailable: {e}. "
                "Agents will execute without sandbox isolation."
            )
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the OpenShell gateway is reachable."""
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
                session.sandbox.name, timeout_seconds=120,
            )
            result = session.exec(
                command, timeout_seconds=timeout_seconds,
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
