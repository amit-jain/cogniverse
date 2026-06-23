"""
SandboxManager — wraps OpenShell SDK to create/manage per-agent sandboxes.

Each agent type runs inside an OpenShell sandbox with a per-agent YAML
policy controlling network egress, filesystem access, inference routing,
and process constraints.

Requires an OpenShell gateway (K3s cluster). The ``SandboxPolicy`` knob
controls behaviour when the gateway is unreachable: ``required`` refuses
to start, ``optional`` degrades with a warning, ``disabled`` skips entirely.

Every sandbox lifecycle event (create_session, wait_ready, exec, delete)
is wrapped in an OpenTelemetry span so Phoenix can correlate sandbox
behaviour with the parent agent span. OOM and policy-denied errors are
surfaced as span attributes from stderr / exit_code patterns.
"""

from __future__ import annotations

import enum
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from opentelemetry import trace

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


# OOM / policy-denied detection from stderr + exit_code. These
# patterns are heuristic; OpenShell's exec result does not (today) carry
# structured failure reason. The patterns are conservative so we don't
# false-positive on user code that happens to mention these words.
_OOM_EXIT_CODES = {137, 139}  # SIGKILL, SIGSEGV (which OOM-killer often uses)
_OOM_STDERR_MARKERS = ("Killed", "OOMKilled", "out of memory", "oom-kill")
_DENIED_STDERR_MARKERS = (
    "Operation not permitted",
    "permission denied",
    "syscall denied",
    "blocked by policy",
)


def _classify_exec_failure(exit_code: int, stderr: str) -> Dict[str, bool]:
    """Categorise an exec result as oom / denied / neither for span attributes."""
    stderr_l = (stderr or "").lower()
    oom = exit_code in _OOM_EXIT_CODES or any(
        m.lower() in stderr_l for m in _OOM_STDERR_MARKERS
    )
    denied = any(m.lower() in stderr_l for m in _DENIED_STDERR_MARKERS)
    return {"openshell.oom": oom, "openshell.policy_denied": denied}


def _exec_under_span(
    session, command, timeout_seconds, common_attrs, parent_span, tracer
) -> Dict[str, Any]:
    """Run ``session.exec`` under a ``sandbox.exec`` span and return the result.

    Stamps the exit code + failure classification on both the exec span and
    the parent span (wall time only on the exec span), then returns the
    stdout/stderr/exit_code dict. Shared by the pooled and non-pooled exec
    paths so the span-emission contract lives in one place.
    """
    with tracer.start_as_current_span(
        "sandbox.exec", attributes=common_attrs
    ) as exec_span:
        start = time.monotonic()
        result = session.exec(command, timeout_seconds=timeout_seconds)
        wall_ms = (time.monotonic() - start) * 1000.0

        classification = _classify_exec_failure(
            int(getattr(result, "exit_code", -1)),
            getattr(result, "stderr", "") or "",
        )
        exit_code = int(result.exit_code)
        for span in (exec_span, parent_span):
            span.set_attribute("openshell.exit_code", exit_code)
            for k, v in classification.items():
                span.set_attribute(k, v)
        exec_span.set_attribute("openshell.wall_ms", wall_ms)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }


_DEFAULT_POLICY_DIR = Path("configs/agent_policies")


def _probe_gateway_endpoint(endpoint: str, timeout: float = 2.0) -> None:
    """TCP-probe the host:port of an OpenShell gateway endpoint.

    grpc creates lazy channels and does not dial at construction, so a
    bogus endpoint produces a happy-looking SandboxClient that only
    fails on first RPC. policy=REQUIRED needs to refuse boot eagerly,
    so we open a short TCP connection here. Raises ``OSError`` (or
    subclass) when the endpoint is unreachable; caller catches and
    flips ``_available=False``.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    host = parsed.hostname or endpoint
    port = parsed.port
    if port is None:
        # Default grpc-over-http(s) ports.
        port = 443 if parsed.scheme == "https" else 80
    with socket.create_connection((host, port), timeout=timeout):
        return


class SandboxManager:
    """
    Manages OpenShell sandboxes for per-agent execution isolation.

    Each agent type gets a policy declaration loaded from
    ``configs/agent_policies/{agent_type}.yaml``. CodingAgent is the
    only agent whose policy is currently *enforced* by an OpenShell
    container sandbox; the other agents' policies are consumed at
    runtime by ``consult_egress_policy`` (audit log) and at deploy time
    by ``cogniverse-runtime egress-netpol`` (k8s NetworkPolicy
    generator).
    """

    def __init__(
        self,
        policy_dir: Path | None = None,
        cluster: str | None = None,
        enabled: bool | None = None,
        policy: SandboxPolicy | str | None = None,
    ):
        """Initialize the sandbox manager.

        Args:
            policy_dir: Directory containing per-agent policy YAMLs.
                Defaults to ``configs/agent_policies/``.
            cluster: OpenShell cluster name (None = active).
            enabled: ``True`` maps to ``policy=optional``,
                ``False`` to ``policy=disabled``. Prefer ``policy`` for
                explicit configuration; this kwarg remains for tests
                and shorthand callers.
            policy: Explicit policy knob; takes precedence over
                ``enabled`` when both are passed.
        """
        self._policy_dir = self._resolve_policy_dir(policy_dir)
        self._cluster = cluster
        self._policy = self._resolve_policy(policy=policy, enabled=enabled)
        # Maintain ``_enabled`` as the legacy "should we try at all" flag —
        # external callers (tests, dispatcher hot-path) read .enabled.
        self._enabled = self._policy is not SandboxPolicy.DISABLED
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._client = None
        self._available = False

        # pooled sessions. Lazily created on first exec; uses
        # SandboxPoolConfig.from_environment() so operators can disable or
        # tune via env vars without code changes.
        self._pool: Optional[Any] = None

        # optional cert-rotation watcher. Operators wire one in via
        # ``attach_cert_rotator()`` so an external poller can call
        # ``trigger_on_auth_failure()`` from the exec error path. Kept
        # optional so unit tests + non-mTLS deployments don't pay for it.
        self._cert_rotator: Optional[Any] = None

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
    def _resolve_policy_dir(policy_dir: Path | None) -> Path:
        """Pick the policy dir: the explicit override or the default
        ``configs/agent_policies/``. A missing directory is surfaced later in
        ``_load_policies`` as a warning."""
        if policy_dir is not None:
            return Path(policy_dir)
        return _DEFAULT_POLICY_DIR

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
                # ``SandboxClient(endpoint=...)`` only stores the endpoint
                # and creates a lazy grpc channel — no eager dial. So a
                # bogus endpoint produces a happy-looking client that
                # only fails on first RPC. Probe the endpoint host:port
                # with a short TCP connect so policy=REQUIRED actually
                # refuses to boot when the gateway is unreachable.
                _probe_gateway_endpoint(override_endpoint)
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

    def make_http_client(self, agent_type: str, **client_kwargs: Any) -> Any:
        """Build an httpx.AsyncClient with policy enforcement for an agent.

        When an agent has a registered OpenShell policy, its outbound HTTP
        traffic is vetted against the policy's egress allow-list.
        Agents that do not have a registered policy fall through to a plain
        ``httpx.AsyncClient`` (back-compat: existing callers keep working).

        Operators can disable enforcement by setting
        ``COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT=disabled`` at boot — useful
        in dev when iterating on policies.
        """
        import os as _os

        import httpx as _httpx

        from cogniverse_runtime.sandbox_http import (
            make_policy_enforcing_client,
        )

        enforcement = _os.environ.get(
            "COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT", ""
        ).lower()
        policy = self._policies.get(agent_type)
        if (
            policy is None
            or enforcement == "disabled"
            or self._policy is SandboxPolicy.DISABLED
        ):
            return _httpx.AsyncClient(**client_kwargs)
        return make_policy_enforcing_client(policy, **client_kwargs)

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
        Each lifecycle phase (create_session, wait_ready, exec, delete)
        emits an OpenTelemetry span so Phoenix correlates sandbox
        behaviour with the parent agent span. OOM and policy-denied
        outcomes surface as span attributes derived from stderr/exit_code.

        Returns:
            Dict with stdout, stderr, exit_code. None if sandbox unavailable.
        """
        if not self._available or not self._client:
            return None

        # go through the pool when enabled. Pool reuses one session
        # per agent_type across calls, eliminating per-call container churn.
        pool = self._get_or_create_pool()
        if pool is not None and pool.config.enabled:
            return self._exec_pooled(pool, agent_type, command, timeout_seconds)

        tracer = trace.get_tracer(__name__)
        common_attrs = {
            "openshell.agent_type": agent_type,
            "openshell.command_first": command[0] if command else "",
            "openshell.timeout_seconds": int(timeout_seconds),
        }

        session = None
        with tracer.start_as_current_span(
            "sandbox.exec_in_sandbox", attributes=common_attrs
        ) as parent_span:
            try:
                with tracer.start_as_current_span(
                    "sandbox.create_session", attributes=common_attrs
                ):
                    session = self._client.create_session()
                    parent_span.set_attribute(
                        "openshell.session_name",
                        getattr(getattr(session, "sandbox", None), "name", "")
                        or getattr(session, "id", ""),
                    )

                with tracer.start_as_current_span(
                    "sandbox.wait_ready",
                    attributes={**common_attrs, "openshell.wait_timeout_s": 120},
                ):
                    self._client.wait_ready(
                        session.sandbox.name,
                        timeout_seconds=120,
                    )

                return _exec_under_span(
                    session,
                    command,
                    timeout_seconds,
                    common_attrs,
                    parent_span,
                    tracer,
                )
            except Exception as e:
                logger.warning(f"Sandbox exec failed for {agent_type}: {e}")
                parent_span.set_attribute("openshell.error", type(e).__name__)
                parent_span.record_exception(e)
                self._maybe_trigger_cert_rotator(e)
                return {"stdout": "", "stderr": str(e), "exit_code": -1}
            finally:
                if session:
                    with tracer.start_as_current_span(
                        "sandbox.delete", attributes=common_attrs
                    ):
                        try:
                            session.delete()
                        except Exception as exc:
                            logger.debug("sandbox.delete failed (non-fatal): %s", exc)

    def attach_cert_rotator(self, rotator: Any) -> None:
        """Wire a :class:`CertRotator` into the exec error path.

        Once attached, any exec failure that looks like an auth/TLS
        problem (matched by class name or stderr substring) eagerly
        triggers a reconnect via ``rotator.trigger_on_auth_failure()`` —
        rotation is then visible to the next request without waiting for
        the rotator's polling tick. The rotator's own rate-limit
        prevents thrashing.
        """
        self._cert_rotator = rotator

    def _maybe_trigger_cert_rotator(self, err: BaseException) -> None:
        """Eager reconnect on auth/TLS-shaped exec failures."""
        if self._cert_rotator is None:
            return
        marker = f"{type(err).__name__}:{str(err)[:120]}".lower()
        if any(
            tag in marker
            for tag in (
                "auth",
                "x509",
                "tls",
                "ssl",
                "certificate",
                "permission",
                "unauthenticated",
                "unauthorized",
            )
        ):
            try:
                self._cert_rotator.trigger_on_auth_failure(repr(err))
            except Exception as exc:
                logger.debug("cert rotator trigger raised (non-fatal): %s", exc)

    def _get_or_create_pool(self):
        """Lazily build the SandboxSessionPool from env config."""
        if self._pool is not None:
            return self._pool
        if not self._available or not self._client:
            return None
        from cogniverse_runtime.sandbox_pool import (
            SandboxPoolConfig,
            SandboxSessionPool,
        )

        cfg = SandboxPoolConfig.from_environment()
        if not cfg.enabled:
            # Cache a disabled placeholder so we don't rebuild every call.
            self._pool = SandboxSessionPool(self._client, config=cfg)
            return self._pool
        self._pool = SandboxSessionPool(self._client, config=cfg)
        logger.info(
            "Sandbox session pool initialised (max_size=%d, idle_s=%.0f)",
            cfg.max_pool_size,
            cfg.max_idle_seconds,
        )
        return self._pool

    def _exec_pooled(
        self,
        pool: Any,
        agent_type: str,
        command: list,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        """Run an exec through the session pool with full span emission."""
        tracer = trace.get_tracer(__name__)
        common_attrs = {
            "openshell.agent_type": agent_type,
            "openshell.command_first": command[0] if command else "",
            "openshell.timeout_seconds": int(timeout_seconds),
            "openshell.pooled": True,
        }
        with tracer.start_as_current_span(
            "sandbox.exec_in_sandbox", attributes=common_attrs
        ) as parent_span:

            def _run(session: Any) -> Dict[str, Any]:
                parent_span.set_attribute(
                    "openshell.session_name",
                    getattr(getattr(session, "sandbox", None), "name", "")
                    or getattr(session, "id", ""),
                )
                return _exec_under_span(
                    session,
                    command,
                    timeout_seconds,
                    common_attrs,
                    parent_span,
                    tracer,
                )

            try:
                return pool.with_session(agent_type, _run)
            except Exception as e:
                logger.warning(f"Pooled sandbox exec failed for {agent_type}: {e}")
                parent_span.set_attribute("openshell.error", type(e).__name__)
                parent_span.record_exception(e)
                return {"stdout": "", "stderr": str(e), "exit_code": -1}

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
        """Close the gateway connection and tear down any pooled sessions."""
        if self._pool is not None:
            try:
                self._pool.close_all()
            except Exception as exc:
                logger.debug("Pool close_all failed (non-fatal): %s", exc)
            self._pool = None
        if self._client:
            self._client.close()
            self._client = None
            self._available = False
