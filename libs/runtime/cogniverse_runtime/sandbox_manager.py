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

import asyncio
import enum
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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


async def _settle_sandbox_call(task):
    """Wait for an owned SDK call even when its waiter is cancelled again."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _run_sandbox_call(function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _settle_sandbox_call(task)
        raise


class SandboxTaskSession:
    """Exclusive task execution; cancellation joins the active SDK call."""

    def __init__(
        self,
        session,
        agent_type: str,
        tenant_id: str,
        on_exec_error: Optional[Callable[[BaseException], None]] = None,
    ):
        self._session = session
        self._agent_type = agent_type
        self._tenant_id = tenant_id
        self._closed = False
        self._on_exec_error = on_exec_error

    @property
    def session_name(self) -> str:
        return self._session.sandbox.name

    async def exec(self, command: list[str], timeout_seconds: int) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("Sandbox task session is closed")
        return await _run_sandbox_call(self._exec, command, timeout_seconds)

    def _exec(self, command, timeout_seconds):
        tracer = trace.get_tracer(__name__)
        attrs = {
            "openshell.agent_type": self._agent_type,
            "openshell.tenant_id": self._tenant_id,
            "openshell.session_name": self.session_name,
            "openshell.command_first": command[0] if command else "",
            "openshell.timeout_seconds": timeout_seconds,
        }
        with tracer.start_as_current_span(
            "sandbox.task_exec", attributes=attrs
        ) as parent_span:
            try:
                return _exec_under_span(
                    self._session, command, timeout_seconds, attrs, parent_span, tracer
                )
            except Exception as exc:
                parent_span.set_attribute("openshell.error", type(exc).__name__)
                if self._on_exec_error is not None:
                    self._on_exec_error(exc)
                raise


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
    stdout/stderr/exit_code dict.
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

    # A scheme-less ``host:port`` (the in-cluster gRPC endpoint) misparses:
    # urlparse treats the dotted hostname as the URL scheme, so ``hostname``
    # is None and the whole string (port included) is left as the host, which
    # then fails DNS. Prepend ``//`` when there's no scheme so it parses as a
    # netloc. ``https://host:port`` (host mode) already parses correctly.
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    host = parsed.hostname or endpoint
    port = parsed.port
    if port is None:
        # gRPC over TLS defaults; the in-cluster service port is 8080.
        port = 443 if parsed.scheme == "https" else 8080
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
        policy: SandboxPolicy | str | None = None,
    ):
        """Initialize the sandbox manager.

        Args:
            policy_dir: Directory containing per-agent policy YAMLs.
                Defaults to ``configs/agent_policies/``.
            cluster: OpenShell cluster name (None = active).
            policy: Sandbox policy knob (disabled / optional / required);
                defaults to ``optional`` (degrade-with-warning).
        """
        self._policy_dir = self._resolve_policy_dir(policy_dir)
        self._cluster = cluster
        self._policy = self._resolve_policy(policy)
        # ``_enabled`` is the "should we try at all" flag that external
        # callers (dispatcher hot-path, tests) read via ``.enabled``.
        self._enabled = self._policy is not SandboxPolicy.DISABLED
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._client = None
        self._available = False

        # Task-session pool. Lazily created on the first task; uses
        # SandboxPoolConfig.from_environment() so operators tune the capacity
        # cap via env vars without code changes. The lock guards the
        # build-once and drop-on-reconnect transitions — concurrent cold
        # tasks otherwise each build a pool and orphan the losers' live
        # gateway sessions.
        self._pool: Optional[Any] = None
        self._pool_lock = threading.Lock()
        # Serializes gateway dials: the cert-rotation tick and the exec-error
        # trigger can both reconnect at once; without this the loser's client
        # is overwritten and its channel never closed.
        self._connect_lock = threading.Lock()

        # Gateway circuit breaker: after a few failed dials it trips open and
        # a task lease fails fast (CircuitOpenError) instead of every request
        # eating the readiness budget, so one dead gateway can't stall the
        # worker pool.
        from cogniverse_core.common.utils.circuit_breaker import (
            BreakerConfig,
            CircuitBreaker,
        )

        self._gateway_breaker = CircuitBreaker.get(
            BreakerConfig(
                name="openshell_gateway",
                failure_threshold=3,
                reset_timeout_s=30.0,
            )
        )

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
    def _resolve_policy(policy: SandboxPolicy | str | None) -> SandboxPolicy:
        """Resolve the policy input into a SandboxPolicy, defaulting to
        ``OPTIONAL`` (degrade-with-warning) when none is given."""
        if policy is None:
            return SandboxPolicy.OPTIONAL
        if isinstance(policy, str):
            return SandboxPolicy(policy.lower())
        return policy

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

    def _resolve_tls_config(self):
        """Build the client mTLS config from certs mounted at
        ``<config-dir>/gateways/<name>/mtls/{ca,tls}.{crt,key}``, where the
        base honors ``OPENSHELL_CONFIG_DIR`` — the same resolution the cert
        rotator watches, so connect and rotation read one tree. Returns
        ``None`` (insecure channel) when no complete cert set is present, so
        a plaintext gateway still works."""
        import os

        from openshell import TlsConfig

        base = (
            Path(
                os.environ.get(
                    "OPENSHELL_CONFIG_DIR",
                    str(Path.home() / ".config" / "openshell"),
                )
            )
            / "gateways"
        )
        if not base.is_dir():
            return None
        for gw in sorted(base.iterdir()):
            mtls = gw / "mtls"
            ca, cert, key = mtls / "ca.crt", mtls / "tls.crt", mtls / "tls.key"
            if ca.is_file() and cert.is_file() and key.is_file():
                return TlsConfig(ca_path=ca, cert_path=cert, key_path=key)
        return None

    def _connect(self) -> None:
        """Connect to the OpenShell gateway.

        Prefers `OPENSHELL_GATEWAY_ENDPOINT` env var when set (for
        containerized deployments where the host gateway is reachable
        via host.docker.internal or similar). Falls back to the active
        cluster metadata (~/.config/openshell/gateways/<name>/metadata.json).
        """
        import os

        with self._connect_lock:
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
                    # The gateway serves mTLS; build the client TLS config from
                    # the certs mounted into the pod so the endpoint path isn't
                    # a plaintext channel talking to a TLS server. None =>
                    # insecure (backward compatible when no certs are mounted).
                    new_client = SandboxClient(
                        endpoint=override_endpoint, tls=self._resolve_tls_config()
                    )
                    logger.info(
                        f"Connected to OpenShell gateway at {override_endpoint}"
                    )
                else:
                    new_client = SandboxClient.from_active_cluster(
                        cluster=self._cluster
                    )
                    logger.info(
                        f"Connected to OpenShell gateway "
                        f"(cluster={self._cluster or 'active'})"
                    )
                displaced, self._client = self._client, new_client
                self._available = True
                self._drop_stale_pool()
            except Exception as e:
                logger.warning(
                    f"OpenShell gateway unavailable: {e}. "
                    "Agents will execute without sandbox isolation."
                )
                self._available = False
                return
        if displaced is not None:
            try:
                displaced.close()
            except Exception as exc:
                logger.debug("Displaced gateway client close: %s", exc)

    def _drop_stale_pool(self) -> None:
        """Close and forget a pool built on a previous client.

        A reconnect (cert rotation, gateway recovery) swaps ``self._client``;
        sessions the pool creates on the old client keep failing auth while
        the health probe reads the new client and reports green. Dropping the
        pool makes the next task rebuild it on the fresh client.
        """
        with self._pool_lock:
            stale, self._pool = self._pool, None
        if stale is not None:
            try:
                stale.close_all()
            except Exception as e:
                logger.debug("Stale sandbox pool close after reconnect: %s", e)

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

    def make_http_client(
        self,
        agent_type: str,
        *,
        endpoint_bindings: Optional[Dict[Any, Any]] = None,
        **client_kwargs: Any,
    ) -> Any:
        """Build an httpx.AsyncClient with policy enforcement for an agent.

        When an agent has a registered OpenShell policy, its outbound HTTP
        traffic is vetted against the policy's egress allow-list.
        Agents that do not have a registered policy fall through to a plain
        ``httpx.AsyncClient`` (no registered policy → no enforcement).

        ``endpoint_bindings`` (from ``sandbox_http.deployed_endpoint_bindings``)
        lets a policy rule for a service's default address admit the address
        the deployment configures for it.

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
        return make_policy_enforcing_client(
            policy, endpoint_bindings=endpoint_bindings, **client_kwargs
        )

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

    @asynccontextmanager
    async def task_session(self, agent_type: str, tenant_id: str):
        """Lease one fresh sandbox for a tenant's complete task."""
        from cogniverse_core.common.tenant_utils import require_tenant_id

        require_tenant_id(tenant_id, source="SandboxManager.task_session")
        if not await asyncio.to_thread(lambda: self.available):
            raise SandboxGatewayUnavailableError("Sandbox gateway is unavailable")
        pool = self._get_or_create_pool()
        lease = pool.task_session()
        acquisition = asyncio.create_task(asyncio.to_thread(lease.__enter__))
        try:
            session = await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            await _settle_sandbox_call(acquisition)
            await _run_sandbox_call(lease.__exit__, None, None, None)
            raise
        except Exception as exc:
            self._maybe_trigger_cert_rotator(exc)
            raise
        owned = SandboxTaskSession(
            session,
            agent_type,
            tenant_id,
            on_exec_error=self._maybe_trigger_cert_rotator,
        )
        try:
            yield owned
        finally:
            owned._closed = True
            await _run_sandbox_call(lease.__exit__, None, None, None)

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

        with self._pool_lock:
            if self._pool is not None:
                return self._pool
            cfg = SandboxPoolConfig.from_environment()
            pool = self._pool = SandboxSessionPool(
                self._client, config=cfg, gateway_breaker=self._gateway_breaker
            )
        logger.info(
            "Sandbox session pool initialised (max_size=%d)",
            cfg.max_pool_size,
        )
        return pool

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
        """Close the gateway connection and tear down live task sessions."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.close_all()
            except Exception as exc:
                logger.debug("Pool close_all failed (non-fatal): %s", exc)
        with self._connect_lock:
            client, self._client = self._client, None
            self._available = False
        if client is not None:
            client.close()
