"""The e2e k3d cluster: its name, kube context, host ports and seeded tenant."""

from __future__ import annotations

from pathlib import Path

import httpx
import yaml
from cogniverse_cli.argo import (
    ARGO_NAMESPACE,
    ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
)

E2E_CLUSTER_NAME = "cogniverse-e2e"
KUBECTL_CONTEXT = f"k3d-{E2E_CLUSTER_NAME}"

# Host-side loadbalancer ports of the e2e cluster. The right-hand side is
# the chart's canonical NodePort (unchanged); the host side is offset into
# 33xxx so the e2e stack never collides with a dev cluster's 28xxx/8080
# mappings or the 29xxx test-sidecar range. Every localhost URL in this
# suite uses the 33xxx side.
E2E_HOST_PORTS = {
    33080: 8080,  # vespa http
    33071: 19071,  # vespa config
    33000: 28000,  # runtime
    33400: 28400,  # web
    33501: 28501,  # dashboard
    33006: 26006,  # phoenix ui
    33317: 4317,  # otel grpc
    33434: 11434,  # llm (ollama-compat)
    33746: 2746,  # argo server
    33881: 28081,  # semantic-router envoy
    33901: 29001,  # inference sidecars
    33902: 29002,
    33903: 29003,
    33904: 29004,
    33905: 29005,
    33906: 29006,
    33907: 29007,
    33908: 29008,
    33909: 29009,
    33910: 29010,
    33911: 29011,
    33912: 29012,  # video_embed (X-CLIP)
}

# k3d NodePort URLs — defined in charts/cogniverse/values.yaml
RUNTIME = "http://localhost:33000"  # runtime.service.nodePort
GLINER_URL = "http://localhost:33907"  # gliner NodePort 29007 via E2E_HOST_PORTS
TENANT_DEPLOY_TIMEOUT_S = 180.0
"""Budget for one tenant create or profile deploy. Both recompile the whole
Vespa application package and wait for convergence, so they scale with the
cluster's schema count. Measured: 33.3 s, 35.9 s, 43.3 s idle; 86.7 s under
sweep load; the convergence gate alone may hold up to 120 s."""
K3S_VALUES = Path(__file__).resolve().parents[2] / "charts/cogniverse/values.k3s.yaml"


def seeded_tenant_id(values_path: Path = K3S_VALUES) -> str:
    """The tenant the e2e cluster is seeded with: the k3s overlay's
    quality-monitor tenant, so the session seeds the tenant the chart's own
    CronJobs run against."""
    values = yaml.safe_load(values_path.read_text())
    tenant_id = values["runtime"]["qualityMonitor"]["tenantId"]
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError(f"{values_path} sets no runtime.qualityMonitor.tenantId")
    return tenant_id


TENANT_ID = seeded_tenant_id()
IN_POD_TELEMETRY_PRELUDE = (
    "from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults; "
    "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
    "get_telemetry_manager(otlp_endpoint=resolve_library_env_defaults()['telemetry_otlp_endpoint']); "
)


def runtime_available() -> bool:
    # /health/live is cheap; /health does backend + registry lookups and
    # can block under LLM load, producing false-negative skips.
    try:
        r = httpx.get(f"{RUNTIME}/health/live", timeout=30.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
        return False


def argo_workflow_controller_probe_command(
    namespace: str = ARGO_NAMESPACE,
) -> list[str]:
    return [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "-n",
        namespace,
        "get",
        "pods",
        "-l",
        ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
        "--field-selector=status.phase=Running",
        "-o",
        "name",
    ]


def argo_workflow_controller_probe_failure_message(
    *,
    command: list[str],
    namespace: str = ARGO_NAMESPACE,
) -> str:
    return (
        "Argo workflow controller unavailable after E2E stack setup; "
        f"command={' '.join(command)!r}; context={KUBECTL_CONTEXT!r}; "
        f"namespace={namespace!r}; "
        f"selector={ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR!r}"
    )
