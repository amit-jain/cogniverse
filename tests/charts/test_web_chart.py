"""The web client is the release's UI: its Deployment, Service and probes,
the ingress rule that serves it, and the host port the local cluster publishes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from cogniverse_cli.cluster import DEFAULT_PORTS

from tests.e2e.cluster import E2E_HOST_PORTS

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
APP_VERSION = str(yaml.safe_load((CHART_PATH / "Chart.yaml").read_text())["appVersion"])
PROD_SECRETS = (
    "minio.rootPassword=x",
    "openshell.server.sshHandshakeSecret=x",
    "phoenix.postgres.auth.password=x",
    "redis.auth.password=x",
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed"),
]


def _helm(*set_args: str, values: str | None = None) -> subprocess.CompletedProcess:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    if values is not None:
        command += ["-f", str(CHART_PATH / values)]
    for value in set_args:
        command += ["--set", value]
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def _render(*set_args: str, values: str | None = None) -> list[dict]:
    result = _helm(*set_args, values=values)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _named(docs: list[dict], kind: str, name: str) -> dict:
    matches = [d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name]
    assert len(matches) == 1, f"{kind}/{name}: {len(matches)} rendered"
    return matches[0]


def _container(docs: list[dict], deployment: str, name: str) -> dict:
    spec = _named(docs, "Deployment", deployment)["spec"]["template"]["spec"]
    return next(c for c in spec["containers"] if c["name"] == name)


def _names(docs: list[dict], kind: str) -> set[str]:
    return {d["metadata"]["name"] for d in docs if d["kind"] == kind}


def test_the_web_deployment_serves_its_port_with_health_probes():
    docs = _render()
    web = _container(docs, "cogniverse-web", "web")

    assert web["image"] == f"cogniverse/web:{APP_VERSION}"
    assert web["imagePullPolicy"] == "IfNotPresent"
    assert web["ports"] == [{"name": "http", "containerPort": 4000}]
    probe = {"httpGet": {"path": "/healthz", "port": "http"}}
    assert {key: web["livenessProbe"][key] for key in probe} == probe
    assert {key: web["readinessProbe"][key] for key in probe} == probe
    assert web["resources"] == {
        "limits": {"cpu": "1", "memory": "512Mi"},
        "requests": {"cpu": "100m", "memory": "512Mi"},
    }
    assert web["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    assert web["volumeMounts"] == [{"name": "tmp", "mountPath": "/tmp"}]
    service = _named(docs, "Service", "cogniverse-web")["spec"]
    assert service["type"] == "ClusterIP"
    assert service["ports"] == [
        {"port": 4000, "targetPort": "http", "protocol": "TCP", "name": "http"}
    ]
    assert service["selector"] == {
        "app.kubernetes.io/name": "cogniverse",
        "app.kubernetes.io/instance": "cogniverse",
        "app.kubernetes.io/component": "web",
    }


def test_the_server_holds_no_key_and_calls_the_runtime_service_directly():
    """The server mints each tenant's harness key through the runtime's
    /admin/harness/keys, so the chart gives it no key: only the runtime's
    Service address, which serves /admin without the ingress."""
    docs = _render()
    web = _container(docs, "cogniverse-web", "web")
    runtime = _container(docs, "cogniverse-runtime", "runtime")

    assert web["env"] == [
        {"name": "COGNIVERSE_RUNTIME_URL", "value": "http://cogniverse-runtime:8000"},
        {"name": "HOST", "value": "0.0.0.0"},
        {"name": "PORT", "value": "4000"},
    ]
    assert "cogniverse-web" not in _names(docs, "Secret")
    assert "envFrom" not in web
    assert "COGNIVERSE_HARNESS_API_KEY" not in {e["name"] for e in runtime["env"]}
    runtime_service = _named(docs, "Service", "cogniverse-runtime")["spec"]
    assert 8000 in [port["port"] for port in runtime_service["ports"]]


def test_an_operator_harness_key_reaches_the_runtime_beside_the_web_client():
    """COGNIVERSE_HARNESS_API_KEY is the key files/config.json maps to the
    default tenant for harness clients such as the pi extension; the web
    client does not claim it."""
    docs = _render("runtime.env.COGNIVERSE_HARNESS_API_KEY=sk-pi")
    runtime = _container(docs, "cogniverse-runtime", "runtime")

    assert [e for e in runtime["env"] if e["name"] == "COGNIVERSE_HARNESS_API_KEY"] == [
        {"name": "COGNIVERSE_HARNESS_API_KEY", "value": "sk-pi"}
    ]
    assert "cogniverse-web" in _names(docs, "Deployment")
    web = _container(docs, "cogniverse-web", "web")
    assert [e["name"] for e in web["env"]] == ["COGNIVERSE_RUNTIME_URL", "HOST", "PORT"]


def test_extra_env_and_env_sources_reach_the_server():
    docs = _render(
        "web.env.WEB_SESSION_TTL=3600",
        "web.envFrom[0].secretRef.name=web-auth",
        "web.runtimeUrl=http://runtime.example:8000",
    )
    web = _container(docs, "cogniverse-web", "web")

    assert {e["name"]: e.get("value") for e in web["env"] if "value" in e} == {
        "COGNIVERSE_RUNTIME_URL": "http://runtime.example:8000",
        "HOST": "0.0.0.0",
        "PORT": "4000",
        "WEB_SESSION_TTL": "3600",
    }
    assert web["envFrom"] == [{"secretRef": {"name": "web-auth"}}]


@pytest.mark.parametrize(
    "setting",
    [
        "web.env.COGNIVERSE_RUNTIME_URL=http://x",
        "web.env.HOST=127.0.0.1",
        "web.env.PORT=1",
    ],
)
def test_chart_owned_variables_are_refused_in_extra_env(setting):
    result = _helm(setting)
    name = setting.split("=", 1)[0].removeprefix("web.env.")

    assert result.returncode != 0
    assert (
        f"web.env.{name} is set by the chart; use web.runtimeUrl or web.service.port"
    ) in result.stderr


def test_prod_renders_the_web_client_without_a_key():
    docs = _render(*PROD_SECRETS, values="values.prod.yaml")

    assert "cogniverse-web" in _names(docs, "Deployment")
    assert "cogniverse-web" not in _names(docs, "Secret")
    assert _named(docs, "Service", "cogniverse-web")["spec"]["ports"] == [
        {"port": 4000, "targetPort": "http", "protocol": "TCP", "name": "http"}
    ]
    assert [e["name"] for e in _container(docs, "cogniverse-web", "web")["env"]] == [
        "COGNIVERSE_RUNTIME_URL",
        "HOST",
        "PORT",
    ]


def test_disabling_the_web_client_removes_it():
    docs = _render("web.enabled=false")
    runtime = _container(docs, "cogniverse-runtime", "runtime")

    assert not any(d["metadata"]["name"] == "cogniverse-web" for d in docs)
    assert "COGNIVERSE_HARNESS_API_KEY" not in {e["name"] for e in runtime["env"]}
    ingress = _named(docs, "Ingress", "cogniverse")
    assert [
        (path["path"], path["backend"]["service"]["name"])
        for rule in ingress["spec"]["rules"]
        for path in rule["http"]["paths"]
    ] == [("/api", "cogniverse-runtime")]


@pytest.mark.parametrize("values", [None, "values.k3s.yaml"])
def test_the_dashboard_is_off_by_default_and_still_deployable(values):
    default = _render(values=values)
    enabled = _render("dashboard.enabled=true", values=values)

    assert "cogniverse-dashboard" not in _names(default, "Deployment")
    assert "cogniverse-dashboard" not in _names(default, "Service")
    assert "cogniverse-dashboard" in _names(enabled, "Deployment")
    assert "cogniverse-dashboard" in _names(enabled, "Service")


@pytest.mark.parametrize(
    ("values", "host"),
    [
        (None, "cogniverse.example.com"),
        ("values.k3s.yaml", "cogniverse.local"),
        ("values.prod.yaml", "cogniverse.example.com"),
    ],
)
def test_every_ingress_serves_the_web_client_at_the_root(values, host):
    extra = PROD_SECRETS if values == "values.prod.yaml" else ()
    docs = _render(*extra, values=values)
    ingress = _named(docs, "Ingress", "cogniverse")
    services = {
        d["metadata"]["name"]: [p["port"] for p in d["spec"]["ports"]]
        for d in docs
        if d["kind"] == "Service"
    }
    routes = [
        (
            path["path"],
            path["backend"]["service"]["name"],
            path["backend"]["service"]["port"]["number"],
        )
        for rule in ingress["spec"]["rules"]
        for path in rule["http"]["paths"]
    ]

    assert [rule["host"] for rule in ingress["spec"]["rules"]] == [host]
    assert routes == [
        ("/api", "cogniverse-runtime", 8000),
        ("/", "cogniverse-web", 4000),
    ]
    for _, service, port in routes:
        assert port in services[service], f"{service} does not expose {port}"


def test_nginx_streams_server_sent_events_unbuffered():
    ingress = _named(_render(), "Ingress", "cogniverse")

    assert (
        ingress["metadata"]["annotations"][
            "nginx.ingress.kubernetes.io/proxy-buffering"
        ]
        == "off"
    )


def test_k3s_publishes_the_web_client_on_the_clusters_host_ports():
    docs = _render(values="values.k3s.yaml")
    web = _container(docs, "cogniverse-web", "web")
    service = _named(docs, "Service", "cogniverse-web")["spec"]

    assert web["image"] == f"cogniverse/web:{APP_VERSION}-dev"
    assert web["imagePullPolicy"] == "Never"
    assert service["type"] == "NodePort"
    assert service["ports"][0]["nodePort"] == 28400
    assert 28400 in DEFAULT_PORTS
    assert {host for host, node in E2E_HOST_PORTS.items() if node == 28400} == {33400}
