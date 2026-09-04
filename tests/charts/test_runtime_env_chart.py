"""Unit tests for the runtime Deployment's instrumentation/tuning env vars.

``OPENINFERENCE_DSPY`` and ``ITER_RETRIEVAL_WALL_CLOCK_MS`` were previously
set only by patching the live Deployment; every redeploy silently dropped
them, killing DSPy LM span export and shrinking the iterative-retrieval
wall clock back to the 30s library default. These tests render the chart
with ``helm template`` and pin both env vars on the runtime container so
the wiring can only be removed deliberately.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"


pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render_chart(*set_args: str, values: str | tuple[str, ...] | None = None) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for values_file in (values,) if isinstance(values, str) else (values or ()):
        args += ["-f", str(CHART_PATH / values_file)]
    for s in set_args:
        args += ["--set", s]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _runtime_container_env(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-runtime Deployment, got {len(deployments)}"
    )
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    runtime = [c for c in containers if c["name"] == "runtime"]
    assert len(runtime) == 1, "runtime container missing from the Deployment"
    return {e["name"]: e.get("value") for e in runtime[0].get("env", [])}


def _ingestor_container_env(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-ingestor"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-ingestor Deployment, got {len(deployments)}"
    )
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    ingestor = [c for c in containers if c["name"] == "ingestor"]
    assert len(ingestor) == 1, "ingestor container missing from the Deployment"
    return {e["name"]: e.get("value") for e in ingestor[0].get("env", [])}


def _config_json(manifests: list) -> dict:
    configmaps = [
        m
        for m in manifests
        if m.get("kind") == "ConfigMap"
        and m.get("metadata", {}).get("name") == "cogniverse-config"
    ]
    assert len(configmaps) == 1, (
        f"Expected exactly one cogniverse-config ConfigMap, got {len(configmaps)}"
    )
    return json.loads(configmaps[0]["data"]["config.json"])


def _runtime_pod_spec(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-runtime Deployment, got {len(deployments)}"
    )
    return deployments[0]["spec"]["template"]["spec"]


def _runtime_container(manifests: list) -> dict:
    containers = _runtime_pod_spec(manifests)["containers"]
    runtime = [c for c in containers if c["name"] == "runtime"]
    assert len(runtime) == 1, "runtime container missing from the Deployment"
    return runtime[0]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIngestorMinioEnv:
    def test_ingestor_has_minio_default_bucket_matching_runtime(self):
        """The ingestor uploads extracted keyframes to object storage during
        ingestion, so it needs MINIO_DEFAULT_BUCKET — the same bucket the videos
        land in, so answer-time agents resolve keyframes from the hit's
        source_url bucket."""
        manifests = _render_chart()
        ingestor = _ingestor_container_env(manifests)
        runtime = _runtime_container_env(manifests)
        assert ingestor.get("MINIO_DEFAULT_BUCKET")
        assert ingestor["MINIO_DEFAULT_BUCKET"] == runtime.get("MINIO_DEFAULT_BUCKET")

    def test_ingestor_llm_env_and_config_track_primary_endpoint(self):
        manifests = _render_chart("llm.engine=vllm")
        ingestor = _ingestor_container_env(manifests)
        runtime = _runtime_container_env(manifests)
        config_json = _config_json(manifests)

        assert ingestor["LLM_ENDPOINT"] == "http://cogniverse-vllm-llm-student:8000/v1"
        assert ingestor["LLM_MODEL"] == "openai/google/gemma-4-e4b-it"
        assert config_json["llm_config"]["primary"]["api_base"] == (
            "http://cogniverse-vllm-llm-student:8000/v1"
        )
        assert config_json["llm_config"]["primary"]["model"] == (
            "openai/google/gemma-4-e4b-it"
        )
        assert runtime["SEMANTIC_ROUTER_URL"] == (
            "http://cogniverse-semantic-router-envoy:8801/v1"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRuntimeInstrumentationEnv:
    def test_openinference_dspy_enabled(self):
        env = _runtime_container_env(_render_chart())
        assert env.get("OPENINFERENCE_DSPY") == "1", (
            "OPENINFERENCE_DSPY must be '1' on the runtime container — without "
            "it DSPy LM spans are never instrumented and so never reach the "
            "tenant's Phoenix project"
        )

    def test_iter_retrieval_wall_clock_set_from_values(self):
        env = _runtime_container_env(_render_chart())
        assert env.get("ITER_RETRIEVAL_WALL_CLOCK_MS") == "120000", (
            "ITER_RETRIEVAL_WALL_CLOCK_MS must come from "
            "runtime.iterRetrieval.wallClockMs (default 120000) — the 30s "
            "library default hits wall_clock before max_iter on the "
            "in-cluster LM"
        )

    def test_iter_retrieval_wall_clock_override(self):
        env = _runtime_container_env(
            _render_chart("runtime.iterRetrieval.wallClockMs=45000")
        )
        assert env.get("ITER_RETRIEVAL_WALL_CLOCK_MS") == "45000"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRuntimeSandboxHostMode:
    def test_host_mode_renders_host_alias_and_host_mounts(self):
        manifests = _render_chart(
            "runtime.sandbox.enabled=true",
            "runtime.sandbox.inCluster.enabled=false",
            "runtime.sandbox.gatewayEndpoint=https://host.docker.internal:28080",
            "runtime.sandbox.hostGatewayIP=172.18.0.1",
        )
        spec = _runtime_pod_spec(manifests)
        assert spec["hostAliases"] == [
            {"ip": "172.18.0.1", "hostnames": ["host.docker.internal"]}
        ]

        runtime = _runtime_container(manifests)
        host_mounts = [
            mount
            for mount in runtime["volumeMounts"]
            if mount["name"].startswith("openshell")
        ]
        assert host_mounts == [
            {
                "name": "openshell-mtls",
                "mountPath": "/home/cogniverse/.config/openshell/gateways/cogniverse/mtls/ca.crt",
                "subPath": "ca.crt",
                "readOnly": True,
            },
            {
                "name": "openshell-mtls",
                "mountPath": "/home/cogniverse/.config/openshell/gateways/cogniverse/mtls/tls.crt",
                "subPath": "tls.crt",
                "readOnly": True,
            },
            {
                "name": "openshell-mtls",
                "mountPath": "/home/cogniverse/.config/openshell/gateways/cogniverse/mtls/tls.key",
                "subPath": "tls.key",
                "readOnly": True,
            },
            {
                "name": "openshell-metadata",
                "mountPath": "/home/cogniverse/.config/openshell/gateways/cogniverse/metadata.json",
                "subPath": "metadata.json",
                "readOnly": True,
            },
            {
                "name": "openshell-active",
                "mountPath": "/home/cogniverse/.config/openshell/active_gateway",
                "subPath": "active_gateway",
                "readOnly": True,
            },
        ]

    def test_disabled_mode_renders_no_host_alias_or_host_mounts(self):
        manifests = _render_chart()
        spec = _runtime_pod_spec(manifests)
        assert spec.get("hostAliases") is None
        runtime = _runtime_container(manifests)
        host_mounts = [
            mount
            for mount in runtime["volumeMounts"]
            if mount["name"].startswith("openshell")
        ]
        assert host_mounts == []

    @pytest.mark.parametrize(
        "sandbox_override",
        [
            "runtime.sandbox.inCluster.enabled=true",
            "runtime.sandbox.external.enabled=true",
        ],
    )
    def test_non_host_sandbox_modes_do_not_render_host_aliases(
        self, sandbox_override: str
    ):
        extra = [sandbox_override]
        if sandbox_override.endswith("external.enabled=true"):
            extra.append("runtime.sandbox.external.endpoint=openshell.example.com:8080")
        manifests = _render_chart(
            "runtime.sandbox.enabled=true",
            *extra,
            "runtime.sandbox.hostGatewayIP=172.18.0.1",
        )
        spec = _runtime_pod_spec(manifests)
        assert spec.get("hostAliases") is None
        runtime = _runtime_container(manifests)
        host_mounts = [
            mount
            for mount in runtime["volumeMounts"]
            if mount["name"].startswith("openshell-mtls")
            or mount["name"] in {"openshell-metadata", "openshell-active"}
        ]
        assert host_mounts == []


def _render_with_values(*values_files: str) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for f in values_files:
        args += ["-f", str(CHART_PATH / f)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _dev_mount_deployments(manifests: list) -> list[str]:
    names = []
    for m in manifests:
        if m.get("kind") != "Deployment":
            continue
        for vol in (
            m.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
            or []
        ):
            if vol.get("name") == "src-libs":
                names.append(m["metadata"]["name"])
    return sorted(names)


class TestDeviceOverlaysKeepDevMode:
    """The device overlays layer on top of values.k3s.yaml in `cogniverse up`
    (base first, device file second — later files win per key). A devMode
    key in a device overlay silently disables the k3s dev-source mounts on
    that device's hosts, killing the edit→restart loop."""

    def test_k3s_plus_cuda_keeps_dev_mounts(self):
        manifests = _render_with_values("values.k3s.yaml", "values.cuda.yaml")
        assert _dev_mount_deployments(manifests) == [
            "cogniverse-dashboard",
            "cogniverse-quality-monitor",
            "cogniverse-runtime",
        ]

    def test_base_plus_cuda_stays_non_dev(self):
        manifests = _render_with_values("values.cuda.yaml")
        assert _dev_mount_deployments(manifests) == []

    def test_k3s_plus_rocm_keeps_dev_mounts(self):
        manifests = _render_with_values("values.k3s.yaml", "values.rocm.yaml")
        assert _dev_mount_deployments(manifests) == [
            "cogniverse-dashboard",
            "cogniverse-quality-monitor",
            "cogniverse-runtime",
        ]


class TestWorkflowApiUrlTargetsTheDeployedArgo:
    """``WORKFLOW_API_URL`` must name an Argo server the release actually has.

    The in-release ``argo-workflows`` subchart is gated on
    ``argo.subchart.enabled`` and OFF by default — Chart.yaml calls it "a
    redundant second install" because ``cogniverse up`` manages a standalone
    Argo at argo-server.argo.svc. Addressing the subchart unconditionally left
    the runtime pointing at a Service that does not exist in the default
    configuration, so every scheduled-job submission failed DNS resolution and
    the route answered 503 ("Argo unreachable while scheduling job ...
    Name or service not known").
    """

    def test_default_targets_the_standalone_argo_install(self):
        env = _runtime_container_env(_render_chart())

        assert (
            env["WORKFLOW_API_URL"] == "https://argo-server.argo.svc.cluster.local:2746"
        )
        assert env["WORKFLOW_NAMESPACE"] == "default"

    def test_enabling_the_subchart_targets_the_in_release_server(self):
        env = _runtime_container_env(_render_chart("argo.subchart.enabled=true"))

        assert env["WORKFLOW_API_URL"] == (
            "http://cogniverse-argo-workflows-server.default.svc.cluster.local:2746"
        )

    def test_the_addressed_host_matches_the_cronworkflow_submission_target(self):
        """The runtime and the chart's CronWorkflows must submit to one Argo."""
        manifests = _render_chart()
        env = _runtime_container_env(manifests)
        rendered = yaml.safe_dump_all(manifests)

        host = env["WORKFLOW_API_URL"].split("://", 1)[1]
        assert "argo-server.argo.svc.cluster.local:2746" == host
        assert f"https://{host}" in rendered


def _runtime_container_env_entries(manifests: list) -> list[dict]:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    runtime = [c for c in containers if c["name"] == "runtime"]
    assert len(runtime) == 1
    return runtime[0].get("env", [])


def _ingestor_container_env_entries(manifests: list) -> list[dict]:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-ingestor"
    ]
    assert len(deployments) == 1
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    ingestor = [c for c in containers if c["name"] == "ingestor"]
    assert len(ingestor) == 1
    return ingestor[0].get("env", [])


@pytest.mark.unit
@pytest.mark.ci_fast
class TestInferenceApiKeyDelivery:
    """Pods that dial inference endpoints use the shared Modal bearer when
    any enabled inference service is external. Fully in-cluster renders keep
    the no-auth placeholder so the stack still starts without the Secret."""

    PLACEHOLDER_ENTRY = {
        "name": "COGNIVERSE_INFERENCE_API_KEY",
        "value": "placeholder-no-auth-needed",
    }
    SECRET_ENTRY = {
        "name": "COGNIVERSE_INFERENCE_API_KEY",
        "valueFrom": {
            "secretKeyRef": {
                "name": "cogniverse-inference-api-key",
                "key": "COGNIVERSE_INFERENCE_API_KEY",
                "optional": False,
            }
        },
    }

    @staticmethod
    def _api_key_entry(entries: list[dict]) -> dict:
        matches = [e for e in entries if e["name"] == "COGNIVERSE_INFERENCE_API_KEY"]
        assert len(matches) == 1
        return matches[0]

    def test_runtime_and_ingestor_receive_the_placeholder_when_teacher_stays_cluster_local(
        self,
    ):
        manifests = _render_chart(values="values.rocm.yaml")

        assert self._api_key_entry(_runtime_container_env_entries(manifests)) == (
            self.PLACEHOLDER_ENTRY
        )
        assert self._api_key_entry(_ingestor_container_env_entries(manifests)) == (
            self.PLACEHOLDER_ENTRY
        )

    def test_runtime_and_ingestor_switch_to_the_synced_secret_for_an_external_teacher(
        self,
    ):
        manifests = _render_chart(
            "inference.vllm_llm_teacher.externalUrl=https://example.modal.run",
            values="values.rocm.yaml",
        )

        assert self._api_key_entry(_runtime_container_env_entries(manifests)) == (
            self.SECRET_ENTRY
        )
        assert self._api_key_entry(_ingestor_container_env_entries(manifests)) == (
            self.SECRET_ENTRY
        )


def _cogniverse_app_containers(manifests: list) -> dict[str, dict]:
    """Every container running first-party cogniverse application code.

    Derived from the render rather than a restated list, so a Deployment added
    later is covered without editing this test. Sidecars built from third-party
    images (vLLM, redis, minio, phoenix) are excluded: they do not import
    cogniverse code and have no reason to carry its configuration.
    """
    found: dict[str, dict] = {}
    for m in manifests:
        if m.get("kind") != "Deployment":
            continue
        name = m["metadata"]["name"]
        for c in m["spec"]["template"]["spec"]["containers"]:
            image = c.get("image", "")
            if not image.startswith("cogniverse/"):
                continue
            if "/pylate" in image or "/clap" in image or "/gliner" in image:
                continue  # model servers, not application code
            found[f"{name}/{c['name']}"] = {
                e["name"]: e.get("value") for e in c.get("env", [])
            }
    return found


def test_every_cogniverse_app_container_gets_redis_url():
    """REDIS_URL reaches every container running cogniverse application code.

    The dashboard shipped without it. cogniverse_dashboard.tabs.approval_queue
    raises ValueError("REDIS_URL is required for approval item replacement"),
    which aborted the Synthetic Data tab's render partway through, so its
    primary "Generate Synthetic Data" button never appeared -- measured live:
    the button existed 0 times inside that panel while the panel's text ended
    at the error.
    """
    manifests = _render_chart("redis.enabled=true")
    containers = _cogniverse_app_containers(manifests)
    assert containers, "no cogniverse application containers found in the render"
    missing = sorted(n for n, env in containers.items() if "REDIS_URL" not in env)
    assert missing == [], (
        f"cogniverse application containers without REDIS_URL: {missing}; "
        f"inspected {sorted(containers)}"
    )


def test_no_container_declares_the_same_env_var_twice():
    """No container repeats an env var name.

    Kubernetes keeps the last occurrence silently, so a duplicate hides which
    value actually applies. Factoring REDIS_URL into a shared template briefly
    produced exactly that on the runtime container: the include emitted it
    alongside the inline block it was meant to replace.
    """
    manifests = _render_chart("redis.enabled=true")
    offenders = {}
    for m in manifests:
        if m.get("kind") != "Deployment":
            continue
        for c in m["spec"]["template"]["spec"]["containers"]:
            names = [e["name"] for e in c.get("env", [])]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                offenders[f"{m['metadata']['name']}/{c['name']}"] = dupes
    assert offenders == {}, f"containers declaring an env var twice: {offenders}"
