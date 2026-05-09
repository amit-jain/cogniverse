"""helm chart applies a runtime-egress NetworkPolicy that selects
the actual runtime pod.

Audit caught: the previous emit-mode generated 5 per-agent
NetworkPolicies selecting on ``cogniverse-agent: <name>``, but the
unified-runtime Deployment carries no such label. Enabling
``agentEgress.enabled=true`` produced 5 NetworkPolicies that targeted
zero pods.

Now the egress-netpol CLI runs in unified mode (``--unified-pod-selector
app.kubernetes.io/component=runtime``) and emits ONE NetworkPolicy
selecting on the runtime Deployment's actual labels with the
de-duplicated UNION of every agent's egress destinations. Per-agent
isolation lives at the application layer via OpenShell sandbox; the
NetworkPolicy is defense-in-depth at L4.

This test verifies, against a real ``helm template`` invocation:

  * the chart's templates/agent-egress/ directory is the unified shape:
    one runtime-egress-netpol.yaml, no stale per-agent files;
  * helm renders the chart with the flag off → no NetworkPolicy emitted;
  * helm renders the chart with the flag on → exactly one runtime
    NetworkPolicy with the right selector labels;
  * a Deployment in the same chart actually CARRIES those labels — i.e.
    the NetworkPolicy will select something. This is the missing
    assertion that caused the original audit gap.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


HELM = shutil.which("helm")
CHART_PATH = Path("charts/cogniverse")
TEMPLATES_DIR = CHART_PATH / "templates" / "agent-egress"
EXPECTED_NETPOL_NAME = "cogniverse-runtime-egress"
EXPECTED_SELECTOR = {"app.kubernetes.io/component": "runtime"}


def _helm_template(values_overrides: dict | None = None) -> str:
    """Run `helm template` against the cogniverse chart and return stdout."""
    assert HELM is not None, "helm not on PATH; install it for this test"
    assert CHART_PATH.exists(), f"chart not found at {CHART_PATH}"
    baseline = {
        "runtime.qualityMonitor.tenantId": "test_tenant",
    }
    merged = {**baseline, **(values_overrides or {})}
    cmd = [HELM, "template", "test-release", str(CHART_PATH)]
    for k, v in merged.items():
        cmd += ["--set", f"{k}={v}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}): "
            f"stderr={result.stderr[-2000:]}"
        )
    return result.stdout


def _parse_yaml_documents(rendered: str) -> list:
    """Split helm output into individual yaml documents and parse them."""
    docs = []
    for raw in re.split(r"^---\s*$", rendered, flags=re.MULTILINE):
        try:
            doc = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and doc:
            docs.append(doc)
    return docs


class TestUnifiedYAMLOnDisk:
    def test_only_runtime_egress_yaml_present(self):
        files = sorted(p.name for p in TEMPLATES_DIR.glob("*.yaml"))
        assert files == ["runtime-egress-netpol.yaml"], (
            "templates/agent-egress/ must contain exactly the unified "
            f"runtime-egress-netpol.yaml; got {files}. Stale per-agent "
            "files (search_agent-..., coding_agent-..., etc.) target a "
            "non-existent pod label and must not be committed."
        )

    def test_runtime_yaml_carries_helm_conditional(self):
        path = TEMPLATES_DIR / "runtime-egress-netpol.yaml"
        content = path.read_text(encoding="utf-8")
        assert "{{- if " in content, (
            f"{path.name} missing helm conditional wrapper; the "
            "values.yaml flag won't gate the policy"
        )
        assert "agentEgress.enabled" in content, (
            f"{path.name} missing the agentEgress.enabled gate"
        )
        assert "{{- end }}" in content, f"{path.name} missing closing `{{- end }}`"


class TestHelmRendersChartCorrectly:
    def test_default_values_omit_network_policies(self):
        rendered = _helm_template()
        docs = _parse_yaml_documents(rendered)
        netpols = [d for d in docs if d.get("kind") == "NetworkPolicy"]
        runtime_netpols = [
            d
            for d in netpols
            if str(d.get("metadata", {}).get("name", "")).endswith("-egress")
        ]
        assert runtime_netpols == [], (
            "with networkPolicy.agentEgress.enabled=false (the default), "
            "no agent-egress NetworkPolicy resources should be rendered; "
            f"got {[d['metadata']['name'] for d in runtime_netpols]}"
        )

    def test_enabled_flag_renders_runtime_netpol(self):
        rendered = _helm_template({"networkPolicy.agentEgress.enabled": "true"})
        docs = _parse_yaml_documents(rendered)
        runtime_netpols = [
            d
            for d in docs
            if d.get("kind") == "NetworkPolicy"
            and d.get("metadata", {}).get("name") == EXPECTED_NETPOL_NAME
        ]
        assert len(runtime_netpols) == 1, (
            f"expected exactly one {EXPECTED_NETPOL_NAME} NetworkPolicy; "
            f"got {len(runtime_netpols)}: {[d['metadata']['name'] for d in runtime_netpols]}"
        )
        np = runtime_netpols[0]
        labels = np["spec"]["podSelector"]["matchLabels"]
        for k, v in EXPECTED_SELECTOR.items():
            assert labels.get(k) == v, (
                f"runtime NetworkPolicy podSelector missing expected "
                f"{k}={v}; got {labels!r}"
            )
        assert np["spec"]["policyTypes"] == ["Egress"]
        assert isinstance(np["spec"]["egress"], list) and len(np["spec"]["egress"]) >= 1

    def test_runtime_netpol_targets_correct_namespace(self):
        rendered = _helm_template({"networkPolicy.agentEgress.enabled": "true"})
        docs = _parse_yaml_documents(rendered)
        for d in docs:
            if d.get("kind") != "NetworkPolicy":
                continue
            name = d["metadata"]["name"]
            if not name.endswith("-egress"):
                continue
            assert d["metadata"]["namespace"] == "cogniverse", (
                f"{name} rendered into namespace={d['metadata']['namespace']!r}; "
                "expected 'cogniverse' from the chart's default namespace"
            )


class TestNetpolActuallySelectsAPod:
    """The audit-blocking gap: the previous YAMLs targeted pod labels
    that no Deployment carried. This test asserts a Deployment in the
    chart carries every label the NetworkPolicy selects on, so the
    NetworkPolicy actually selects pods at runtime.
    """

    def test_runtime_deployment_has_selector_labels(self):
        rendered = _helm_template({"networkPolicy.agentEgress.enabled": "true"})
        docs = _parse_yaml_documents(rendered)
        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        assert deployments, "chart must render at least one Deployment"

        # Find every pod-template label set on every Deployment.
        all_pod_labels = []
        for d in deployments:
            tmpl = d["spec"]["template"]["metadata"].get("labels", {}) or {}
            all_pod_labels.append((d["metadata"]["name"], tmpl))

        def _matches_selector(labels: dict, selector: dict) -> bool:
            return all(labels.get(k) == v for k, v in selector.items())

        targeted = [
            (name, labels)
            for name, labels in all_pod_labels
            if _matches_selector(labels, EXPECTED_SELECTOR)
        ]
        assert targeted, (
            f"NetworkPolicy {EXPECTED_NETPOL_NAME!r} selects on "
            f"{EXPECTED_SELECTOR!r} but no Deployment in the chart "
            "applies all of those labels to its pod template. The "
            "NetworkPolicy would target zero pods at runtime, the same "
            "audit gap that caused this rewrite. Deployments and their "
            "pod-template labels: "
            + ", ".join(f"{name}={labels!r}" for name, labels in all_pod_labels)
        )


class TestStaleness:
    def test_chart_yaml_matches_current_agent_policies(self, tmp_path: Path):
        """Re-run the CLI in unified mode and diff against the committed YAML.

        Catches drift: if someone updates configs/agent_policies/*.yaml
        without re-running the CLI, this fails so the operator knows
        to regenerate.
        """
        out = tmp_path / "fresh"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "cogniverse_runtime.optimization_cli",
                "--mode",
                "egress-netpol",
                "--tenant-id",
                "any",
                "--policy-dir",
                "configs/agent_policies",
                "--output-dir",
                str(out),
                "--service-map",
                "vespa=cogniverse/vespa-service:8080",
                "--service-map",
                "ollama=cogniverse/ollama-service:11434",
                "--service-map",
                "runtime=cogniverse/cogniverse-runtime:8000",
                "--netpol-namespace",
                "cogniverse",
                "--helm-conditional",
                ".Values.networkPolicy.agentEgress.enabled",
                "--unified-pod-selector",
                "app.kubernetes.io/component=runtime",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"egress-netpol CLI failed in subprocess: stderr={result.stderr[-2000:]}"
        )
        for fresh_path in out.glob("*.yaml"):
            committed_path = TEMPLATES_DIR / fresh_path.name
            assert committed_path.exists(), (
                f"new YAML {fresh_path.name} from CLI but not committed "
                "to the chart — re-run the CLI and commit"
            )
            fresh_content = fresh_path.read_text(encoding="utf-8")
            committed_content = committed_path.read_text(encoding="utf-8")
            assert fresh_content == committed_content, (
                f"committed {fresh_path.name} differs from a fresh CLI "
                "run — re-run `cogniverse-optim --mode egress-netpol "
                "--unified-pod-selector app.kubernetes.io/component=runtime "
                "...` and commit"
            )
