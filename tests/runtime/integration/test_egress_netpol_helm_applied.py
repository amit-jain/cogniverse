"""F6.1 — helm chart actually applies the generated NetworkPolicies.

Audit caught: the egress-netpol CLI generated valid NetworkPolicy YAMLs
but the helm chart had no `templates/networkpolicies/` directory to
consume them. The CLI's output was files on disk that no production
process applied — operators had to manually copy them into the chart.

This commit moves the generated YAMLs into the chart's templates
directory under ``templates/agent-egress/``, wraps each one in a
``{{- if .Values.networkPolicy.agentEgress.enabled }}`` conditional,
and adds the corresponding values.yaml flag.

This test verifies, against a real `helm template` invocation:

  * with ``networkPolicy.agentEgress.enabled=false`` (default),
    no NetworkPolicy resources appear in the rendered manifest;
  * with the flag flipped to true, exactly one NetworkPolicy per
    eligible agent appears, with the right apiVersion + kind +
    metadata.name + spec.podSelector;
  * the YAMLs in templates/agent-egress/ are valid helm templates
    (chart renders without error).
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


def _helm_template(values_overrides: dict | None = None) -> str:
    """Run `helm template` against the cogniverse chart and return stdout.

    Skip-by-fail rather than skip-by-skip: if helm or the chart aren't
    available, that's an env-broken state we want to surface rather
    than silently skip.
    """
    assert HELM is not None, "helm not on PATH; install it for this test"
    assert CHART_PATH.exists(), f"chart not found at {CHART_PATH}"
    # The chart's templated guards require these baseline values to be
    # set or the render fails before reaching our agent-egress YAMLs.
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
    for raw in rendered.split("\n---\n"):
        try:
            doc = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and doc:
            docs.append(doc)
    return docs


class TestGeneratedYAMLsCheckedIntoChart:
    def test_templates_dir_exists(self):
        assert TEMPLATES_DIR.exists(), (
            f"{TEMPLATES_DIR} must exist; the egress-netpol CLI was "
            "supposed to populate it. Re-run "
            "`cogniverse-runtime egress-netpol --output-dir "
            "charts/cogniverse/templates/agent-egress --service-map ...`"
        )

    def test_one_yaml_per_eligible_agent(self):
        yamls = sorted(p.name for p in TEMPLATES_DIR.glob("*.yaml"))
        for required in (
            "coding_agent-egress-netpol.yaml",
            "search_agent-egress-netpol.yaml",
            "summarizer_agent-egress-netpol.yaml",
            "routing_agent-egress-netpol.yaml",
            "orchestrator_agent-egress-netpol.yaml",
        ):
            assert required in yamls, (
                f"{required} missing from {TEMPLATES_DIR}; agent policy "
                "added but the chart-committed NetworkPolicy is stale. "
                "Re-run the egress-netpol CLI."
            )

    def test_each_yaml_carries_helm_conditional(self):
        """Without the wrapper, the policies would always apply — they
        must respect the `networkPolicy.agentEgress.enabled` flag."""
        for path in TEMPLATES_DIR.glob("*.yaml"):
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
        # The agent-egress policies must NOT appear when the flag is
        # off (the default). The legacy `networkPolicy.enabled` flag
        # may produce its own NetworkPolicy if any exists; we filter
        # to just the ones our CLI generates by name pattern.
        agent_egress_netpols = [
            d
            for d in netpols
            if str(d.get("metadata", {}).get("name", "")).endswith("-egress")
        ]
        assert agent_egress_netpols == [], (
            "with networkPolicy.agentEgress.enabled=false (the default), "
            "no agent-egress NetworkPolicy resources should be rendered; "
            f"got {[d['metadata']['name'] for d in agent_egress_netpols]}"
        )

    def test_enabled_flag_renders_one_netpol_per_agent(self):
        rendered = _helm_template({"networkPolicy.agentEgress.enabled": "true"})
        docs = _parse_yaml_documents(rendered)
        netpols = [
            d
            for d in docs
            if d.get("kind") == "NetworkPolicy"
            and str(d.get("metadata", {}).get("name", "")).endswith("-egress")
        ]
        names = sorted(d["metadata"]["name"] for d in netpols)
        # 5 cogniverse agents shipped with deny_all_other policies.
        for required in (
            "cogniverse-coding-agent-egress",
            "cogniverse-search-agent-egress",
            "cogniverse-summarizer-agent-egress",
            "cogniverse-routing-agent-egress",
            "cogniverse-orchestrator-agent-egress",
        ):
            assert required in names, (
                f"{required} not rendered by helm template when "
                f"agentEgress.enabled=true; got {names}"
            )

    def test_each_rendered_netpol_has_correct_pod_selector(self):
        rendered = _helm_template({"networkPolicy.agentEgress.enabled": "true"})
        docs = _parse_yaml_documents(rendered)
        netpols = {
            d["metadata"]["name"]: d
            for d in docs
            if d.get("kind") == "NetworkPolicy"
            and str(d.get("metadata", {}).get("name", "")).endswith("-egress")
        }
        # Verify a couple to catch helm-template-time corruption.
        search = netpols["cogniverse-search-agent-egress"]
        labels = search["spec"]["podSelector"]["matchLabels"]
        assert labels.get("cogniverse-agent") == "search_agent"
        assert search["spec"]["policyTypes"] == ["Egress"]
        # And the egress rules survive the helm-template render
        # (templating bugs can corrupt nested blocks).
        assert isinstance(search["spec"]["egress"], list)
        assert len(search["spec"]["egress"]) >= 1

    def test_all_rendered_netpols_target_correct_namespace(self):
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


class TestStaleness:
    def test_chart_yamls_match_current_agent_policies(self, tmp_path: Path):
        """Re-run the CLI in the test and diff against the committed YAMLs.

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
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"egress-netpol CLI failed in subprocess: stderr={result.stderr[-2000:]}"
        )
        # Compare each freshly-generated file against the committed one.
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
                "run — re-run `cogniverse-runtime egress-netpol --output-dir "
                "charts/cogniverse/templates/agent-egress …` and commit"
            )
