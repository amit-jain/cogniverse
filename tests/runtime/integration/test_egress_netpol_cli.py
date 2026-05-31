"""`cogniverse-optim --mode egress-netpol` emits k8s NetworkPolicy.

Translates the
agent policy YAMLs (the source of truth declaring per-agent egress
allow-lists) into k8s NetworkPolicy CRDs, which the cluster's CNI plugin
enforces in the kernel — process-bypass-proof, library-agnostic.

Verifies:
  * the CLI accepts ``--mode egress-netpol`` plus required flags;
  * for the real ``configs/agent_policies/*.yaml`` files in this repo,
    one NetworkPolicy YAML is generated per agent under the operator's
    output dir;
  * each emitted YAML is a valid NetworkPolicy resource (apiVersion,
    kind, metadata.name, spec.podSelector, spec.policyTypes, spec.egress);
  * each emitted policy's egress rules cover exactly the (port, protocol)
    pairs from the source YAML's ``network_policies.egress`` list, plus
    DNS (always required for service-name resolution);
  * agents whose source YAML omits ``deny_all_other`` are skipped
    (we only emit policies for opt-in declared agents);
  * argparse rejects malformed invocations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


def _run_cli(args: list) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["BACKEND_URL"] = env.get("BACKEND_URL", "http://localhost:8080")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.optimization_cli",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


class TestArgumentParsing:
    def test_missing_output_dir_rejected(self):
        result = _run_cli(["--mode", "egress-netpol", "--tenant-id", "any"])
        assert result.returncode != 0
        assert "output-dir" in result.stderr.lower()

    def test_missing_service_map_rejected(self, tmp_path):
        result = _run_cli(
            [
                "--mode",
                "egress-netpol",
                "--tenant-id",
                "any",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert result.returncode != 0
        assert "service-map" in result.stderr.lower()

    def test_malformed_service_map_rejected(self, tmp_path):
        result = _run_cli(
            [
                "--mode",
                "egress-netpol",
                "--tenant-id",
                "any",
                "--output-dir",
                str(tmp_path),
                "--service-map",
                "no_equals_separator",
            ]
        )
        assert result.returncode != 0


class TestEmitFromRealRepo:
    """Run the CLI against the actual configs/agent_policies/ directory."""

    @pytest.fixture
    def emit_dir(self, tmp_path: Path) -> Path:
        out = tmp_path / "netpols"
        result = _run_cli(
            [
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
            ]
        )
        assert result.returncode == 0, (
            f"egress-netpol exited non-zero. stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        summary = json.loads(result.stdout)
        assert summary["status"] == "ok"
        return out

    def test_outputs_one_yaml_per_eligible_agent(self, emit_dir: Path):
        # All 5 shipped agents have deny_all_other: true, so all 5
        # should produce a NetworkPolicy.
        emitted = sorted(p.name for p in emit_dir.glob("*-egress-netpol.yaml"))
        for required in (
            "coding_agent-egress-netpol.yaml",
            "search_agent-egress-netpol.yaml",
            "summarizer_agent-egress-netpol.yaml",
            "routing_agent-egress-netpol.yaml",
            "orchestrator_agent-egress-netpol.yaml",
        ):
            assert required in emitted, (
                f"{required} not emitted; got {emitted}. Either the source "
                "YAML lacks deny_all_other:true or its ports are unmapped."
            )

    def test_each_yaml_is_a_valid_networkpolicy(self, emit_dir: Path):
        for path in emit_dir.glob("*-egress-netpol.yaml"):
            with open(path) as f:
                doc = yaml.safe_load(f)
            assert doc["apiVersion"] == "networking.k8s.io/v1", (
                f"{path.name}: wrong apiVersion"
            )
            assert doc["kind"] == "NetworkPolicy", f"{path.name}: wrong kind"
            assert doc["metadata"]["name"].startswith("cogniverse-"), (
                f"{path.name}: name must be prefixed cogniverse-"
            )
            assert doc["metadata"]["namespace"] == "cogniverse"
            spec = doc["spec"]
            assert "podSelector" in spec
            assert spec["policyTypes"] == ["Egress"]
            assert isinstance(spec["egress"], list)
            assert len(spec["egress"]) >= 1, (
                f"{path.name}: must have at least one egress rule (plus DNS)"
            )

    def test_search_agent_netpol_includes_vespa_and_ollama(self, emit_dir: Path):
        with open(emit_dir / "search_agent-egress-netpol.yaml") as f:
            doc = yaml.safe_load(f)
        # Pull out (port, protocol) tuples from the egress rules.
        port_protocols = set()
        for rule in doc["spec"]["egress"]:
            for p in rule.get("ports", []):
                port_protocols.add((p["port"], p["protocol"]))
        # search_agent.yaml allows vespa:8080 + ollama:11434 + DNS 53.
        assert (8080, "TCP") in port_protocols, (
            "search_agent NetworkPolicy must allow Vespa (8080/TCP); "
            f"got {port_protocols}"
        )
        assert (11434, "TCP") in port_protocols, (
            "search_agent NetworkPolicy must allow Ollama (11434/TCP); "
            f"got {port_protocols}"
        )
        # DNS is always added.
        assert (53, "UDP") in port_protocols, (
            "DNS must be allowed in every NetworkPolicy or service-name "
            "resolution will fail"
        )

    def test_summarizer_agent_netpol_omits_vespa(self, emit_dir: Path):
        # summarizer_agent.yaml only declares ollama:11434 — Vespa
        # must not slip into its NetworkPolicy.
        with open(emit_dir / "summarizer_agent-egress-netpol.yaml") as f:
            doc = yaml.safe_load(f)
        port_protocols = set()
        for rule in doc["spec"]["egress"]:
            for p in rule.get("ports", []):
                port_protocols.add((p["port"], p["protocol"]))
        assert (11434, "TCP") in port_protocols
        assert (8080, "TCP") not in port_protocols, (
            "summarizer must NOT have Vespa egress allowed — its YAML "
            "only declares Ollama. Cross-contamination across agent "
            "policies would defeat the per-agent allow-list contract."
        )

    def test_pod_selector_scopes_to_specific_agent(self, emit_dir: Path):
        # Each NetworkPolicy must select only its own agent's pods.
        # Otherwise summarizer's allow-list would apply to search,
        # widening the search agent's egress.
        with open(emit_dir / "search_agent-egress-netpol.yaml") as f:
            doc = yaml.safe_load(f)
        match_labels = doc["spec"]["podSelector"]["matchLabels"]
        assert match_labels.get("cogniverse-agent") == "search_agent", (
            "search_agent NetworkPolicy must select only search_agent pods; "
            "missing or wrong cogniverse-agent label means the policy "
            "applies more broadly than intended"
        )
        # Cross-check summarizer.
        with open(emit_dir / "summarizer_agent-egress-netpol.yaml") as f:
            doc2 = yaml.safe_load(f)
        assert (
            doc2["spec"]["podSelector"]["matchLabels"]["cogniverse-agent"]
            == "summarizer_agent"
        )


class TestSkipBehavior:
    def test_yaml_without_deny_all_other_is_skipped(self, tmp_path: Path):
        # Build a tmp policy dir with one YAML that doesn't opt in.
        policy_dir = tmp_path / "policies"
        policy_dir.mkdir()
        (policy_dir / "permissive_agent.yaml").write_text(
            """
network_policies:
  egress:
    - host: localhost
      port: 8080
  deny_all_other: false
""",
            encoding="utf-8",
        )
        out = tmp_path / "netpols"
        result = _run_cli(
            [
                "--mode",
                "egress-netpol",
                "--tenant-id",
                "any",
                "--policy-dir",
                str(policy_dir),
                "--output-dir",
                str(out),
                "--service-map",
                "vespa=cogniverse/vespa-service:8080",
            ]
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        # Permissive agent must be skipped, and the reason recorded.
        assert summary["written"] == []
        assert any(
            s["agent"] == "permissive_agent" and "deny_all_other" in s["reason"]
            for s in summary["skipped"]
        )

    def test_unmapped_port_skips_with_clear_reason(self, tmp_path: Path):
        # A YAML referencing a port the operator forgot to put in --service-map
        # must be skipped (otherwise we'd silently produce a NetworkPolicy
        # that omits that port — a quiet over-restriction).
        policy_dir = tmp_path / "policies"
        policy_dir.mkdir()
        (policy_dir / "ghost_agent.yaml").write_text(
            """
network_policies:
  egress:
    - host: localhost
      port: 9999  # nothing in --service-map maps to this port
  deny_all_other: true
""",
            encoding="utf-8",
        )
        out = tmp_path / "netpols"
        result = _run_cli(
            [
                "--mode",
                "egress-netpol",
                "--tenant-id",
                "any",
                "--policy-dir",
                str(policy_dir),
                "--output-dir",
                str(out),
                "--service-map",
                "vespa=cogniverse/vespa-service:8080",
            ]
        )
        assert result.returncode == 0
        summary = json.loads(result.stdout)
        assert summary["written"] == []
        assert any(
            s["agent"] == "ghost_agent" and "9999" in s["reason"]
            for s in summary["skipped"]
        ), (
            f"unmapped port must be surfaced in skipped reasons; got "
            f"{summary['skipped']}"
        )
