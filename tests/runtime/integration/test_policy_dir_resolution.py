"""SandboxManager resolves its agent-policy directory.

Verifies:
  * the default ``configs/agent_policies/`` is used when no override is given;
  * an explicit ``policy_dir`` argument always wins;
  * the real repo's ``configs/agent_policies/`` loads with every agent YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

pytestmark = pytest.mark.integration


def _write_minimal_policy(path: Path) -> None:
    """Write a minimal valid policy YAML."""
    path.write_text(
        """
network_policies:
  egress:
    - host: localhost
      port: 11434
      comment: ollama
  deny_all_other: true
""",
        encoding="utf-8",
    )


class TestDefaultDirectory:
    def test_default_dir_resolved_when_no_override(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "configs" / "agent_policies"
        new_dir.mkdir(parents=True)
        _write_minimal_policy(new_dir / "search_agent.yaml")

        monkeypatch.chdir(tmp_path)
        mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
        assert mgr._policy_dir.name == "agent_policies"


class TestExplicitOverride:
    def test_explicit_policy_dir_wins_over_default(self, tmp_path, monkeypatch):
        (tmp_path / "configs" / "agent_policies").mkdir(parents=True)
        custom = tmp_path / "custom_policies"
        custom.mkdir()
        _write_minimal_policy(custom / "search_agent.yaml")

        monkeypatch.chdir(tmp_path)
        mgr = SandboxManager(policy_dir=custom, policy=SandboxPolicy.DISABLED)
        assert mgr._policy_dir == custom


class TestRealRepoYAMLs:
    def test_real_agent_policies_directory_loads(self):
        """The repo's configs/agent_policies/ must load with every agent YAML."""
        repo_root = Path(__file__).resolve().parents[3]
        policy_dir = repo_root / "configs" / "agent_policies"
        assert policy_dir.exists(), f"{policy_dir} must exist with all agent YAMLs"
        names = {y.stem for y in policy_dir.glob("*.yaml")}
        for required in (
            "coding_agent",
            "orchestrator_agent",
            "routing_agent",
            "search_agent",
            "summarizer_agent",
        ):
            assert required in names, (
                f"{required}.yaml missing from configs/agent_policies/"
            )
