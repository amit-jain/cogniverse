"""SandboxManager loads policies from configs/agent_policies/ + legacy fallback.

The directory was renamed from ``configs/openshell/`` (named after the
runtime that consumed it for CodingAgent) to ``configs/agent_policies/``
(named for what the YAMLs actually declare — agent constraints). This
test verifies:

  * the new default is loaded when ``configs/agent_policies/`` exists;
  * the legacy ``configs/openshell/`` directory is still honored when
    the new one is absent (with a deprecation warning) — existing
    deployments that mount the old path keep working;
  * an explicit ``policy_dir`` argument always wins regardless of which
    directory exists.

This is the back-compat half of the rename. The forward half — emitting
k8s NetworkPolicy from the YAMLs — is in a follow-up commit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

pytestmark = pytest.mark.integration


def _write_minimal_policy(path: Path, agent: str) -> None:
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


class TestNewCanonicalDirectory:
    def test_new_dir_loaded_when_present(self, tmp_path, monkeypatch):
        # Build a tmp tree with both directories — new one should win.
        new_dir = tmp_path / "configs" / "agent_policies"
        new_dir.mkdir(parents=True)
        _write_minimal_policy(new_dir / "search_agent.yaml", "search_agent")

        legacy_dir = tmp_path / "configs" / "openshell"
        legacy_dir.mkdir(parents=True)
        _write_minimal_policy(legacy_dir / "ignored_agent.yaml", "ignored_agent")

        monkeypatch.chdir(tmp_path)
        mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
        # Should resolve to the new dir.
        assert mgr._policy_dir.name == "agent_policies"
        # Loaded the agent that lives only in the new dir.
        # (Note: SandboxManager skips _load_policies when policy=DISABLED;
        # we verify the resolution itself rather than the loaded contents.)


class TestLegacyFallback:
    def test_legacy_dir_used_when_new_dir_missing(self, tmp_path, monkeypatch, caplog):
        # Only the legacy dir exists.
        legacy_dir = tmp_path / "configs" / "openshell"
        legacy_dir.mkdir(parents=True)
        _write_minimal_policy(legacy_dir / "search_agent.yaml", "search_agent")

        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.WARNING):
            mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
        assert mgr._policy_dir.name == "openshell", (
            "back-compat: when only the legacy dir exists, SandboxManager "
            "must load from it so existing deployments don't break"
        )
        # Deprecation warning surfaces so operators know to migrate.
        assert any(
            "deprecated" in rec.message.lower() and "agent_policies" in rec.message
            for rec in caplog.records
        ), (
            "loading from configs/openshell/ must log a deprecation "
            "warning so operators see the migration path"
        )

    def test_no_warning_when_new_dir_present(self, tmp_path, monkeypatch, caplog):
        # New dir present → no deprecation warning.
        new_dir = tmp_path / "configs" / "agent_policies"
        new_dir.mkdir(parents=True)
        _write_minimal_policy(new_dir / "search_agent.yaml", "search_agent")

        legacy_dir = tmp_path / "configs" / "openshell"
        legacy_dir.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.WARNING):
            SandboxManager(policy=SandboxPolicy.DISABLED)
        deprecation = [
            r
            for r in caplog.records
            if "deprecated" in r.message.lower() and "agent_policies" in r.message
        ]
        assert deprecation == [], (
            "no deprecation warning should fire when the new dir is in use; "
            f"got {[r.message for r in deprecation]}"
        )


class TestExplicitOverride:
    def test_explicit_policy_dir_wins_over_default(self, tmp_path, monkeypatch):
        # Both default dirs exist in the tree, but the operator passes a
        # third location explicitly.
        (tmp_path / "configs" / "agent_policies").mkdir(parents=True)
        (tmp_path / "configs" / "openshell").mkdir(parents=True)
        custom = tmp_path / "custom_policies"
        custom.mkdir()
        _write_minimal_policy(custom / "search_agent.yaml", "search_agent")

        monkeypatch.chdir(tmp_path)
        mgr = SandboxManager(
            policy_dir=custom,
            policy=SandboxPolicy.DISABLED,
        )
        assert mgr._policy_dir == custom


class TestRealRepoYAMLs:
    def test_real_agent_policies_directory_loads(self):
        """The actual repo's configs/agent_policies/ must be loadable.

        Catches any rename mistake — if a YAML was missed in the rename or
        the new directory wasn't created, this fails.
        """
        repo_root = Path(__file__).resolve().parents[3]
        new_dir = repo_root / "configs" / "agent_policies"
        assert new_dir.exists(), (
            f"after the rename, {new_dir} must exist with all agent "
            "policy YAMLs that used to live in configs/openshell/"
        )
        yamls = list(new_dir.glob("*.yaml"))
        assert len(yamls) >= 5, (
            "expected at least 5 agent policy YAMLs (coding, orchestrator, "
            f"routing, search, summarizer); found {[y.name for y in yamls]}"
        )
        # The 5 originally shipped agents must all be present.
        names = {y.stem for y in yamls}
        for required in (
            "coding_agent",
            "orchestrator_agent",
            "routing_agent",
            "search_agent",
            "summarizer_agent",
        ):
            assert required in names, (
                f"{required}.yaml missing from configs/agent_policies/ after the rename"
            )
