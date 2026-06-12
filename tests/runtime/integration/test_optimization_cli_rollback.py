"""`cogniverse-optim --mode rollback` against real Phoenix.

Without this CLI, the rollback API existed only in Python — operators
had no way to revert an artefact promotion without writing custom code.
This test verifies, against a real Phoenix container, that:

  * the CLI accepts ``--mode rollback --agent ... --prompts-version N``
    and exits 0 on success;
  * after the CLI runs, the active prompts dataset matches the
    requested version's content;
  * the operator's rollback is itself reversible — the response includes
    ``backup_versions`` they can pass back in to undo;
  * argparse rejects malformed invocations (no agent, no version).

The CLI is invoked via direct subprocess so the test exercises argparse,
asyncio.run, JSON output, and the real ``run_rollback`` helper.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def tenant_id() -> str:
    return f"c4cli_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def manager(phoenix_container, tenant_id: str) -> ArtifactManager:
    """Manager wired to the docker-managed Phoenix on a per-pid port."""
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.fixture(autouse=True)
def _backend_url_from_shared_vespa(shared_vespa, monkeypatch):
    """CLI subprocesses must hit the session Vespa container, never the
    k3d cluster — integration provisions its own infrastructure."""
    monkeypatch.setenv("BACKEND_URL", f"http://localhost:{shared_vespa['http_port']}")
    monkeypatch.setenv("BACKEND_PORT", str(shared_vespa["http_port"]))


def _run_cli(
    args: list,
    phoenix_container: dict,
    env_overlay: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run the CLI as a real subprocess. Captures stdout/stderr.

    The subprocess inherits the test process's env so it sees Phoenix
    config (TELEMETRY_OTLP_ENDPOINT etc. set by the phoenix_container
    fixture in tests/conftest.py).
    """
    env = dict(os.environ)
    env["BACKEND_URL"] = os.environ["BACKEND_URL"]  # set by the
    # autouse shared-vespa fixture; never the k3d cluster.
    # Point the subprocess at the docker-managed Phoenix from
    # tests/conftest.py (per-pid HTTP / OTLP gRPC ports).
    env["PHOENIX_HTTP_ENDPOINT"] = phoenix_container["http_endpoint"]
    env["PHOENIX_GRPC_ENDPOINT"] = phoenix_container["otlp_endpoint"]
    if env_overlay:
        env.update(env_overlay)
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
        timeout=120,
    )


class TestArgumentParsing:
    def test_missing_agent_rejected(self, phoenix_container):
        result = _run_cli(
            ["--mode", "rollback", "--tenant-id", "any", "--prompts-version", "1"],
            phoenix_container,
        )
        assert result.returncode != 0, "missing --agent must error out"
        assert "--agent" in result.stderr or "agent" in result.stderr.lower()

    def test_missing_version_rejected(self, phoenix_container):
        result = _run_cli(
            ["--mode", "rollback", "--tenant-id", "any", "--agent", "x"],
            phoenix_container,
        )
        assert result.returncode != 0
        assert "version" in result.stderr.lower()


@pytest.mark.asyncio
class TestRollbackRoundTrip:
    async def test_cli_rollback_restores_versioned_prompts(
        self, manager: ArtifactManager, tenant_id: str, phoenix_container
    ):
        # Save three versions of prompts. save_prompts_versioned auto-
        # increments, so v1 → first call, v2 → second, etc.
        await manager.save_prompts_versioned(
            "rollback_agent", {"system": "VERSION_1_PROMPT"}
        )
        await manager.save_prompts_versioned(
            "rollback_agent", {"system": "VERSION_2_PROMPT"}
        )
        await manager.save_prompts_versioned(
            "rollback_agent", {"system": "VERSION_3_PROMPT"}
        )
        # Set the active to v3.
        await manager.save_prompts("rollback_agent", {"system": "VERSION_3_PROMPT"})

        # Run the CLI to roll back to v1.
        result = _run_cli(
            [
                "--mode",
                "rollback",
                "--tenant-id",
                tenant_id,
                "--agent",
                "rollback_agent",
                "--prompts-version",
                "1",
            ],
            phoenix_container,
        )
        assert result.returncode == 0, (
            f"CLI rollback exited non-zero. stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

        # The CLI prints a JSON summary on stdout.
        summary = json.loads(result.stdout)
        assert summary["agent_type"] == "rollback_agent"
        assert summary["restored"]["prompts_version"] == 1
        # Backup versions must be populated so the operator can undo.
        assert "backup_versions" in summary

        # Active prompts now match v1 content.
        active = await manager.load_prompts("rollback_agent")
        assert active == {"system": "VERSION_1_PROMPT"}, (
            "after CLI rollback, active prompts must reflect the requested "
            "version's content"
        )

    async def test_rollback_is_reversible_via_backup_versions(
        self, manager: ArtifactManager, tenant_id: str, phoenix_container
    ):
        # Save v1 + v2.
        await manager.save_prompts_versioned("reversible_agent", {"system": "V1"})
        await manager.save_prompts_versioned("reversible_agent", {"system": "V2"})
        # Active = v2.
        await manager.save_prompts("reversible_agent", {"system": "V2"})

        # Roll back to v1; the CLI returns the v2 backup version it created.
        first = _run_cli(
            [
                "--mode",
                "rollback",
                "--tenant-id",
                tenant_id,
                "--agent",
                "reversible_agent",
                "--prompts-version",
                "1",
            ],
            phoenix_container,
        )
        assert first.returncode == 0
        first_summary = json.loads(first.stdout)
        backup_v = first_summary["backup_versions"].get("prompts_version")
        assert backup_v is not None, (
            "rollback must snapshot the prior active state and report "
            "the version so the operator can undo"
        )
        assert (await manager.load_prompts("reversible_agent")) == {"system": "V1"}

        # Use the backup version to roll back the rollback.
        second = _run_cli(
            [
                "--mode",
                "rollback",
                "--tenant-id",
                tenant_id,
                "--agent",
                "reversible_agent",
                "--prompts-version",
                str(backup_v),
            ],
            phoenix_container,
        )
        assert second.returncode == 0
        # Active is back to v2.
        assert (await manager.load_prompts("reversible_agent")) == {"system": "V2"}
