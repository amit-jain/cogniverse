"""Phase 6c — ArtifactManager rollback CLI + hot-reload + reverse-rollback.

Pins:

  * ``cogniverse-optim --mode rollback`` (run via
    ``python -m cogniverse_runtime.optimization_cli``) restores the
    active prompts to a prior versioned snapshot AND captures a backup
    snapshot of the prior active before overwriting (so the rollback is
    itself reversible);
  * ``ArtifactManager.load_for_request`` reflects active-version flips
    without restarting the Python process;
  * Rolling back to the backup snapshot restores the original active
    prompts (rollback-of-rollback contract).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.e2e.conftest import run_async, skip_if_no_runtime, unique_id

PHOENIX_HTTP = "http://localhost:26006"
PHOENIX_GRPC = "localhost:4317"


def _make_artifact_manager(tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": PHOENIX_HTTP,
            "grpc_endpoint": PHOENIX_GRPC,
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


def _run(coro):
    # Run coroutine in a fresh OS thread to avoid pytest-asyncio's
    # auto-mode loop conflict — see tests/e2e/conftest.py:run_async.
    return run_async(coro)


def _parse_cli_json(stdout: str) -> dict:
    """Extract the trailing JSON object from the CLI's stdout.

    The CLI's last action is ``print(json.dumps(result, indent=2, default=str))``
    (multi-line JSON). Logging goes to stderr but stdout could still
    contain non-JSON noise (e.g. uv's progress lines), so locate the
    trailing top-level ``{ ... }`` block by scanning backwards for a
    line starting with ``{`` that yields a parseable JSON suffix.
    """
    text = stdout.strip()
    # Fast path: stdout IS the JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lines = text.splitlines()
    for i in range(len(lines)):
        candidate = "\n".join(lines[i:])
        if candidate.lstrip().startswith("{"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise AssertionError(
        f"could not locate a JSON object in CLI stdout:\n{text[:1000]}"
    )


def _invoke_rollback_cli(
    *, tenant_id: str, agent: str, prompts_version: int
) -> subprocess.CompletedProcess:
    """Invoke the rollback CLI as a subprocess (the operator's interface).

    Sets PHOENIX_HTTP_ENDPOINT / PHOENIX_GRPC_ENDPOINT so the CLI's
    ``_build_phoenix_provider_for_cli`` resolves to the same Phoenix
    instance the in-process ArtifactManager uses.
    """
    env = os.environ.copy()
    env["PHOENIX_HTTP_ENDPOINT"] = PHOENIX_HTTP
    env["PHOENIX_GRPC_ENDPOINT"] = PHOENIX_GRPC
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "rollback",
            "--tenant-id",
            tenant_id,
            "--agent",
            agent,
            "--prompts-version",
            str(prompts_version),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


# ---------------------------------------------------------------------------
# 1. CLI rollback restores prior active + records a backup snapshot
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRollbackCLIRestoresPriorActive:
    """``cogniverse-optim --mode rollback`` brings the active prompts back to v1."""

    def test_rollback_restores_v1_and_snapshots_v2(self) -> None:
        tenant_id = unique_id("opt_rb") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _setup() -> None:
            # Two distinct versions so the rollback path actually changes
            # the visible prompts.
            await am.save_prompts_versioned(agent_type, {"system": "v1-text"})
            await am.save_prompts_versioned(agent_type, {"system": "v2-text"})
            # Active = v2 (latest is what save_prompts wrote on the
            # versioned save's parallel un-versioned write — set
            # explicitly via the un-versioned save_prompts to be sure).
            await am.save_prompts(agent_type, {"system": "v2-text"})

        _run(_setup())

        # Sanity: active prompts before rollback are v2-text.
        assert _run(am.load_prompts(agent_type)) == {"system": "v2-text"}

        proc = _invoke_rollback_cli(
            tenant_id=tenant_id, agent=agent_type, prompts_version=1
        )
        assert proc.returncode == 0, (
            f"rollback CLI failed rc={proc.returncode}\nSTDOUT: {proc.stdout[:500]}\n"
            f"STDERR: {proc.stderr[:500]}"
        )
        # CLI prints its result dict as JSON on the last line.
        result = _parse_cli_json(proc.stdout)
        assert result["agent_type"] == agent_type
        assert result["restored"] == {"prompts_version": 1}
        # Backup of the v2 state was written before the restore (so the
        # rollback is itself reversible).
        assert "prompts_version" in result["backup_versions"]
        backup_v = int(result["backup_versions"]["prompts_version"])
        assert backup_v >= 3, result["backup_versions"]

        # Active prompts are now v1-text.
        assert _run(am.load_prompts(agent_type)) == {"system": "v1-text"}

        # And a third versioned snapshot now exists (the backup of the
        # prior v2 active — the snapshot_active call inside rollback).
        versions = _run(am.list_versions("prompts", agent_type))
        version_numbers = sorted(v["version"] for v in versions)
        assert backup_v in version_numbers, (
            f"backup version {backup_v} missing from {version_numbers}"
        )


# ---------------------------------------------------------------------------
# 2. load_for_request reflects active flips without process restart
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestHotReloadWithoutRestart:
    """A second load_for_request after promote_canary_to_active sees v2 prompts."""

    def test_load_for_request_reflects_post_promotion_active(self) -> None:
        tenant_id = unique_id("opt_hot") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _scenario() -> None:
            await am.save_prompts_versioned(agent_type, {"system": "v1-text"})
            # Promote v1 to active first.
            await am.promote_to_canary(agent_type, version=1, traffic_pct=100)
            await am.promote_canary_to_active(agent_type)
            r1 = await am.load_for_request(agent_type, request_seed="seed_a")
            assert r1["prompts"] == {"system": "v1-text"}
            assert r1["served_from"] == "active"
            assert r1["version"] == 1

            # Save v2 + promote to active.
            await am.save_prompts_versioned(agent_type, {"system": "v2-text"})
            await am.promote_to_canary(agent_type, version=2, traffic_pct=100)
            await am.promote_canary_to_active(agent_type)

            # SAME process, SAME ArtifactManager instance — load_for_request
            # must now serve v2 without any restart.
            r2 = await am.load_for_request(agent_type, request_seed="seed_a")
            assert r2["prompts"] == {"system": "v2-text"}
            assert r2["served_from"] == "active"
            assert r2["version"] == 2

        _run(_scenario())


# ---------------------------------------------------------------------------
# 3. Rollback-of-rollback restores the prior active
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRollbackOfRollback:
    """Rollback to v1 → backup_v captures v2 → rollback to backup_v restores v2."""

    def test_reverse_rollback_restores_v2(self) -> None:
        tenant_id = unique_id("opt_rb2") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _setup() -> None:
            await am.save_prompts_versioned(agent_type, {"system": "v1-text"})
            await am.save_prompts_versioned(agent_type, {"system": "v2-text"})
            await am.save_prompts(agent_type, {"system": "v2-text"})

        _run(_setup())

        # First rollback: v2 active → v1 active. backup_v captures v2.
        first = _invoke_rollback_cli(
            tenant_id=tenant_id, agent=agent_type, prompts_version=1
        )
        assert first.returncode == 0, first.stderr[:500]
        first_result = _parse_cli_json(first.stdout)
        backup_v = int(first_result["backup_versions"]["prompts_version"])
        assert _run(am.load_prompts(agent_type)) == {"system": "v1-text"}

        # Second rollback: target the backup — must restore v2-text.
        second = _invoke_rollback_cli(
            tenant_id=tenant_id, agent=agent_type, prompts_version=backup_v
        )
        assert second.returncode == 0, second.stderr[:500]
        second_result = _parse_cli_json(second.stdout)
        assert second_result["restored"] == {"prompts_version": backup_v}
        assert _run(am.load_prompts(agent_type)) == {"system": "v2-text"}
