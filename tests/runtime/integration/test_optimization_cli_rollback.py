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

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


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
    # Point the subprocess at the docker-managed Phoenix from
    # tests/conftest.py (per-pid HTTP / OTLP gRPC ports).
    env["TELEMETRY_HTTP_ENDPOINT"] = phoenix_container["http_endpoint"]
    env["TELEMETRY_OTLP_ENDPOINT"] = phoenix_container["otlp_endpoint"]
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


@pytest.mark.asyncio
async def test_cli_rollback_replaces_active_and_retires_canary_for_every_seed(
    manager, tenant_id, phoenix_container
):
    import asyncio

    agent = "rollback_served_agent"
    for version in range(1, 4):
        await manager.save_prompts_versioned(agent, {"summarizer": f"PROMPT_{version}"})
    await manager.promote_to_canary(agent, 2)
    await manager.promote_canary_to_active(agent)
    await manager.promote_to_canary(agent, 3, traffic_pct=50)
    seeds = [f"request-{index}" for index in range(32)]
    before = await asyncio.gather(
        *(manager.load_for_request(agent, request_seed=seed) for seed in seeds)
    )
    assert {row["version"] for row in before} == {2, 3}

    result = await asyncio.to_thread(
        _run_cli,
        [
            "--mode",
            "rollback",
            "--tenant-id",
            tenant_id,
            "--agent",
            agent,
            "--prompts-version",
            "1",
        ],
        phoenix_container,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["restored"] == {"prompts_version": 1}
    fresh = ArtifactManager(manager._provider, tenant_id)
    state = await fresh.get_artefact_state(agent)
    assert state["active"]["version"] == 1
    assert state["canary"] is None
    assert [(row["version"], row["reason"]) for row in state["retired"]] == [
        (2, "rollback"),
        (3, "rollback"),
    ]
    after = await asyncio.gather(
        *(fresh.load_for_request(agent, request_seed=seed) for seed in seeds)
    )
    assert (
        after
        == [
            {
                "prompts": {"summarizer": "PROMPT_1"},
                "served_from": "active",
                "version": 1,
                "variant_id": "default",
            }
        ]
        * 32
    )
    for key, (_, value) in list(manager._request_cache.items()):
        manager._request_cache[key] = (0.0, value)
    assert await manager.load_for_request(agent, request_seed=seeds[0]) == after[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_publication", [False, True])
async def test_rollback_publication_preserves_complete_concurrent_reads(
    manager, tenant_id, phoenix_container, fail_publication
):
    import asyncio
    import threading

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    agent = "rollback_publication_agent"
    for version in (1, 2):
        await manager.save_prompts_versioned(agent, {"summarizer": f"PROMPT_{version}"})
    await manager.promote_to_canary(agent, 2)
    await manager.promote_canary_to_active(agent)
    entered, release = threading.Event(), threading.Event()

    def intercept(method, path, body):
        if method == "POST" and f"artefact_state_{agent}".encode() in body:
            entered.set()
            if not release.wait(15):
                return 503, b'{"detail":"state publication gate timed out"}'
            if fail_publication:
                return 503, b'{"detail":"state publication unavailable"}'
        return None

    with InterceptFaultProxy(phoenix_container["http_endpoint"], intercept) as proxy:
        cli = asyncio.create_task(
            asyncio.to_thread(
                _run_cli,
                [
                    "--mode",
                    "rollback",
                    "--tenant-id",
                    tenant_id,
                    "--agent",
                    agent,
                    "--prompts-version",
                    "1",
                ],
                phoenix_container,
                {"TELEMETRY_HTTP_ENDPOINT": proxy.url},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10) is True
            fresh = ArtifactManager(manager._provider, tenant_id)
            reads = await asyncio.gather(
                *(
                    fresh.load_for_request(agent, request_seed=f"blocked-{index}")
                    for index in range(8)
                )
            )
            # Every concurrent read returns ONE complete artefact: the serving
            # revision the rollback is replacing stays readable for the whole
            # publication, so a reader sees the complete pre-rollback active
            # view, never a mixture of one version's prompts with another's
            # identity and never an absent artefact.
            assert (
                reads
                == [
                    {
                        "prompts": {"summarizer": "PROMPT_2"},
                        "served_from": "active",
                        "version": 2,
                        "variant_id": "default",
                    }
                ]
                * 8
            )
            assert [
                (read["prompts"]["summarizer"], read["version"]) for read in reads
            ] == [("PROMPT_2", 2)] * 8
            assert {read["served_from"] for read in reads} == {"active"}
        finally:
            release.set()
            result = await cli
        assert result.returncode == (1 if fail_publication else 0), result.stderr
        assert ("Rollback complete:" in result.stderr) is (not fail_publication)
        reader = ArtifactManager(manager._provider, tenant_id)
        # A publication the store keeps refusing leaves the pre-rollback
        # revision committed, so serving keeps the complete pre-rollback active
        # view, never a half-applied mixture, and the CLI exits nonzero instead
        # of claiming completion.
        assert await reader.load_for_request(agent, request_seed="after") == (
            {
                "prompts": {"summarizer": "PROMPT_2"},
                "served_from": "active",
                "version": 2,
                "variant_id": "default",
            }
            if fail_publication
            else {
                "prompts": {"summarizer": "PROMPT_1"},
                "served_from": "active",
                "version": 1,
                "variant_id": "default",
            }
        )
        state = await reader.get_artefact_state(agent)
        assert state["canary"] is None
        assert state["active"]["version"] == (2 if fail_publication else 1)
        assert await reader.load_prompts(agent) == {
            "summarizer": f"PROMPT_{2 if fail_publication else 1}"
        }
