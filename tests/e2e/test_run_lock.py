"""Unit tests for the e2e run lock and GPU residency preflight."""

from __future__ import annotations

import os
import subprocess

import pytest

import tests.e2e.run_lock as run_lock


@pytest.fixture(scope="session", autouse=True)
def e2e_stack():
    yield


class _FakeDocker:
    def __init__(
        self,
        *,
        exact_rows=(),
        running_names=(),
        listing_error: str | None = None,
    ):
        self.exact_rows = list(exact_rows)
        self.running_names = list(running_names)
        self.listing_error = listing_error
        self.commands: list[list[str]] = []

    @property
    def removed(self) -> list[str]:
        return [
            command[3]
            for command in self.commands
            if command[:3] == ["docker", "rm", "-f"]
        ]

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        if command[:3] == ["docker", "ps", "-a"]:
            if self.listing_error is not None:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr=self.listing_error
                )
            stdout = "".join(
                f"{container_id}\t{name}\n" for container_id, name in self.exact_rows
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{command[3]}\n", stderr=""
            )
        if command[:2] == ["docker", "ps"]:
            stdout = "".join(f"{name}\n" for name in self.running_names)
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise AssertionError(f"unexpected command: {command}")


def _patch_lease_dir(monkeypatch, tmp_path):
    lease_dir = tmp_path / "exact-model-leases"
    monkeypatch.setattr(run_lock, "_EXACT_MODEL_LEASE_DIR", lease_dir)
    return lease_dir


def test_classify_exact_model_containers_splits_leased_and_unleased_rows():
    leased, unleased = run_lock.classify_exact_model_containers(
        [
            ("aaaaaaaaaaaa", "cogniverse-test-llm"),
            ("bbbbbbbbbbbb", "cogniverse-test-llm-teacher"),
            ("cccccccccccc", "cogniverse-test-llm-fresh"),
        ],
        {"cogniverse-test-llm-teacher"},
    )

    assert leased == {"cogniverse-test-llm-teacher"}
    assert unleased == {
        "cogniverse-test-llm",
        "cogniverse-test-llm-fresh",
    }


def test_ensure_e2e_gpu_residency_reclaims_unleased_exact_model_sidecars(
    monkeypatch, tmp_path
):
    lease_dir = _patch_lease_dir(monkeypatch, tmp_path)
    lease_dir.mkdir(parents=True)
    lease_dir.joinpath(f"cogniverse-test-llm.{os.getpid()}").touch()

    docker = _FakeDocker(
        exact_rows=(
            ("aaaaaaaaaaaa", "cogniverse-test-llm-stale"),
            ("bbbbbbbbbbbb", "cogniverse-test-llm"),
        ),
    )
    monkeypatch.setattr(run_lock.subprocess, "run", docker.run)
    reap_calls: list[str] = []
    monkeypatch.setattr(
        run_lock, "reap_dead_owner_containers", lambda: reap_calls.append("reaped")
    )
    gtt_calls: list[str] = []

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            "exact-model sidecar 'cogniverse-test-llm' is leased by live pytest pid "
            f"{os.getpid()}; refusing to start the e2e stack"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency(
            gtt_reader=lambda: gtt_calls.append("called") or 0
        )

    assert reap_calls == ["reaped"]
    assert docker.removed == ["cogniverse-test-llm-stale"]
    assert docker.commands == [
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=cogniverse-test-exact-model",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        ["docker", "rm", "-f", "cogniverse-test-llm-stale"],
    ]
    assert gtt_calls == []


def test_ensure_e2e_gpu_residency_fails_when_gtt_remains_high_after_reclaim(
    monkeypatch, tmp_path
):
    _patch_lease_dir(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_rows=(("aaaaaaaaaaaa", "cogniverse-test-llm-stale"),),
        running_names=(
            "cogniverse-test-dashboard",
            "cogniverse-test-ingest",
        ),
    )
    monkeypatch.setattr(run_lock.subprocess, "run", docker.run)
    reap_calls: list[str] = []
    monkeypatch.setattr(
        run_lock, "reap_dead_owner_containers", lambda: reap_calls.append("reaped")
    )
    gtt_calls: list[str] = []

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            "GTT remains at 3.00 GiB after reclaim; running cogniverse-test-\\* "
            "containers: cogniverse-test-dashboard, cogniverse-test-ingest"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency(
            gtt_reader=lambda: gtt_calls.append("called") or (3 * 1024**3)
        )

    assert reap_calls == ["reaped"]
    assert docker.removed == ["cogniverse-test-llm-stale"]
    assert docker.commands == [
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=cogniverse-test-exact-model",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        ["docker", "rm", "-f", "cogniverse-test-llm-stale"],
        ["docker", "ps", "--format", "{{.Names}}"],
    ]
    assert gtt_calls == ["called"]


def test_ensure_e2e_gpu_residency_raises_when_docker_cannot_list_exact_models(
    monkeypatch, tmp_path
):
    _patch_lease_dir(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_rows=(("aaaaaaaaaaaa", "cogniverse-test-llm-stale"),),
        listing_error="Cannot connect to the Docker daemon at unix:///docker.sock",
    )
    monkeypatch.setattr(run_lock.subprocess, "run", docker.run)
    reap_calls: list[str] = []
    monkeypatch.setattr(
        run_lock, "reap_dead_owner_containers", lambda: reap_calls.append("reaped")
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "docker could not list exact-model containers: Cannot connect to "
            "the Docker daemon at unix:///docker.sock"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency(gtt_reader=lambda: 0)

    assert reap_calls == ["reaped"]
    assert docker.removed == []
    assert docker.commands == [
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=cogniverse-test-exact-model",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ]
    ]
