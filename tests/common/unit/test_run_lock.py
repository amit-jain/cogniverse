"""Unit tests for the e2e run lock and GPU residency preflight."""

from __future__ import annotations

import os
import subprocess

import pytest

import tests.e2e.run_lock as run_lock


class _FakeDocker:
    def __init__(
        self,
        *,
        exact_rows=(),
        running_rows=(),
        listing_error: str | None = None,
    ):
        self.exact_rows = list(exact_rows)
        # (container_id, name, devices_json) for `docker ps` + `docker inspect`
        self.running_rows = list(running_rows)
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
            stdout = "".join(
                f"{container_id}\t{name}\n"
                for container_id, name, _ in self.running_rows
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if command[:2] == ["docker", "inspect"]:
            wanted = command[4:]
            stdout = "".join(
                f"{name}\t{devices}\n"
                for container_id, name, devices in self.running_rows
                if container_id in wanted
            )
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

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            "exact-model sidecar 'cogniverse-test-llm' is leased by live pytest pid "
            f"{os.getpid()}; refusing to start the e2e stack"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency()

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


def test_ensure_e2e_gpu_residency_passes_with_the_cluster_warmed(monkeypatch, tmp_path):
    """The cluster's own model pods hold GPU memory inside the k3d node; only a
    non-k3d container with the GPU device mounted is a stray holder."""
    _patch_lease_dir(monkeypatch, tmp_path)
    docker = _FakeDocker(
        running_rows=(
            ("111111111111", "k3d-cogniverse-e2e-server-0", "null"),
            ("222222222222", "k3d-cogniverse-e2e-serverlb", "null"),
            ("333333333333", "openshell-cluster-openshell", "null"),
        ),
    )
    monkeypatch.setattr(run_lock.subprocess, "run", docker.run)
    reap_calls: list[str] = []
    monkeypatch.setattr(
        run_lock, "reap_dead_owner_containers", lambda: reap_calls.append("reaped")
    )

    run_lock.ensure_e2e_gpu_residency()

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
        ],
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        [
            "docker",
            "inspect",
            "--format",
            "{{.Name}}\t{{json .HostConfig.Devices}}",
            "333333333333",
        ],
    ]


def test_ensure_e2e_gpu_residency_fails_on_a_stray_gpu_container(monkeypatch, tmp_path):
    _patch_lease_dir(monkeypatch, tmp_path)
    devices = (
        '[{"PathOnHost":"/dev/kfd","PathInContainer":"/dev/kfd","CgroupPermissions":"rwm"},'
        '{"PathOnHost":"/dev/dri","PathInContainer":"/dev/dri","CgroupPermissions":"rwm"}]'
    )
    docker = _FakeDocker(
        exact_rows=(("aaaaaaaaaaaa", "cogniverse-test-llm-stale"),),
        running_rows=(
            ("111111111111", "k3d-cogniverse-e2e-server-0", "null"),
            ("444444444444", "some-vllm-experiment", devices),
        ),
    )
    monkeypatch.setattr(run_lock.subprocess, "run", docker.run)
    reap_calls: list[str] = []
    monkeypatch.setattr(
        run_lock, "reap_dead_owner_containers", lambda: reap_calls.append("reaped")
    )

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            "GPU device holders outside the e2e cluster after reclaim: "
            "some-vllm-experiment \\(/dev/dri, /dev/kfd\\); refusing to start the e2e stack"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency()

    assert reap_calls == ["reaped"]
    assert docker.removed == ["cogniverse-test-llm-stale"]
    assert docker.commands[-1] == [
        "docker",
        "inspect",
        "--format",
        "{{.Name}}\t{{json .HostConfig.Devices}}",
        "444444444444",
    ]


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
        run_lock.ensure_e2e_gpu_residency()

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
