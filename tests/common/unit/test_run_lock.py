"""Unit tests for the e2e run lock and GPU residency preflight."""

from __future__ import annotations

import subprocess

import pytest

import tests.e2e.run_lock as run_lock


class _FakeDocker:
    def __init__(self, *, running_rows=(), listing_error: str | None = None):
        # (container_id, name, devices_json) for `docker ps` + `docker inspect`
        self.running_rows = list(running_rows)
        self.listing_error = listing_error
        self.commands: list[list[str]] = []

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        if command[:2] == ["docker", "ps"]:
            if self.listing_error is not None:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr=self.listing_error
                )
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


def test_ensure_e2e_gpu_residency_passes_with_the_cluster_warmed(monkeypatch):
    """The cluster's own model pods hold GPU memory inside the k3d node; only a
    non-k3d container with the GPU device mounted is a stray holder."""
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
    assert docker.commands == [
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        [
            "docker",
            "inspect",
            "--format",
            "{{.Name}}\t{{json .HostConfig.Devices}}",
            "333333333333",
        ],
    ]


def test_ensure_e2e_gpu_residency_fails_on_a_stray_gpu_container(monkeypatch):
    devices = (
        '[{"PathOnHost":"/dev/kfd","PathInContainer":"/dev/kfd","CgroupPermissions":"rwm"},'
        '{"PathOnHost":"/dev/dri","PathInContainer":"/dev/dri","CgroupPermissions":"rwm"}]'
    )
    docker = _FakeDocker(
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
    assert docker.commands == [
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        [
            "docker",
            "inspect",
            "--format",
            "{{.Name}}\t{{json .HostConfig.Devices}}",
            "444444444444",
        ],
    ]


def test_ensure_e2e_gpu_residency_raises_when_docker_cannot_list_containers(
    monkeypatch,
):
    docker = _FakeDocker(
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
            "docker could not list running containers: Cannot connect to "
            "the Docker daemon at unix:///docker.sock"
        ),
    ):
        run_lock.ensure_e2e_gpu_residency()

    assert reap_calls == ["reaped"]
    assert docker.commands == [["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"]]
