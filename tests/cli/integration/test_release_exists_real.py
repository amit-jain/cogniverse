"""Real-subprocess tests for ``release_exists`` helm-failure classification.

``release_exists`` forks/execs a real ``helm`` (resolved via PATH). These tests
drive that real fork/exec — with the real helm binary when it is installed, and
otherwise with a real executable on PATH that emits helm's exact stderr — so the
classification is exercised end to end, never through a python-level mock.
"""

from __future__ import annotations

import os
import shlex
import shutil
import stat

import pytest
from cogniverse_cli.deploy import release_exists

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


def _install_fake_helm(tmp_path, monkeypatch, stderr_text: str, exit_code: int) -> None:
    """Put a real executable named ``helm`` first on PATH."""
    stderr_file = tmp_path / "helm_stderr.txt"
    stderr_file.write_text(stderr_text)
    fake = tmp_path / "helm"
    fake.write_text(
        f"#!/bin/sh\ncat {shlex.quote(str(stderr_file))} >&2\nexit {exit_code}\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_release_not_found_returns_false(tmp_path, monkeypatch) -> None:
    # Real helm cannot emit this offline (needs a reachable cluster with the
    # release absent), so drive the real fork/exec with a real helm-shaped exe.
    _install_fake_helm(tmp_path, monkeypatch, "Error: release: not found\n", 1)
    assert release_exists("cogniverse", "cogniverse") is False


def test_unreachable_cluster_aborts(tmp_path, monkeypatch) -> None:
    if shutil.which("helm"):
        # Real helm against a dead API server: a genuine unreachable-cluster
        # failure, whose stderr is not "release: not found".
        kubeconfig = tmp_path / "dead-kubeconfig.yaml"
        kubeconfig.write_text(
            "apiVersion: v1\n"
            "kind: Config\n"
            "clusters:\n"
            "- cluster:\n"
            "    server: https://127.0.0.1:59999\n"
            "  name: dead\n"
            "contexts:\n"
            "- context:\n"
            "    cluster: dead\n"
            "    user: none\n"
            "  name: dead\n"
            "current-context: dead\n"
            "users: []\n"
        )
        kubeconfig.chmod(0o600)
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    else:
        _install_fake_helm(
            tmp_path,
            monkeypatch,
            "Error: Kubernetes cluster unreachable: Get "
            '"https://127.0.0.1:59999/version": dial tcp 127.0.0.1:59999: '
            "connect: connection refused\n",
            1,
        )
    with pytest.raises(SystemExit):
        release_exists("cogniverse", "cogniverse")
