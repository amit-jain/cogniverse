"""Build the web client's image the way ``cogniverse up`` does and run it.

The container runs under the chart's security context (uid 1000, read-only
root filesystem with a writable /tmp, every capability dropped) on the host
network, so it reaches a runtime listening on 127.0.0.1.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx
from cogniverse_cli.images import IMAGE_DOCKERFILES, image_build_context

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TAG = "cogniverse/web:test"
BUILD_TIMEOUT_S = 1200
START_TIMEOUT_S = 60


def build_web_image(tag: str = TEST_TAG) -> str:
    """Build ``clients/web`` from the Dockerfile and context the CLI uses."""
    built = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            IMAGE_DOCKERFILES["web"],
            "-t",
            tag,
            image_build_context("web"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
    )
    assert built.returncode == 0, f"web image build failed:\n{built.stderr[-4000:]}"
    return tag


@contextmanager
def run_web_container(
    runtime_url: str, port: int, *, tag: str = TEST_TAG
) -> Iterator[tuple[str, str]]:
    """Run the image; yields its base URL and container name. On exit the
    container is stopped with SIGTERM and must exit 0."""
    name = f"cogniverse-web-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "--network=host",
            "--user",
            "1000:1000",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-e",
            f"COGNIVERSE_RUNTIME_URL={runtime_url}",
            "-e",
            f"PORT={port}",
            tag,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            try:
                if httpx.get(f"{url}/healthz", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, (
                f"the web container did not answer /healthz:\n{_logs(name)}"
            )
            time.sleep(0.2)
        yield url, name
        stopped = subprocess.run(
            ["docker", "stop", "-t", "10", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert stopped.returncode == 0, stopped.stderr
        code = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", name],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        assert code == "0", f"the web server exited {code}:\n{_logs(name)}"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _logs(name: str) -> str:
    logs = subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True, timeout=30
    )
    return logs.stdout + logs.stderr
