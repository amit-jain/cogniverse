"""Spin up a MinIO container for integration tests.

Mirrors the shape of :class:`VespaTestManager` so each test module gets an
isolated S3-compatible store with unique ports. Provides a thin boto3 client
helper for fixture setup (bucket creation, object upload).
"""

from __future__ import annotations

import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"MinIO never came up on {host}:{port}")


@dataclass
class MinIOInstance:
    container_name: str
    api_port: int
    console_port: int
    access_key: str
    secret_key: str

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def boto3_client(self) -> Any:
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
        )


class MinIOTestManager:
    """Start and stop a one-off MinIO container."""

    def __init__(
        self,
        access_key: str = DEFAULT_ACCESS_KEY,
        secret_key: str = DEFAULT_SECRET_KEY,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self._instance: MinIOInstance | None = None

    def start(self, name_prefix: str = "minio-cogniverse-test") -> MinIOInstance:
        name = f"{name_prefix}-{uuid.uuid4().hex[:8]}"
        api_port = _free_port()
        console_port = _free_port()

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "-p",
                f"{api_port}:9000",
                "-p",
                f"{console_port}:9001",
                "-e",
                f"MINIO_ROOT_USER={self.access_key}",
                "-e",
                f"MINIO_ROOT_PASSWORD={self.secret_key}",
                "minio/minio:latest",
                "server",
                "/data",
                "--console-address",
                ":9001",
            ],
            check=True,
            capture_output=True,
        )

        _wait_for_port("127.0.0.1", api_port)
        time.sleep(1.0)

        self._instance = MinIOInstance(
            container_name=name,
            api_port=api_port,
            console_port=console_port,
            access_key=self.access_key,
            secret_key=self.secret_key,
        )
        return self._instance

    def stop(self) -> None:
        if self._instance is None:
            return
        subprocess.run(
            ["docker", "stop", self._instance.container_name],
            check=False,
            capture_output=True,
        )
        self._instance = None

    @contextmanager
    def lifecycle(self, name_prefix: str = "minio-cogniverse-test"):
        try:
            yield self.start(name_prefix=name_prefix)
        finally:
            self.stop()
