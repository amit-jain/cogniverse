"""An OpenShell gateway this test owns, served over real gRPC on localhost.

The gateway speaks the shipped ``openshell`` protobuf service, so the
production ``SandboxClient``, ``SandboxSessionPool`` and ``SandboxManager``
run unmodified against it. Each sandbox is a separate temp directory and
every command runs as a real OS process inside it, which is what makes
filesystem and process isolation between sessions observable.
"""

from __future__ import annotations

import concurrent.futures
import re
import shlex
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import grpc
from openshell._proto import datamodel_pb2 as dm
from openshell._proto import openshell_pb2 as pb
from openshell._proto import openshell_pb2_grpc as rpc


class LocalGateway(rpc.OpenShellServicer):
    """Sandboxes as temp directories; commands as real processes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.live: dict[str, dm.Sandbox] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.execs: list[dict] = []
        self.readiness_error: grpc.StatusCode | None = None
        self.exec_error: grpc.StatusCode | None = None
        self._lock = threading.Lock()

    def sandbox_dir(self, name: str) -> Path:
        return self.root / name

    def CreateSandbox(self, request, context):
        with self._lock:
            name = f"sandbox-{len(self.created) + 1}"
            sandbox = dm.Sandbox(
                id=name,
                name=name,
                namespace="prodfixagents",
                phase=dm.SANDBOX_PHASE_READY,
            )
            self.live[name] = sandbox
            self.created.append(name)
            self.sandbox_dir(name).mkdir()
        return pb.SandboxResponse(sandbox=sandbox)

    def GetSandbox(self, request, context):
        if self.readiness_error is not None:
            context.abort(self.readiness_error, "controlled readiness outage")
        with self._lock:
            sandbox = self.live.get(request.name)
        if sandbox is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no sandbox {request.name}")
        return pb.SandboxResponse(sandbox=sandbox)

    def ListSandboxes(self, request, context):
        with self._lock:
            sandboxes = [self.live[name] for name in sorted(self.live)]
        return pb.ListSandboxesResponse(sandboxes=sandboxes)

    def DeleteSandbox(self, request, context):
        with self._lock:
            self.deleted.append(request.name)
            self.live.pop(request.name, None)
        shutil.rmtree(self.sandbox_dir(request.name), ignore_errors=True)
        return pb.DeleteSandboxResponse(deleted=True)

    def ExecSandbox(self, request, context):
        if self.exec_error is not None:
            context.abort(self.exec_error, "controlled exec outage")
        sandbox_root = self.sandbox_dir(request.sandbox_id)
        # Absolute task paths map into this sandbox's own directory so the
        # sandboxes have independent filesystems.
        args = [re.sub(r"/tmp\b", str(sandbox_root), a) for a in request.command]
        if args[:2] == ["sh", "-c"] and args[2].startswith("python "):
            args[2] = shlex.quote(sys.executable) + args[2][len("python") :]
        with self._lock:
            self.execs.append(
                {"session": request.sandbox_id, "command": list(request.command)}
            )
        result = subprocess.run(
            args,
            cwd=sandbox_root,
            capture_output=True,
            timeout=min(request.timeout_seconds or 10, 30),
        )
        yield pb.ExecSandboxEvent(stdout=pb.ExecSandboxStdout(data=result.stdout))
        yield pb.ExecSandboxEvent(stderr=pb.ExecSandboxStderr(data=result.stderr))
        yield pb.ExecSandboxEvent(exit=pb.ExecSandboxExit(exit_code=result.returncode))


@contextmanager
def serve_local_gateway(root: Path, max_workers: int = 8):
    """Serve ``LocalGateway`` on a loopback port; yields (gateway, endpoint)."""
    gateway = LocalGateway(root)
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=max_workers))
    rpc.add_OpenShellServicer_to_server(gateway, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield gateway, f"127.0.0.1:{port}"
    finally:
        server.stop(0).wait(5)
