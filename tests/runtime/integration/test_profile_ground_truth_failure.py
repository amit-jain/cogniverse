"""The profile CLI reports failed Phoenix reads as failed workflow steps."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from urllib.parse import unquote, urlencode

import httpx
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    load_profile_selection_ground_truth_rows,
)
from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_runtime.optimization_cli import _run_failed
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.metadata_schemas import create_config_metadata_schema
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.conftest import _shared_vespa_application_package
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


def _manager(tenant: str, endpoint: str, grpc_endpoint: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant,
            "http_endpoint": endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return ArtifactManager(provider, tenant)


@pytest.fixture(scope="module")
def profile_config_vespa():
    manager = VespaDockerManager()
    info = manager.start_container(f"profile-config-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=info["config_port"]
        )._deploy_package(
            _shared_vespa_application_package([create_config_metadata_schema()])
        )
        manager.wait_for_application_ready(info)
        yield info["http_port"]
    finally:
        manager.stop_container(info)


def _cli(tenant: str, endpoint: str, grpc_endpoint: str, backend_port: int):
    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=backend_port
        )
    )
    config_manager.set_telemetry_config(
        TelemetryConfig(
            otlp_endpoint=grpc_endpoint,
            provider_config={"http_endpoint": endpoint, "grpc_endpoint": grpc_endpoint},
        ),
        tenant_id=SYSTEM_TENANT_ID,
    )
    env = dict(os.environ)
    env.update(
        BACKEND_URL="http://127.0.0.1",
        BACKEND_PORT=str(backend_port),
        PHOENIX_HTTP_ENDPOINT=endpoint,
        TELEMETRY_HTTP_ENDPOINT=endpoint,
        PHOENIX_GRPC_ENDPOINT=grpc_endpoint,
        TELEMETRY_OTLP_ENDPOINT=grpc_endpoint,
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "profile",
            "--tenant-id",
            tenant,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_reader", [False, True])
async def test_cli_ground_truth_read_failure_is_terminal(
    phoenix_container, profile_config_vespa, concurrent_reader
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    peer = f"prodfixoptimization:t{uuid.uuid4().hex}"
    endpoint = phoenix_container["http_endpoint"]
    grpc_endpoint = phoenix_container["otlp_endpoint"]
    manager = _manager(tenant, endpoint, grpc_endpoint)
    peer_manager = _manager(peer, endpoint, grpc_endpoint)
    ground_truth = [{"query": "find red kite", "expected_videos": ["red-kite"]}]
    await manager.save_blob(
        "config", "profile_selection_ground_truth", json.dumps(ground_truth)
    )
    await peer_manager.save_blob(
        "config", "profile_selection_ground_truth", json.dumps(ground_truth)
    )
    await manager.save_blob("model", "profile_selection", '{"incumbent":"keep"}')
    dataset_name = manager._blob_dataset_name(
        "config", "profile_selection_ground_truth"
    )
    before = httpx.get(f"{endpoint}/v1/datasets", params={"limit": 100}).json()
    reached = threading.Event()
    release = threading.Event()
    if not concurrent_reader:
        release.set()

    def fail_ground_truth(method, path, _body):
        if method == "GET" and dataset_name in unquote(path):
            reached.set()
            if not release.wait(timeout=30):
                raise AssertionError("ground-truth read barrier was not released")
            return 503, b'{"detail":"profile ground truth unavailable"}'
        return None

    with HTTPFaultProxy(endpoint, fail_ground_truth) as proxy:
        run = asyncio.create_task(
            asyncio.to_thread(
                _cli, tenant, proxy.url, grpc_endpoint, profile_config_vespa
            )
        )
        try:
            assert await asyncio.to_thread(reached.wait, 30) is True
            if concurrent_reader:
                peer_through_proxy = _manager(peer, proxy.url, grpc_endpoint)
                assert (
                    await asyncio.wait_for(
                        load_profile_selection_ground_truth_rows(peer_through_proxy), 10
                    )
                    == ground_truth
                )
                assert run.done() is False
            release.set()
            result = await run
        finally:
            release.set()
            if not run.done():
                await run

        summary = json.loads(result.stdout)
        expected_cause = (
            f"dataset store at {proxy.url} could not answer for dataset "
            f"{dataset_name!r}: HTTPStatusError: Server error '503 Service Unavailable' "
            f"for url '{proxy.url}/v1/datasets?{urlencode({'name': dataset_name})}'\n"
            "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503"
        )
        assert summary == {
            "status": "failed",
            "reason": "profile_selection_ground_truth_store_unavailable",
            "retryable": True,
            "error": "profile_selection_ground_truth store unavailable",
            "cause": {
                "type": "DatasetStoreUnavailableError",
                "message": expected_cause,
            },
        }
        assert result.returncode == 1, result.stderr
        assert _run_failed(summary) is True
        assert (
            _run_failed({"status": "success", "requested": {"profile": summary}})
            is True
        )
        assert [method for method, _, _ in proxy.requests if method != "GET"] == []

    assert (
        await manager.load_blob("model", "profile_selection") == '{"incumbent":"keep"}'
    )
    assert await load_profile_selection_ground_truth_rows(manager) == ground_truth
    assert httpx.get(f"{endpoint}/v1/datasets", params={"limit": 100}).json() == before


@pytest.mark.asyncio
async def test_cli_absent_ground_truth_has_distinct_nonretryable_reason(
    phoenix_container,
    profile_config_vespa,
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    result = await asyncio.to_thread(
        _cli,
        tenant,
        phoenix_container["http_endpoint"],
        phoenix_container["otlp_endpoint"],
        profile_config_vespa,
    )
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason": "profile_selection_ground_truth_missing",
        "retryable": False,
        "error": f"profile_selection_ground_truth is not configured for tenant {tenant}",
    }
    assert result.returncode == 1, result.stderr
