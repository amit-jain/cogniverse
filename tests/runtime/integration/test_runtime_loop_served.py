"""A runtime worker keeps answering liveness through its first LM-backed turns.

Boots the image's own command with two uvicorn workers against the test Vespa
and Redis, the LM resolved from Modal and the encoders from the cluster, and
polls ``GET /health/live`` with the e2e suite's loop probe while each worker
serves its first gateway dispatch beside a profile delete, or its first
deep-research turn whose RLM budget expires. A worker's first LM call used to
import LiteLLM and the OpenAI client's resources on its LM thread; while they
built their pydantic models the serving loop waited for the interpreter, and a
liveness poll took over twice the probe's bound.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_core.agents.rlm_options import RLMOptions
from tests.e2e.loop_probe import LoopProbe, assert_loop_served
from tests.runtime.integration.test_runtime_worker_admin_state import (
    _close,
    _pinned,
    _profile_body,
    _request,
    _tenant,
)
from tests.runtime.integration.test_runtime_worker_processes import (
    _runtime,
    _serving,
)

pytestmark = pytest.mark.integration

SERVICES = ("gliner", "vllm_colpali", "denseon")
PROFILE = "video_colpali_smol500_mv_frame"
QUERY = "search for animal videos"
RESEARCH_QUERY = "What visual patterns appear in outdoor activity videos?"
RLM_TIMEOUT_SECONDS = next(
    constraint.ge
    for constraint in RLMOptions.model_fields["timeout_seconds"].metadata
    if hasattr(constraint, "ge")
)


@pytest.fixture
def runtime(
    tmp_path,
    workflow_state_redis_url,
    vespa_instance,
    ensure_host_ollama,
    remote_inference,
):
    """A fresh two-worker runtime: no worker has made an LM call yet."""
    endpoints = {service: remote_inference.resolve(service) for service in SERVICES}
    env = {
        "INFERENCE_SERVICE_URLS": json.dumps(
            {service: endpoint.base_url for service, endpoint in endpoints.items()}
        ),
        "COGNIVERSE_SEMANTIC_EMBED_URL": endpoints["denseon"].base_url,
        "COGNIVERSE_SEMANTIC_EMBED_MODEL": endpoints["denseon"].model_id,
    }
    with _runtime(tmp_path, workflow_state_redis_url, extra_env=env) as (
        process,
        log,
        port,
    ):
        workers = _serving(process, log)
        runtime = SimpleNamespace(process=process, log=log, port=port, workers=workers)
        tenant = _tenant("loop")
        pinned = _pinned(runtime, 1)
        try:
            created = _request(
                pinned[workers[0]][0],
                "POST",
                "/admin/profiles",
                dict(
                    _profile_body(tenant, PROFILE),
                    embedding_model=endpoints["vllm_colpali"].model_id,
                    deploy_schema=True,
                ),
            )
        finally:
            _close(pinned)
        assert created[0] == 201, created
        runtime.tenant = tenant
        yield runtime


def _gateway(connection, tenant: str):
    return _request(
        connection,
        "POST",
        "/agents/gateway_agent/process",
        {
            "agent_name": "gateway_agent",
            "query": QUERY,
            "context": {"tenant_id": tenant},
            "top_k": 3,
        },
    )


def _research(connection, tenant: str):
    return _request(
        connection,
        "POST",
        "/agents/deep_research_agent/process",
        {
            "agent_name": "deep_research_agent",
            "query": RESEARCH_QUERY,
            "context": {"tenant_id": tenant, "max_iterations": 1},
            "rlm": {
                "enabled": True,
                "timeout_seconds": RLM_TIMEOUT_SECONDS,
                "cache": False,
            },
        },
    )


def test_each_workers_first_gateway_dispatch_beside_a_profile_delete_keeps_liveness(
    runtime,
):
    deleted_tenant = _tenant("loopdel")
    pinned = _pinned(runtime, 2)
    try:
        created = _request(
            pinned[runtime.workers[0]][1],
            "POST",
            "/admin/profiles",
            dict(_profile_body(deleted_tenant, "deleted_profile"), deploy_schema=True),
        )
    finally:
        _close(pinned)
    pinned = _pinned(runtime, 2)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            probe = LoopProbe(base_url=f"http://127.0.0.1:{runtime.port}")
            probe.__enter__()
            dispatched = [
                pool.submit(_gateway, pinned[pid][0], runtime.tenant)
                for pid in runtime.workers
            ]
            deleted = pool.submit(
                _request,
                pinned[runtime.workers[0]][1],
                "DELETE",
                f"/admin/profiles/deleted_profile?tenant_id={deleted_tenant}"
                "&delete_schema=true",
            )
            answers = [future.result() for future in dispatched]
            deleted = deleted.result()
            served = probe.stop()
    finally:
        _close(pinned)

    assert created[0] == 201, created
    assert deleted[0] == 200, deleted
    assert (deleted[1]["profile_name"], deleted[1]["schema_deleted"]) == (
        "deleted_profile",
        True,
    )
    assert [
        (status, body.get("status"), body.get("gateway", {}).get("routed_to"))
        for status, body in answers
    ] == [(200, "success", "search_agent")] * len(runtime.workers), answers
    assert_loop_served(served)


def test_each_workers_first_rlm_turn_whose_budget_expires_keeps_liveness(runtime):
    pinned = _pinned(runtime, 1)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            probe = LoopProbe(base_url=f"http://127.0.0.1:{runtime.port}")
            probe.__enter__()
            turns = [
                pool.submit(_research, pinned[pid][0], runtime.tenant)
                for pid in runtime.workers
            ]
            answers = [future.result() for future in turns]
            served = probe.stop()
    finally:
        _close(pinned)

    assert [
        (status, body.get("status"), body.get("agent")) for status, body in answers
    ] == [(200, "success", "deep_research_agent")] * len(runtime.workers), answers
    assert [
        body["result"]["rlm_telemetry"]["rlm_attempted"] for _, body in answers
    ] == [True] * len(runtime.workers)
    assert_loop_served(served)
