"""Inspect CLI scoring through fixture-owned Phoenix and real HTTP faults."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager

import pandas as pd
import pytest
from phoenix.client import Client

from cogniverse_evaluation.data.datasets import OUTPUT_KEYS, DatasetManager
from cogniverse_foundation.telemetry.context import (
    add_search_results_to_span,
    search_span,
)
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

_CLI_BOOTSTRAP = """
import sys
import cogniverse_foundation.telemetry.manager as tm
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_evaluation.providers import set_evaluation_provider
from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import PhoenixEvaluationProvider
endpoint, grpc = sys.argv[1:3]
tm._telemetry_manager = TelemetryManager(TelemetryConfig(
    otlp_endpoint=grpc, provider_config={'http_endpoint': endpoint, 'grpc_endpoint': grpc}
))
provider = PhoenixEvaluationProvider()
provider.initialize({'tenant_id': 'prodfixoptimization:cli', 'http_endpoint': endpoint, 'grpc_endpoint': grpc})
set_evaluation_provider(provider)
from cogniverse_evaluation.cli import cli
cli.main(args=sys.argv[3:])
"""


def _emit(tenant, query, item, content, *, malformed=False):
    with search_span(tenant_id=tenant, query=query, top_k=1) as span:
        add_search_results_to_span(
            span,
            [{"id": item, "source_id": item, "score": 1.0, "content": content}],
        )
        if malformed:
            span.set_attribute("output.value", "{broken")
        trace_id = format(span.get_span_context().trace_id, "032x")
    get_telemetry_manager().force_flush(timeout_millis=10000)
    return trace_id


def _dataset(endpoint, rows):
    """Write the dataset the way DatasetManager writes it in production."""
    name = f"offline-scoring-{uuid.uuid4().hex}"
    Client(base_url=endpoint).datasets.create_dataset(
        name=name,
        dataframe=pd.DataFrame(
            [
                {
                    "query": row["query"],
                    "expected_videos": DatasetManager._join_expected(
                        row["expected_videos"]
                    ),
                }
                for row in rows
            ]
        ),
        input_keys=["query"],
        output_keys=OUTPUT_KEYS,
    )
    return name


def _wait_for_ids(endpoint, tenant, expected):
    client = Client(base_url=endpoint)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        frame = client.spans.get_spans_dataframe(
            project_identifier=f"cogniverse-{tenant}", limit=100
        )
        if frame is not None and set(frame["context.trace_id"]) == expected:
            return
        time.sleep(0.05)
    pytest.fail(f"Phoenix did not index exact trace IDs {sorted(expected)}")


def _start_cli(tmp_path, phoenix, dataset, tenant, *, mode="batch", endpoint=None):
    output = tmp_path / f"{tenant.replace(':', '-')}-{mode}.json"
    config_path = output.with_suffix(".config.json")
    config_path.write_text(json.dumps({"max_iterations": 3, "poll_interval": 0.1}))
    env = dict(os.environ)
    # Trace solvers never invoke a model. Inspect still requires a model identity.
    env["INSPECT_EVAL_MODEL"] = "mockllm/model"
    env["INSPECT_LOG_DIR"] = str(output.with_suffix(".inspect"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CLI_BOOTSTRAP,
            endpoint or phoenix["http_endpoint"],
            phoenix["grpc_endpoint"],
            "evaluate",
            "--mode",
            mode,
            "--dataset",
            dataset,
            "--tenant-id",
            tenant,
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return process, output


def _finish(process):
    try:
        stdout, _ = process.communicate(timeout=90)
    except BaseException:
        process.kill()
        process.wait(timeout=10)
        raise
    return process.returncode, stdout


@contextmanager
def _phoenix_proxy(endpoint, *, block_reads=0, fail_after=None):
    """Delay or fail real span HTTP reads while forwarding other Phoenix calls."""
    state = {
        "reads": 0,
        "entered": threading.Event(),
        "release": threading.Event(),
        "lock": threading.Lock(),
    }

    def intercept(method, path, body):
        if "/spans" not in path:
            return None
        with state["lock"]:
            state["reads"] += 1
            read_number = state["reads"]
            if read_number == block_reads:
                state["entered"].set()
        if read_number <= block_reads and not state["release"].wait(30):
            return 504, b'{"error":"span read barrier timed out"}'
        if fail_after is not None and read_number > fail_after:
            return 503, b'{"error":"Phoenix span read unavailable"}'
        return None

    with InterceptFaultProxy(endpoint, intercept) as proxy:
        try:
            yield proxy.url, state
        finally:
            state["release"].set()


def _assert_sample(row, query, target, trace_id, relevance, precision):
    assert row["input"] == query
    assert row["target"] == [target]
    assert row["epoch"] == 1
    assert row["trace_ids"] == [trace_id]
    assert {name: score["value"] for name, score in row["scores"].items()} == {
        "relevance_scorer": relevance,
        "diversity_scorer": 1.0,
        "result_count_scorer": 0.1,
        "precision_scorer": precision,
        "recall_scorer": precision,
    }


def test_batch_cli_exports_exact_query_target_trace_and_scores(
    search_evaluator_provider, phoenix_container, tmp_path
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    endpoint = phoenix_container["http_endpoint"]
    alpha = _emit(tenant, "alpha", "video-a", "alpha")
    beta = _emit(tenant, "beta", "video-wrong", "unrelated")
    _wait_for_ids(endpoint, tenant, {alpha, beta})
    dataset = _dataset(
        endpoint,
        [
            {"query": "alpha", "expected_videos": ["video-a"]},
            {"query": "beta", "expected_videos": ["video-b"]},
        ],
    )
    process, output = _start_cli(tmp_path, phoenix_container, dataset, tenant)
    code, stdout = _finish(process)
    assert code == 0, stdout
    rows = json.loads(output.read_text())["results"]
    assert len(rows) == 2
    by_query = {row["input"]: row for row in rows}
    _assert_sample(by_query["alpha"], "alpha", "video-a", alpha, 1.0, 1.0)
    _assert_sample(by_query["beta"], "beta", "video-b", beta, 0.0, 0.0)
    assert len({(row["eval_id"], row["sample_id"]) for row in rows}) == 2
    assert stdout.endswith("\n✓ Evaluation complete\n")


def test_live_cli_concurrent_tenants_keep_new_trace_ids_without_duplicates(
    search_evaluator_provider, phoenix_container, tmp_path
):
    endpoint = phoenix_container["http_endpoint"]
    tenants = [f"prodfixoptimization:t{uuid.uuid4().hex}" for _ in range(2)]
    datasets = [
        _dataset(endpoint, [{"query": "shared", "expected_videos": [item]}])
        for item in ["video-a", "video-b"]
    ]
    with _phoenix_proxy(endpoint, block_reads=2) as (proxy, state):
        runs = [
            _start_cli(
                tmp_path,
                phoenix_container,
                dataset,
                tenant,
                mode="live",
                endpoint=proxy,
            )
            for tenant, dataset in zip(tenants, datasets, strict=True)
        ]
        try:
            assert state["entered"].wait(30) is True
            ids = [
                _emit(tenant, "shared", item, "shared")
                for tenant, item in zip(tenants, ["video-a", "video-b"], strict=True)
            ]
            for tenant, trace_id in zip(tenants, ids, strict=True):
                _wait_for_ids(endpoint, tenant, {trace_id})
            state["release"].set()
            exported = []
            for (process, output), item, trace_id in zip(
                runs, ["video-a", "video-b"], ids, strict=True
            ):
                code, stdout = _finish(process)
                assert code == 0, stdout
                rows = json.loads(output.read_text())["results"]
                assert len(rows) == 1
                _assert_sample(rows[0], "shared", item, trace_id, 1.0, 1.0)
                exported.append(rows[0]["eval_id"])
            assert len(set(exported)) == 2
            assert state["reads"] == 6
        finally:
            state["release"].set()
            for process, _ in runs:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)


@pytest.mark.parametrize("fault", ["payload", "provider"])
def test_cli_failing_inspect_run_has_no_success_artifact(
    search_evaluator_provider, phoenix_container, tmp_path, fault
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    endpoint = phoenix_container["http_endpoint"]
    trace_id = _emit(tenant, "alpha", "video-a", "alpha", malformed=fault == "payload")
    _wait_for_ids(endpoint, tenant, {trace_id})
    dataset = _dataset(endpoint, [{"query": "alpha", "expected_videos": ["video-a"]}])
    with _phoenix_proxy(endpoint, fail_after=0 if fault == "provider" else None) as (
        proxy,
        _,
    ):
        process, output = _start_cli(
            tmp_path, phoenix_container, dataset, tenant, endpoint=proxy
        )
        code, stdout = _finish(process)
    assert code == 1, stdout
    assert "✓ Evaluation complete" not in stdout
    assert output.exists() is False
    assert "✗ Evaluation failed: Inspect evaluation" in stdout
    assert "status=error" in stdout


def test_live_provider_failure_after_first_poll_discards_partial_run(
    search_evaluator_provider, phoenix_container, tmp_path
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    endpoint = phoenix_container["http_endpoint"]
    dataset = _dataset(endpoint, [{"query": "alpha", "expected_videos": ["video-a"]}])
    with _phoenix_proxy(endpoint, block_reads=1, fail_after=1) as (proxy, state):
        process, output = _start_cli(
            tmp_path, phoenix_container, dataset, tenant, mode="live", endpoint=proxy
        )
        try:
            assert state["entered"].wait(30) is True
            trace_id = _emit(tenant, "alpha", "video-a", "alpha")
            _wait_for_ids(endpoint, tenant, {trace_id})
            state["release"].set()
            code, stdout = _finish(process)
        finally:
            state["release"].set()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert code == 1, stdout
    assert "status=error" in stdout
    assert "✓ Evaluation complete" not in stdout
    assert output.exists() is False


def _run_installed_cli(tmp_path, shared_vespa, dataset, tenant, *, http_endpoint, grpc):
    """Run the ``cogniverse-eval`` console script as a deployment runs it.

    Nothing is bootstrapped in the child: its telemetry manager is built from
    the config store, and its Phoenix comes only from the variables the chart
    sets on the pod.
    """
    output = tmp_path / f"{tenant.replace(':', '-')}-installed.json"
    config_path = output.with_suffix(".config.json")
    config_path.write_text(json.dumps({"max_iterations": 3, "poll_interval": 0.1}))
    env = dict(os.environ)
    env.update(
        INSPECT_EVAL_MODEL="mockllm/model",
        INSPECT_LOG_DIR=str(output.with_suffix(".inspect")),
        BACKEND_URL="http://localhost",
        BACKEND_PORT=str(shared_vespa["http_port"]),
        TELEMETRY_OTLP_ENDPOINT=grpc,
        TELEMETRY_HTTP_ENDPOINT=http_endpoint,
    )
    completed = subprocess.run(
        [
            os.path.join(os.path.dirname(sys.executable), "cogniverse-eval"),
            "evaluate",
            "--mode",
            "batch",
            "--dataset",
            dataset,
            "--tenant-id",
            tenant,
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=180,
    )
    return completed.returncode, completed.stdout, output


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_installed_cli_evaluates_against_the_deployments_phoenix(
    search_evaluator_provider, phoenix_container, shared_vespa, tmp_path
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    endpoint = phoenix_container["http_endpoint"]
    alpha = _emit(tenant, "alpha", "video-a", "alpha")
    _wait_for_ids(endpoint, tenant, {alpha})
    dataset = _dataset(endpoint, [{"query": "alpha", "expected_videos": ["video-a"]}])

    code, stdout, output = _run_installed_cli(
        tmp_path,
        shared_vespa,
        dataset,
        tenant,
        http_endpoint=endpoint,
        grpc=phoenix_container["grpc_endpoint"],
    )

    assert code == 0, stdout
    rows = json.loads(output.read_text())["results"]
    assert len(rows) == 1
    _assert_sample(rows[0], "alpha", "video-a", alpha, 1.0, 1.0)
    assert stdout.endswith("\n✓ Evaluation complete\n")


def test_installed_cli_names_the_unreachable_phoenix_it_was_given(
    search_evaluator_provider, phoenix_container, shared_vespa, tmp_path
):
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex}"
    dataset = f"offline-scoring-{uuid.uuid4().hex}"
    dead_endpoint = f"http://127.0.0.1:{_closed_port()}"

    code, stdout, output = _run_installed_cli(
        tmp_path,
        shared_vespa,
        dataset,
        tenant,
        http_endpoint=dead_endpoint,
        grpc=phoenix_container["grpc_endpoint"],
    )

    assert code == 1, stdout
    assert output.exists() is False
    assert "✓ Evaluation complete" not in stdout
    assert stdout.endswith(
        f"✗ Evaluation failed: Loading dataset '{dataset}' from Phoenix at "
        f"{dead_endpoint} failed: [Errno 111] Connection refused\n"
    )


_CONCURRENT_RESOLUTION = """
import asyncio, json, sys, threading
from cogniverse_evaluation.providers import get_evaluation_provider
from cogniverse_foundation.telemetry import manager as tm
http_endpoint, grpc, order, *tenants = sys.argv[1:]

def configure():
    tm.configure_telemetry_endpoints(otlp_endpoint=grpc, http_endpoint=http_endpoint)

if order == "configure_first":
    configure()
builders = threading.Barrier(5)
managers, providers, lock = [], {}, threading.Lock()

def build():
    builders.wait(timeout=60)
    manager = tm.get_telemetry_manager()
    with lock:
        managers.append(id(manager))

def configure_while_building():
    builders.wait(timeout=60)
    if order == "configure_during_build":
        configure()

threads = [threading.Thread(target=build) for _ in range(4)]
threads.append(threading.Thread(target=configure_while_building))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=120)

resolvers = threading.Barrier(len(tenants))

def resolve(tenant):
    resolvers.wait(timeout=60)
    provider = get_evaluation_provider(tenant_id=tenant)
    with lock:
        providers[tenant] = provider

threads = [threading.Thread(target=resolve, args=(t,)) for t in tenants]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=120)

async def span_count(provider, tenant):
    frame = await provider.telemetry.traces.get_spans(project=f"cogniverse-{tenant}")
    return len(frame)

print("REPORT" + json.dumps({
    "managers": len(set(managers)),
    "manager_endpoints": tm.get_telemetry_manager().provider_endpoints(),
    "providers": {
        tenant: [
            provider.http_endpoint,
            provider.telemetry._http_endpoint,
            asyncio.run(span_count(provider, tenant)),
        ]
        for tenant, provider in sorted(providers.items())
    },
}))
"""


@pytest.mark.parametrize("order", ["configure_first", "configure_during_build"])
def test_concurrent_resolution_uses_the_configured_phoenix(
    search_evaluator_provider, phoenix_container, shared_vespa, order
):
    """The deployment's endpoints reach a manager whose cold build races the
    configure call, and every tenant's evaluation provider resolved
    concurrently on it reads that tenant's spans from that Phoenix."""
    endpoint = phoenix_container["http_endpoint"]
    grpc = phoenix_container["grpc_endpoint"]
    tenants = [f"prodfixoptimization:c{uuid.uuid4().hex}" for _ in range(4)]
    trace_ids = {
        tenant: _emit(tenant, "alpha", "video-a", "alpha") for tenant in tenants
    }
    for tenant, trace_id in trace_ids.items():
        _wait_for_ids(endpoint, tenant, {trace_id})
    env = dict(os.environ)
    env.update(
        BACKEND_URL="http://localhost", BACKEND_PORT=str(shared_vespa["http_port"])
    )
    env.pop("TELEMETRY_OTLP_ENDPOINT", None)
    completed = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_RESOLUTION, endpoint, grpc, order, *tenants],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout.rsplit("REPORT", 1)[1]) == {
        "managers": 1,
        "manager_endpoints": {"grpc_endpoint": grpc, "http_endpoint": endpoint},
        "providers": {tenant: [endpoint, endpoint, 1] for tenant in sorted(tenants)},
    }, completed.stdout[-6000:]
