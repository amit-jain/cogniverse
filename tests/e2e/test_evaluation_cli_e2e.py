"""The offline ``cogniverse-eval`` console script against the deployed cluster.

Pins that the script reports the Inspect run's real terminal status: a run whose
solver reaches the runtime exports one row per dataset sample and exits 0, and a
run whose solver cannot reach the runtime exits nonzero, names the failing eval
and writes no report — never a complete-looking report full of 0.0 scores.

The script is run inside the runtime pod, where the cluster's Phoenix and the
runtime itself resolve the way the shipped configuration says they do.
"""

from __future__ import annotations

import json
import subprocess

import httpx
import pytest

from cogniverse_evaluation.core.inspect_scorers import get_configured_scorers
from cogniverse_evaluation.data.datasets import INPUT_KEYS, OUTPUT_KEYS, DatasetManager
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    PHOENIX_URL,
    RUNTIME,
    TENANT_ID,
    _active_video_profile_name,
    _evaluation_query_rows,
    run_async,
    unique_id,
)

NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"

# The runtime serves itself inside its own pod on the container port the chart
# declares; the eval config names it so the experiment solver's HTTP call is a
# real one.
IN_POD_RUNTIME_URL = "http://localhost:8000"
UNREACHABLE_RUNTIME_URL = "http://127.0.0.1:59603"
EVAL_SAMPLE_COUNT = 3
EVAL_TOP_K = 10
EVAL_STRATEGY = "default"
# Inspect requires a model identity even for a task whose solver and scorers
# never generate; the retrieval solver calls the runtime, not a model.
INSPECT_MODEL = "mockllm/model"
CLI_TIMEOUT_S = 900

SUCCESS_TAIL = "\n✓ Evaluation complete\n"


def _exec_in_pod(argv: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            *argv,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_eval_config_in_pod(path: str, runtime_url: str) -> None:
    payload = json.dumps(
        {
            "runtime_url": runtime_url,
            "tenant_id": TENANT_ID,
            "top_k": EVAL_TOP_K,
        }
    )
    result = _exec_in_pod(
        [
            "python3",
            "-c",
            f"import pathlib; pathlib.Path({path!r}).write_text({payload!r})",
        ]
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _run_eval_cli(
    *, dataset: str, profile: str, config_path: str, output_path: str, log_dir: str
) -> subprocess.CompletedProcess:
    return _exec_in_pod(
        [
            "env",
            f"INSPECT_EVAL_MODEL={INSPECT_MODEL}",
            f"INSPECT_LOG_DIR={log_dir}",
            "cogniverse-eval",
            "evaluate",
            "--mode",
            "experiment",
            "--dataset",
            dataset,
            "-p",
            profile,
            "-s",
            EVAL_STRATEGY,
            "--tenant-id",
            TENANT_ID,
            "--config",
            config_path,
            "--output",
            output_path,
        ],
        timeout=CLI_TIMEOUT_S,
    )


def _pod_file_exists(path: str) -> bool:
    return _exec_in_pod(["test", "-f", path]).returncode == 0


def _read_pod_json(path: str) -> dict:
    result = _exec_in_pod(["cat", path])
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def evaluation_dataset() -> tuple[str, list[dict]]:
    """A Phoenix dataset this test writes, in the shape DatasetManager persists."""
    import pandas as pd
    from phoenix.client import Client as PhoenixSyncClient

    rows = [
        row
        for row in _evaluation_query_rows()
        if str(row.get("query", "")).strip() and row.get("expected_videos")
    ][:EVAL_SAMPLE_COUNT]
    assert len(rows) == EVAL_SAMPLE_COUNT

    frame = pd.DataFrame(
        [
            {
                "query": str(row["query"]),
                "category": str(row.get("category", "general")),
                "expected_videos": DatasetManager._join_expected(
                    row["expected_videos"]
                ),
            }
            for row in rows
        ]
    )
    name = f"e2e-eval-{unique_id('opt_eval').replace(':', '-')}"
    client = PhoenixSyncClient(base_url=PHOENIX_URL)
    client.datasets.create_dataset(
        name=name, dataframe=frame, input_keys=INPUT_KEYS, output_keys=OUTPUT_KEYS
    )
    yield name, frame.to_dict(orient="records")

    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": TENANT_ID,
            "http_endpoint": PHOENIX_URL,
            "grpc_endpoint": "localhost:33317",
        }
    )
    run_async(provider.datasets.delete_dataset(name))


def _configured_scorer_names() -> set[str]:
    return {scorer.__name__ for scorer in get_configured_scorers({})}


def _runtime_result_ids(query: str, profile: str) -> list[str]:
    response = httpx.post(
        f"{RUNTIME}/search/",
        json={
            "query": query,
            "profile": profile,
            "strategy": EVAL_STRATEGY,
            "top_k": EVAL_TOP_K,
            "tenant_id": TENANT_ID,
        },
        timeout=120.0,
    )
    assert response.status_code == 200, response.text[:500]
    return [
        str(result.get("source_id") or result.get("document_id") or "")
        for result in response.json().get("results", [])
    ]


@pytest.mark.e2e
class TestEvaluationCLIReportsTheRunsRealStatus:
    """``cogniverse-eval evaluate`` exits on the Inspect log's terminal status."""

    def test_a_reachable_runtime_exports_one_row_per_sample(self, evaluation_dataset):
        dataset, rows = evaluation_dataset
        profile = _active_video_profile_name(
            json.load(open("configs/config.json"))  # noqa: SIM115
        )
        stem = f"/tmp/{dataset}"
        config_path = f"{stem}.config.json"
        output_path = f"{stem}.result.json"
        _write_eval_config_in_pod(config_path, IN_POD_RUNTIME_URL)

        result = _run_eval_cli(
            dataset=dataset,
            profile=profile,
            config_path=config_path,
            output_path=output_path,
            log_dir=f"{stem}.inspect",
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        assert result.stdout.endswith(SUCCESS_TAIL), result.stdout[-500:]

        report = _read_pod_json(output_path)
        assert set(report) == {"mode", "dataset", "timestamp", "results"}, report
        assert report["mode"] == "experiment", report
        assert report["dataset"] == dataset, report

        exported = report["results"]
        assert [row["input"] for row in exported] == [row["query"] for row in rows], (
            exported
        )
        assert [row["target"] for row in exported] == [
            [row["expected_videos"]] for row in rows
        ], exported

        scorer_names = _configured_scorer_names()
        for exported_row, source_row in zip(exported, rows, strict=True):
            assert set(exported_row) == {
                "eval_id",
                "sample_id",
                "epoch",
                "input",
                "target",
                "trace_ids",
                "scores",
            }, exported_row
            assert exported_row["epoch"] == 1, exported_row
            # The experiment solver retrieves through the runtime, not through
            # recorded spans, so it records no trace ids.
            assert exported_row["trace_ids"] == [], exported_row
            assert set(exported_row["scores"]) == scorer_names, exported_row
            for name in scorer_names:
                assert set(exported_row["scores"][name]) == {
                    "value",
                    "explanation",
                }, exported_row

            # The expected id the runtime does return is scored, never dropped
            # to 0.0 by a missing-evidence fallback.
            retrieved = _runtime_result_ids(source_row["query"], profile)
            expected = source_row["expected_videos"]
            if any(expected in candidate for candidate in retrieved):
                assert exported_row["scores"]["recall_scorer"]["value"] != 0.0, (
                    exported_row,
                    retrieved,
                )

        assert len({row["eval_id"] for row in exported}) == 1, exported
        assert len({row["sample_id"] for row in exported}) == len(rows), exported

    def test_an_unreachable_runtime_fails_the_run_and_writes_no_report(
        self, evaluation_dataset
    ):
        dataset, _ = evaluation_dataset
        profile = _active_video_profile_name(
            json.load(open("configs/config.json"))  # noqa: SIM115
        )
        stem = f"/tmp/{dataset}-dead"
        config_path = f"{stem}.config.json"
        output_path = f"{stem}.result.json"
        _write_eval_config_in_pod(config_path, UNREACHABLE_RUNTIME_URL)

        result = _run_eval_cli(
            dataset=dataset,
            profile=profile,
            config_path=config_path,
            output_path=output_path,
            log_dir=f"{stem}.inspect",
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 1, combined[-4000:]
        assert "✗ Evaluation failed: Inspect evaluation" in combined, combined[-4000:]
        assert "status=error" in combined, combined[-4000:]
        assert SUCCESS_TAIL not in combined, combined[-4000:]
        assert _pod_file_exists(output_path) is False
