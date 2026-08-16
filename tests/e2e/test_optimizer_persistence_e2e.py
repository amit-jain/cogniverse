"""E2E persistence coverage for the workflow optimizer and the
synthetic generation pipeline.

End-to-end runs against a live cluster have historically surfaced real
bugs invisible to unit tests — wrong import paths, factory model-id
rewriting, ConfigMap contract drift, persistence stubs that silently
dropped metrics. The workflow optimizer and synthetic generator share
the same chart + ArtifactManager + telemetry-provider plumbing. Tests
bring each through the full
save → persist → load round-trip against the live runtime pod and
Phoenix dataset store.

Each test sets up its own world — none of them rely on pre-existing
telemetry or operator-supplied config:

  1. *Setup*: drive the inputs the optimizer needs:
     - workflow optimizer: drive
       ``/agents/orchestrator_agent/process`` traffic so
       ``cogniverse.orchestration`` spans accumulate, then wait long
       enough for the BatchSpanProcessor + Phoenix ingest to catch up.
     - synthetic generator: run against the chart-shipped
       ``synthetic`` config block the runtime validates at startup.
  2. *Run*: invoke the optimizer (via ``optimization_cli`` or a tiny
     inline wrapper) inside the runtime pod over kubectl exec.
  3. *Assert*: the optimizer rc == 0 AND it actually produced a
     non-trivial artifact (non-empty pattern dict, or a pending-review
     batch that round-trips through ApprovalStorage with the exact
     example schema). A run that "succeeds" with empty output is a bug.

Marked ``slow`` and ``requires_optimizer_data`` to identify its runtime cost.
The tests provision the optimizer inputs and configuration they consume.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    RUNTIME,
    TENANT_ID,
    _ensure_sample_content_ingested,
)
from tests.e2e.test_api_e2e import DOCUMENT_PROFILE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_optimizer_data,
]

NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"

# Each optimizer call drives a kubectl exec into the runtime pod;
# bound the wait so a hung optimizer fails the test instead of
# hanging the suite forever. Tighter than the router test because
# these don't run a 23-trial MIPROv2 loop.
OPTIMIZER_TIMEOUT_S = int(os.environ.get("OPTIMIZER_TIMEOUT_S", "1800"))

CAPTION_CORPUS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "testset"
    / "Test_Human_Annotated_Captions"
)
CAPTION_CORPUS_LIMIT = 50

# Single orchestrator /process calls on the cluster LM routinely
# overshoot 240s; use the same endpoint budget as
# test_inbound_lm_output_approximations.
ORCHESTRATOR_PROCESS_TIMEOUT_S = 480.0


def _kubectl_exec(*shell_argv: str, timeout: int = OPTIMIZER_TIMEOUT_S):
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "-n",
        NAMESPACE,
        "exec",
        RUNTIME_DEPLOYMENT,
        "-c",
        RUNTIME_CONTAINER,
        "--",
        *shell_argv,
    ]
    command_text = " ".join(command)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            "optimizer runtime prerequisite executable is unavailable; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; error={exc!r}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "optimizer runtime command timed out; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"timeout={exc.timeout}s; stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )


@pytest.fixture(scope="module", autouse=True)
def optimizer_runtime_ready() -> None:
    """Prove the optimizer CLI can execute in this suite's runtime pod."""
    probe = _kubectl_exec(
        "python3",
        "-c",
        "print('__OPTIMIZER_RUNTIME_READY__')",
        timeout=60,
    )
    assert probe.returncode == 0, (
        "optimizer runtime prerequisite probe failed; "
        f"context={KUBECTL_CONTEXT!r}; namespace={NAMESPACE!r}; "
        f"deployment={RUNTIME_DEPLOYMENT!r}; container={RUNTIME_CONTAINER!r}; "
        f"returncode={probe.returncode}; stdout={probe.stdout!r}; "
        f"stderr={probe.stderr!r}"
    )
    assert probe.stdout.strip() == "__OPTIMIZER_RUNTIME_READY__", (
        "optimizer runtime prerequisite probe returned unexpected output; "
        f"context={KUBECTL_CONTEXT!r}; stdout={probe.stdout!r}; "
        f"stderr={probe.stderr!r}"
    )


@pytest.fixture(scope="module", autouse=True)
def optimizer_corpus_ready(optimizer_runtime_ready) -> None:
    """Seed enough real document content for 50-example optimizer runs."""

    if not CAPTION_CORPUS_DIR.exists():
        pytest.fail(
            f"caption corpus directory not found: {CAPTION_CORPUS_DIR}",
            pytrace=False,
        )

    caption_paths = sorted(CAPTION_CORPUS_DIR.glob("*.txt"))[:CAPTION_CORPUS_LIMIT]
    if len(caption_paths) < CAPTION_CORPUS_LIMIT:
        pytest.fail(
            f"expected at least {CAPTION_CORPUS_LIMIT} caption fixtures in "
            f"{CAPTION_CORPUS_DIR}, found {len(caption_paths)}",
            pytrace=False,
        )

    for caption_path in caption_paths:
        _ensure_sample_content_ingested(
            caption_path,
            profile=DOCUMENT_PROFILE,
            media_type="text/plain",
        )


def _drive_orchestrator_traffic(queries: list[str], wait_for_spans_s: int = 8) -> int:
    """Drive ``len(queries)`` requests through the orchestrator endpoint
    so ``cogniverse.orchestration`` spans accumulate in Phoenix, then
    pause for the BatchSpanProcessor to flush. Returns the count of
    successful (HTTP 200) responses — the caller asserts > 0 before
    running an optimizer that depends on those spans.

    The workflow optimizer queries spans by name ==
    ``cogniverse.orchestration`` (emitted by
    ``OrchestratorAgent.emit_orchestration_span``). Driving the
    orchestrator endpoint populates the optimizer's reader with one
    round of traffic.
    """
    success = 0
    failures: list[dict[str, object]] = []
    endpoint = f"{RUNTIME}/agents/orchestrator_agent/process"
    # AgentTask schema: agent_name in body, tenant_id under context.
    # The runtime refuses requests without tenant_id in context (no
    # bootstrap-tenant fallback), so omitting it 422s every call.
    with httpx.Client(timeout=ORCHESTRATOR_PROCESS_TIMEOUT_S) as client:
        for q in queries:
            try:
                response = client.post(
                    endpoint,
                    json={
                        "agent_name": "orchestrator_agent",
                        "query": q,
                        "context": {"tenant_id": TENANT_ID},
                    },
                )
                if response.status_code == 200:
                    success += 1
                else:
                    failures.append(
                        {
                            "query": q,
                            "status": response.status_code,
                            "body": response.text[:500],
                        }
                    )
            except httpx.HTTPError as exc:
                failures.append({"query": q, "error": repr(exc)})
    if success == 0:
        pytest.fail(
            "orchestrator traffic prerequisite produced no successful requests; "
            f"method='POST'; url={endpoint!r}; "
            f"timeout={ORCHESTRATOR_PROCESS_TIMEOUT_S}s; "
            f"tenant_id={TENANT_ID!r}; attempts={len(queries)}; "
            f"failures={json.dumps(failures, default=str)}",
            pytrace=False,
        )
    # BatchSpanProcessor schedules every 500ms with 30s flush ceiling
    # (see TelemetryConfig.batch_config). 8s is comfortably above the
    # schedule_delay so spans land before the optimizer queries Phoenix.
    time.sleep(wait_for_spans_s)
    return success


@pytest.mark.e2e
class TestWorkflowOptimizationPersistence:
    """Run optimization_cli --mode workflow and verify it writes the
    query_patterns blob (and template_index when patterns produce
    templates).
    """

    def test_workflow_optimization_persists_query_patterns(self):
        """End-to-end: drive orchestrator traffic to populate
        ``cogniverse.orchestration`` spans, run the workflow optimizer,
        and verify it actually produced patterns and persisted them.

        The optimizer skips the save_blob call when patterns are empty
        (no orchestration spans → no execution to extract patterns
        from), so the test must DRIVE the spans first. After that the
        contract is: optimizer rc == 0, log line confirms save, blob
        loads back as a non-empty JSON dict.
        """
        # Step 1: drive orchestrator queries so the optimizer has spans
        # to read. The workflow optimizer extracts WorkflowExecution
        # records from cogniverse.orchestration spans; one query
        # produces one orchestration span.
        queries = [
            "machine learning tutorial videos",
            "summarize quantum computing research",
            "find documentation on python decorators",
            "show cooking technique videos",
            "create a detailed report on renewable energy",
            "search for academic papers on AI ethics",
        ]
        sent = _drive_orchestrator_traffic(queries)
        assert sent > 0, (
            f"could not drive any orchestrator traffic — runtime not "
            f"accepting requests at {RUNTIME}/agents/orchestrator_agent/process. "
            "Without spans the optimizer has nothing to optimize."
        )

        # Step 2: run the workflow optimizer.
        result = _kubectl_exec(
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "workflow",
            "--tenant-id",
            TENANT_ID,
            "--lookback-hours",
            "24",
        )
        if result.returncode != 0:
            pytest.fail(
                f"workflow optimization failed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )

        # Step 3: parse the optimizer's JSON status block (printed at
        # end of run_workflow_optimization). Real success means
        # spans_found > 0 AND workflows_extracted > 0 AND at least one
        # of execution_demos_saved / agent_profiles_saved is non-zero.
        # The query_patterns blob is OPTIONAL — only saved when
        # WorkflowIntelligence builds non-empty per-query-type
        # patterns from many similar queries; from a small test traffic
        # mix we typically get demos but no patterns.
        json_start = result.stdout.rfind("{")
        cli_status = None
        if json_start != -1:
            try:
                cli_status = json.loads(result.stdout[json_start:])
            except json.JSONDecodeError:
                pass
        assert cli_status is not None, (
            f"could not parse JSON status from workflow CLI stdout — "
            f"the run_workflow_optimization contract is broken or no "
            f"status was emitted.\n--- stdout (tail) ---\n{result.stdout[-2000:]}"
        )
        if cli_status.get("status") == "no_data":
            pytest.fail(
                f"workflow optimizer reported no_data even though we drove "
                f"{sent} successful orchestrator queries. spans_found="
                f"{cli_status.get('spans_found')}, workflows_extracted="
                f"{cli_status.get('workflows_extracted')}. Either spans "
                f"didn't reach Phoenix in time or extraction failed.\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )
        assert cli_status.get("status") == "success", (
            f"workflow optimizer status != success: {cli_status}"
        )
        assert cli_status.get("spans_found", 0) > 0, (
            f"workflow optimizer reports spans_found=0: {cli_status}"
        )
        assert cli_status.get("workflows_extracted", 0) > 0, (
            f"workflow optimizer extracted 0 workflows from non-empty span "
            f"input: {cli_status}"
        )
        demos_saved = cli_status.get("execution_demos_saved", 0)
        profiles_saved = cli_status.get("agent_profiles_saved", 0)
        assert demos_saved > 0 or profiles_saved > 0, (
            f"workflow optimizer claimed success with workflows_extracted="
            f"{cli_status.get('workflows_extracted')} but persisted 0 demos "
            f"and 0 profiles — save_demonstrations path is broken: {cli_status}"
        )

        # Step 4: round-trip verify — actually load the demos back and
        # assert count matches what the CLI claimed to save. Without
        # this step a save_demonstrations bug that returned a fake
        # dataset id (or wrote 0 rows) would still pass step 3.
        if demos_saved > 0:
            probe_code = (
                "import asyncio\n"
                "from cogniverse_foundation.telemetry import "
                "get_telemetry_manager\n"
                "from cogniverse_agents.optimizer.artifact_manager import "
                "ArtifactManager\n"
                f"prov = get_telemetry_manager().get_provider("
                f"tenant_id={TENANT_ID!r})\n"
                f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
                "out = asyncio.run(mgr.load_demonstrations('workflow')) or []\n"
                "print('__WORKFLOW_DEMO_COUNT__', len(out))\n"
            )
            probe = _kubectl_exec("python3", "-c", probe_code, timeout=120)
            if probe.returncode != 0:
                pytest.fail(
                    f"workflow demo load probe failed: rc={probe.returncode}\n"
                    f"{probe.stderr[-2000:]}"
                )
            count = None
            for line in probe.stdout.splitlines():
                if line.startswith("__WORKFLOW_DEMO_COUNT__"):
                    count = int(line.split()[-1])
                    break
            assert count is not None and count > 0, (
                f"CLI reported execution_demos_saved={demos_saved} but the "
                f"workflow demos dataset is empty when read back — "
                f"save_demonstrations dropped them"
            )


_SYNTHETIC_EXAMPLE_FIELDS = {
    "query_enhancement": {
        "query",
        "enhanced_query",
        "expansion_terms",
        "synonyms",
        "context",
        "reasoning",
    },
    "profile": {
        "query",
        "available_profiles",
        "selected_profile",
        "reasoning",
        "query_intent",
        "modality",
        "complexity",
    },
}


def _load_review_batch(batch_id: str) -> dict:
    """Load a persisted approval batch back through the pod's
    ApprovalStorage, mirroring the CLI's own storage wiring."""
    probe_code = (
        "import asyncio, json\n"
        "from cogniverse_agents.approval.approval_storage import (\n"
        "    ApprovalStorageImpl,\n"
        ")\n"
        "from cogniverse_foundation.config.utils import (\n"
        "    create_default_config_manager,\n"
        ")\n"
        "from cogniverse_foundation.telemetry.manager import (\n"
        "    get_telemetry_manager,\n"
        ")\n"
        "system_config = create_default_config_manager().get_system_config()\n"
        "grpc_endpoint = system_config.telemetry_collector_endpoint\n"
        "if not grpc_endpoint.startswith('http'):\n"
        "    grpc_endpoint = f'http://{grpc_endpoint}'\n"
        "storage = ApprovalStorageImpl(\n"
        "    grpc_endpoint=grpc_endpoint,\n"
        "    http_endpoint=system_config.telemetry_url,\n"
        f"    tenant_id={TENANT_ID!r},\n"
        "    telemetry_manager=get_telemetry_manager(),\n"
        "    redis_url=system_config.redis_url,\n"
        ")\n"
        f"batch = asyncio.run(storage.get_batch({batch_id!r}))\n"
        "assert batch is not None, 'approval batch not found'\n"
        "print('__REVIEW_BATCH__' + json.dumps({\n"
        "    'batch_id': batch.batch_id,\n"
        "    'context': batch.context,\n"
        "    'items': [\n"
        "        {\n"
        "            'item_id': item.item_id,\n"
        "            'status': item.status.value,\n"
        "            'metadata': item.metadata,\n"
        "            'data': item.data,\n"
        "        }\n"
        "        for item in batch.items\n"
        "    ],\n"
        "}, default=str))\n"
    )
    probe = _kubectl_exec("python3", "-c", probe_code, timeout=240)
    if probe.returncode != 0:
        pytest.fail(
            f"review batch probe failed for {batch_id!r}: "
            f"rc={probe.returncode}\n"
            f"--- stdout ---\n{probe.stdout[-2000:]}\n"
            f"--- stderr ---\n{probe.stderr[-2000:]}"
        )
    for line in probe.stdout.splitlines():
        if line.startswith("__REVIEW_BATCH__"):
            return json.loads(line[len("__REVIEW_BATCH__") :])
    pytest.fail(
        f"review batch probe emitted no batch for {batch_id!r}; "
        f"stdout: {probe.stdout[-500:]}"
    )


def _assert_review_batch(
    batch: dict,
    *,
    batch_id: str,
    optimizer_type: str,
    agent_type: str,
    expected_items: int,
) -> None:
    assert batch["batch_id"] == batch_id
    context = batch["context"]
    assert context["tenant_id"] == TENANT_ID, context
    assert context["agent_type"] == agent_type, context
    assert context["optimizer"] == optimizer_type, context
    assert context["purpose"] == "optimizer_training", context
    items = batch["items"]
    assert len(items) == expected_items, (
        f"CLI reported {expected_items} pending items but the persisted "
        f"batch holds {len(items)}"
    )
    assert [item["item_id"] for item in items] == [
        f"{batch_id}_{index}" for index in range(expected_items)
    ]
    for item in items:
        assert item["status"] == "pending_review", item
        metadata = item["metadata"]
        assert metadata["agent_type"] == agent_type, metadata
        assert metadata["optimizer_type"] == optimizer_type, metadata
        assert metadata["synthetic"] is True, metadata
        assert set(item["data"]) == _SYNTHETIC_EXAMPLE_FIELDS[optimizer_type], (
            f"example fields diverge from the {optimizer_type} schema: "
            f"{sorted(item['data'])}"
        )


@pytest.mark.e2e
class TestSyntheticGenerationPersistence:
    """Run ``--mode synthetic`` end-to-end and verify generated examples
    persist as a pending-review approval batch.

    The chart ships the canonical ``synthetic`` config block and every
    runtime pod validates it at startup via
    ``parse_synthetic_runtime_config``, so the CLI runs against the
    pod's own ``/app/configs/config.json``. Generation drives the
    production per-tenant agents and persists a pending-review batch
    through ``ApprovalStorageImpl.save_batch``; optimizers train only
    on items a human later approves. Each test round-trips the
    persisted batch through ``ApprovalStorage.get_batch`` and asserts
    the exact example schema.
    """

    @staticmethod
    def _run_synthetic_cli(agents: str):
        return _kubectl_exec(
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "synthetic",
            "--tenant-id",
            TENANT_ID,
            "--agents",
            agents,
        )

    @staticmethod
    def _successful_outcome(result, optimizer_type: str) -> dict:
        """Parse the CLI's single JSON result document and return the
        per-optimizer outcome, failing with diagnostics otherwise."""
        if result.returncode != 0:
            pytest.fail(
                f"synthetic generation for {optimizer_type!r} failed: "
                f"rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"synthetic CLI must print exactly one JSON result: {exc}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}"
            )
        assert payload["status"] == "success", (
            f"synthetic CLI exited 0 but reported failure for "
            f"{optimizer_type!r}:\n{json.dumps(payload, indent=2)[:2000]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
        )
        assert set(payload["results"]) == {optimizer_type}, payload
        outcome = payload["results"][optimizer_type]
        assert outcome["status"] == "success", outcome
        generated = outcome["examples_generated"]
        assert generated >= 1, outcome
        assert outcome["pending_review"] == generated, outcome
        assert outcome["batch_id"].startswith(f"synthetic_{optimizer_type}_"), outcome
        return outcome

    def test_query_enhancement_demos_persist_as_review_batch(self):
        """``--agents query_enhancement``: the DSPy-LM-backed generator
        runs the production QueryEnhancementAgent per sampled document
        and the examples land in a pending-review batch."""
        result = self._run_synthetic_cli("query_enhancement")
        outcome = self._successful_outcome(result, "query_enhancement")

        batch = _load_review_batch(outcome["batch_id"])
        _assert_review_batch(
            batch,
            batch_id=outcome["batch_id"],
            optimizer_type="query_enhancement",
            agent_type="query_enhancement",
            expected_items=outcome["examples_generated"],
        )
        for item in batch["items"]:
            data = item["data"]
            assert data["query"].strip(), data
            assert data["enhanced_query"].strip(), data
            assert data["enhanced_query"] != data["query"], (
                f"enhanced_query must differ from the original query: {data}"
            )
            assert data["reasoning"].strip(), data

    def test_profile_demos_persist_as_review_batch(self):
        """``--agents profile``: ProfileGenerator labels sampled content
        via the production ProfileSelectionAgent; the examples land in a
        pending-review batch with the exact profile-selection schema."""
        result = self._run_synthetic_cli("profile")
        outcome = self._successful_outcome(result, "profile")

        batch = _load_review_batch(outcome["batch_id"])
        _assert_review_batch(
            batch,
            batch_id=outcome["batch_id"],
            optimizer_type="profile",
            agent_type="profile_selection",
            expected_items=outcome["examples_generated"],
        )
        for item in batch["items"]:
            data = item["data"]
            assert data["query"].strip(), data
            for field in ("query_intent", "modality", "complexity", "reasoning"):
                assert data[field].strip(), f"empty {field!r}: {data}"
            available = [
                profile.strip() for profile in data["available_profiles"].split(",")
            ]
            assert data["selected_profile"] in available, (
                f"selected_profile {data['selected_profile']!r} not in "
                f"available_profiles {available}"
            )

        # Optimizer types outside the approved-consumer contract are
        # refused before any generation runs.
        mixed = self._run_synthetic_cli("profile,missing_optimizer")
        assert mixed.returncode == 1, (
            "synthetic CLI must exit 1 for an optimizer type without an "
            "approved training-data consumer\n"
            f"--- stdout ---\n{mixed.stdout[-2000:]}\n"
            f"--- stderr ---\n{mixed.stderr[-2000:]}"
        )
        assert mixed.stdout.strip() == "", (
            "the consumer gate must reject the request before any "
            "generation output is produced; got stdout:\n"
            f"{mixed.stdout[-2000:]}"
        )
        assert (
            "Error: synthetic optimizer types have no approved "
            "training-data consumer: ['missing_optimizer']"
        ) in mixed.stderr, mixed.stderr[-2000:]
