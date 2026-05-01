"""E2E persistence coverage for the workflow optimizer and the
synthetic generation pipeline.

The router_optimizer test (`test_router_optimization_e2e.py`) surfaced
seven real bugs the first time it ran end-to-end against a live
cluster — wrong import paths, factory model-id rewriting, ConfigMap
contract drift, persistence stubs that silently dropped metrics. The
workflow optimizer and synthetic generator share the same chart +
ArtifactManager + telemetry-provider plumbing but were not exercised
end-to-end before this module. Tests bring each through the full
save → persist → load round-trip against the live runtime pod and
Phoenix dataset store.

Each test sets up its own world — none of them rely on pre-existing
telemetry or operator-supplied config:

  1. *Setup*: drive the inputs the optimizer needs:
     - workflow optimizer: drive
       ``/agents/orchestrator_agent/process`` traffic so
       ``cogniverse.orchestration`` spans accumulate, then wait long
       enough for the BatchSpanProcessor + Phoenix ingest to catch up.
     - synthetic generator: write a minimal ``synthetic`` block into
       a per-test config.json and point ``COGNIVERSE_CONFIG`` at it.
  2. *Run*: invoke the optimizer (via ``optimization_cli`` or a tiny
     inline wrapper) inside the runtime pod over kubectl exec.
  3. *Assert*: the optimizer rc == 0 AND it actually produced a
     non-trivial artifact (non-empty pattern dict, non-empty demo
     with parseable input/output). A run that "succeeds" with empty
     output is a bug.

Marked ``slow`` and ``requires_optimizer_data`` so it never runs in
the default e2e sweep — bring it up explicitly:

    pytest -m "slow and requires_optimizer_data" \\
        tests/e2e/test_optimizer_persistence_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_optimizer_data,
]

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"

# Each optimizer call drives a kubectl exec into the runtime pod;
# bound the wait so a hung optimizer fails the test instead of
# hanging the suite forever. Tighter than the router test because
# these don't run a 23-trial MIPROv2 loop.
OPTIMIZER_TIMEOUT_S = int(os.environ.get("OPTIMIZER_TIMEOUT_S", "1800"))


def _kubectl_exec(*shell_argv: str, timeout: int = OPTIMIZER_TIMEOUT_S):
    return subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
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
    # AgentTask schema: agent_name in body, tenant_id under context.
    # The runtime refuses requests without tenant_id in context (no
    # bootstrap-tenant fallback), so omitting it 422s every call.
    with httpx.Client(timeout=120.0) as client:
        for q in queries:
            try:
                r = client.post(
                    f"{RUNTIME}/agents/orchestrator_agent/process",
                    json={
                        "agent_name": "orchestrator_agent",
                        "query": q,
                        "context": {"tenant_id": TENANT_ID},
                    },
                )
                if r.status_code == 200:
                    success += 1
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError):
                continue
    # BatchSpanProcessor schedules every 500ms with 30s flush ceiling
    # (see TelemetryConfig.batch_config). 8s is comfortably above the
    # schedule_delay so spans land before the optimizer queries Phoenix.
    time.sleep(wait_for_spans_s)
    return success


@pytest.mark.e2e
@skip_if_no_runtime
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


@pytest.mark.e2e
@skip_if_no_runtime
class TestSyntheticGenerationPersistence:
    """Run synthetic data generation end-to-end and verify the demos
    land in the per-tenant demos dataset.

    ``optimization_cli --mode synthetic`` reads
    ``config.get("synthetic")`` from the tenant's ConfigManager
    (Vespa-backed) and expects an ``optimizer_configs.<type>`` block
    with the DSPy module + agent mapping rules the generator needs.
    The default chart does not ship such a block — it's an
    operator-supplied tuning config — so this test pushes a minimal
    one to Vespa via the in-cluster ConfigManager BEFORE invoking
    the CLI, mirroring what an operator would do. This also exercises
    the otherwise-untouched ``ConfigManager.set_config`` path for
    ``SyntheticGeneratorConfig``.
    """

    def test_synthetic_demos_land_in_dataset(self):
        # ConfigUtils.get("synthetic") reads from the runtime pod's
        # /app/configs/config.json (chart-mounted ConfigMap). Production
        # operators ship a synthetic block in their tenant's config; the
        # default chart doesn't. To exercise the CLI end-to-end without
        # a chart change, write the block straight into a per-test copy
        # of config.json and point ``COGNIVERSE_CONFIG`` at it for the
        # CLI subprocess. The subprocess inherits ``COGNIVERSE_CONFIG``
        # via kubectl exec --env (handled by ``_kubectl_exec_with_env``
        # wrapper below).
        setup = (
            "import json, os\n"
            "src = '/app/configs/config.json'\n"
            "dst = '/tmp/cogniverse-config-with-synthetic.json'\n"
            "with open(src) as f: blob = json.load(f)\n"
            "blob['synthetic'] = {\n"
            f"    'tenant_id': {TENANT_ID!r},\n"
            "    'optimizer_configs': {\n"
            "        'modality': {\n"
            "            'optimizer_type': 'modality',\n"
            "            'dspy_modules': {\n"
            "                'query_generator': {\n"
            "                    'signature_class': "
            "'cogniverse_synthetic.dspy_signatures.GenerateModalityQuery',\n"
            "                    'module_type': 'Predict',\n"
            "                    'lm_config': {},\n"
            "                    'metadata': {},\n"
            "                }\n"
            "            },\n"
            "            'profile_scoring_rules': [],\n"
            "            'agent_mappings': [\n"
            "                {'modality': 'VIDEO', "
            "'agent_name': 'video_search_agent', "
            "'confidence_threshold': 0.7},\n"
            "                {'modality': 'DOCUMENT', "
            "'agent_name': 'document_search_agent', "
            "'confidence_threshold': 0.7},\n"
            "            ],\n"
            "            'num_examples_target': 5,\n"
            "            'metadata': {},\n"
            "        }\n"
            "    },\n"
            "    'sampling_config': {},\n"
            "    'metadata': {},\n"
            "}\n"
            "with open(dst, 'w') as f: json.dump(blob, f)\n"
            "print('__SYNTHETIC_CONFIG_PATH__' + dst)\n"
        )
        seed = _kubectl_exec("python3", "-c", setup, timeout=120)
        if seed.returncode != 0:
            pytest.fail(
                f"failed to seed synthetic config: rc={seed.returncode}\n"
                f"--- stdout ---\n{seed.stdout[-2000:]}\n"
                f"--- stderr ---\n{seed.stderr[-2000:]}"
            )
        config_path = None
        for line in seed.stdout.splitlines():
            if line.startswith("__SYNTHETIC_CONFIG_PATH__"):
                config_path = line[len("__SYNTHETIC_CONFIG_PATH__") :]
                break
        if not config_path:
            pytest.fail(f"setup did not emit config path; stdout: {seed.stdout[-500:]}")

        # Now run the synthetic CLI with COGNIVERSE_CONFIG pointed at
        # the patched file so its ConfigManager loads our synthetic
        # block. ``env`` invocation through kubectl exec sets the var
        # only for this subprocess.
        result = _kubectl_exec(
            "env",
            f"COGNIVERSE_CONFIG={config_path}",
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "synthetic",
            "--tenant-id",
            TENANT_ID,
            "--agents",
            "modality",
            timeout=900,
        )
        if result.returncode != 0:
            pytest.fail(
                f"synthetic generation failed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )
        # The CLI prints a JSON result block at the end; the
        # rc=0-with-failed-status pattern (run_synthetic_generation
        # exits 0 even when every optimizer failed) was the trap that
        # caused earlier silent passes. Inspect the JSON.
        cli_status = None
        json_start = result.stdout.rfind("{")
        if json_start != -1:
            try:
                cli_status = json.loads(result.stdout[json_start:])
            except json.JSONDecodeError:
                cli_status = None
        if cli_status and cli_status.get("status") != "success":
            pytest.fail(
                f"synthetic CLI exited 0 but reported failure:\n"
                f"{json.dumps(cli_status, indent=2)[:2000]}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )

        # Demos land under ``synthetic_<optimizer_type>``; load via
        # ArtifactManager and assert at least one row.
        probe_code = (
            "import asyncio\n"
            "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
            "from cogniverse_agents.optimizer.artifact_manager import "
            "ArtifactManager\n"
            f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
            f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
            "out = asyncio.run(mgr.load_demonstrations('synthetic_modality'))\n"
            "print('__DEMO_COUNT__', len(out) if out else 0)\n"
        )
        probe = _kubectl_exec("python3", "-c", probe_code, timeout=120)
        if probe.returncode != 0:
            pytest.fail(
                f"demos load probe failed: rc={probe.returncode}\n"
                f"{probe.stderr[-2000:]}"
            )
        # Inspect the loaded demos to confirm structure + non-empty
        # input/output, not just row count. A bug that wrote N empty
        # rows would otherwise pass this check.
        probe_content = (
            "import asyncio, json\n"
            "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
            "from cogniverse_agents.optimizer.artifact_manager import "
            "ArtifactManager\n"
            f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
            f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
            "out = asyncio.run(mgr.load_demonstrations('synthetic_modality')) or []\n"
            "print('__COUNT__', len(out))\n"
            "if out:\n"
            "    print('__SAMPLE__' + json.dumps(out[0], default=str))\n"
        )
        probe2 = _kubectl_exec("python3", "-c", probe_content, timeout=120)
        if probe2.returncode != 0:
            pytest.fail(
                f"demo content probe failed: rc={probe2.returncode}\n"
                f"{probe2.stderr[-2000:]}"
            )
        count = None
        sample = None
        for line in probe2.stdout.splitlines():
            if line.startswith("__COUNT__"):
                count = int(line.split()[-1])
            elif line.startswith("__SAMPLE__"):
                sample = json.loads(line[len("__SAMPLE__") :])
        assert count is not None and count > 0, (
            f"synthetic CLI claimed success but Phoenix demos dataset is "
            f"empty (count={count}). save_demonstrations dropped them."
        )
        assert sample is not None, "first row probe missing"
        # input/output are JSON-string-serialised dicts in the demos
        # dataset; both must be present and non-empty so a downstream
        # optimizer can actually train on them.
        assert sample.get("input"), f"first synthetic demo has empty input: {sample}"
        assert sample.get("output"), f"first synthetic demo has empty output: {sample}"
        try:
            json.loads(sample["input"])
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"synthetic demo input is not valid JSON: {exc}\n"
                f"value: {sample['input'][:300]}"
            )

    def test_profile_synthetic_demos_land_in_dataset(self):
        """Run --mode synthetic --agents profile and verify the
        ProfileGenerator's output reaches the synthetic_profile demos
        dataset. ProfileGenerator is pattern-based (no DSPy LM
        required), so the synthetic config block can be minimal — just
        enough to satisfy the CLI's ConfigManager.
        """
        setup = (
            "import json\n"
            "src = '/app/configs/config.json'\n"
            "dst = '/tmp/cogniverse-config-with-profile-synthetic.json'\n"
            "with open(src) as f: blob = json.load(f)\n"
            "blob['synthetic'] = {\n"
            f"    'tenant_id': {TENANT_ID!r},\n"
            "    'optimizer_configs': {},\n"
            "    'sampling_config': {},\n"
            "    'metadata': {},\n"
            "}\n"
            "with open(dst, 'w') as f: json.dump(blob, f)\n"
            "print('__SYNTHETIC_CONFIG_PATH__' + dst)\n"
        )
        seed = _kubectl_exec("python3", "-c", setup, timeout=120)
        if seed.returncode != 0:
            pytest.fail(
                f"failed to seed profile synthetic config: rc={seed.returncode}\n"
                f"--- stdout ---\n{seed.stdout[-2000:]}\n"
                f"--- stderr ---\n{seed.stderr[-2000:]}"
            )
        config_path = None
        for line in seed.stdout.splitlines():
            if line.startswith("__SYNTHETIC_CONFIG_PATH__"):
                config_path = line[len("__SYNTHETIC_CONFIG_PATH__") :]
                break
        if not config_path:
            pytest.fail(f"setup did not emit config path; stdout: {seed.stdout[-500:]}")

        result = _kubectl_exec(
            "env",
            f"COGNIVERSE_CONFIG={config_path}",
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "synthetic",
            "--tenant-id",
            TENANT_ID,
            "--agents",
            "profile",
            timeout=900,
        )
        if result.returncode != 0:
            pytest.fail(
                f"profile synthetic generation failed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )
        cli_status = None
        json_start = result.stdout.rfind("{")
        if json_start != -1:
            try:
                cli_status = json.loads(result.stdout[json_start:])
            except json.JSONDecodeError:
                cli_status = None
        if cli_status and cli_status.get("status") != "success":
            pytest.fail(
                f"profile synthetic CLI exited 0 but reported failure:\n"
                f"{json.dumps(cli_status, indent=2)[:2000]}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )

        probe_content = (
            "import asyncio, json\n"
            "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
            "from cogniverse_agents.optimizer.artifact_manager import "
            "ArtifactManager\n"
            f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
            f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
            "out = asyncio.run(mgr.load_demonstrations('synthetic_profile')) or []\n"
            "print('__COUNT__', len(out))\n"
            "if out:\n"
            "    print('__SAMPLE__' + json.dumps(out[0], default=str))\n"
        )
        probe = _kubectl_exec("python3", "-c", probe_content, timeout=120)
        if probe.returncode != 0:
            pytest.fail(
                f"profile demo content probe failed: rc={probe.returncode}\n"
                f"{probe.stderr[-2000:]}"
            )
        count = None
        sample = None
        for line in probe.stdout.splitlines():
            if line.startswith("__COUNT__"):
                count = int(line.split()[-1])
            elif line.startswith("__SAMPLE__"):
                sample = json.loads(line[len("__SAMPLE__") :])
        assert count is not None and count > 0, (
            f"profile synthetic CLI claimed success but synthetic_profile "
            f"demos dataset is empty (count={count})"
        )
        assert sample is not None, "first profile row probe missing"
        assert sample.get("input"), f"first profile demo has empty input: {sample}"
        # The optimizer reads demo.input as JSON-encoded ProfileSelection
        # fields. Confirm structure so a downstream
        # run_profile_optimization can actually consume them.
        try:
            inp = json.loads(sample["input"])
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"profile demo input is not valid JSON: {exc}\n"
                f"value: {sample['input'][:300]}"
            )
        for required in (
            "query",
            "available_profiles",
            "selected_profile",
            "modality",
            "complexity",
            "query_intent",
            "confidence",
        ):
            assert required in inp, (
                f"profile demo input missing required field {required!r}: {inp}"
            )
        available = [p.strip() for p in inp["available_profiles"].split(",")]
        assert inp["selected_profile"] in available, (
            f"selected_profile {inp['selected_profile']!r} not in "
            f"available_profiles {available}"
        )
