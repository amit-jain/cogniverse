from __future__ import annotations

import importlib
import inspect
import os
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.e2e.test_manual_optimization_e2e as manual_optimization
import tests.e2e.test_messaging_e2e as messaging
import tests.e2e.test_quality_monitor_e2e as quality_monitor
from tests.e2e.conftest import KUBECTL_CONTEXT, _telegram_real_flow_deselections


class _FakeItem:
    def __init__(self, nodeid: str, markers: set[str]):
        self.nodeid = nodeid
        self._markers = markers

    def iter_markers(self, name: str | None = None):
        if name is not None and name in self._markers:
            yield SimpleNamespace(name=name)


def test_telegram_real_flow_is_deselected_without_required_env(monkeypatch, tmp_path):
    # The gate resolves through read_secret, which also consults ./.env and
    # ~/.env, so an empty cwd and home are what "unset" actually means here.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TEST_CHAT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    item = _FakeItem(
        "tests/e2e/test_messaging_e2e.py::TestTelegramRealFlow::test_bot_can_send_message",
        {"requires_telegram_bot"},
    )

    deselected, reason = _telegram_real_flow_deselections([item])

    assert deselected == [item]
    assert reason == "missing TELEGRAM_BOT_TOKEN and TELEGRAM_TEST_CHAT_ID"


def test_e2e_vespa_ports_pin_the_33xxx_host_mapping():
    module_specs = {
        "tests.e2e.test_knowledge_summarization_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_multi_document_synthesis_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_federation_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_pinning_quotas_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_contradiction_detection_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_temporal_reasoning_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_citation_and_audit_agents_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_trust_ranking_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_provenance_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_contradiction_reconciliation_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_cross_tenant_comparison_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_kg_traversal_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_federated_query_agent_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
        "tests.e2e.test_annotation_feedback_e2e": ("VESPA_PORT", 33080, None, None),
        "tests.e2e.test_deep_synthesis_workflow_e2e": (
            "VESPA_HTTP_PORT",
            33080,
            "VESPA_CONFIG_PORT",
            33071,
        ),
    }

    for module_name, spec in module_specs.items():
        module = importlib.import_module(module_name)
        http_attr, http_expected, config_attr, config_expected = spec
        assert getattr(module, http_attr) == http_expected, module_name
        if config_attr is not None:
            assert getattr(module, config_attr) == config_expected, module_name


def test_telegram_module_uses_collection_deselection_and_not_runtime_asserts():
    module_src = inspect.getsource(messaging._assert_bot_ready)

    assert "pytest.fail(" in module_src
    assert "assert BOT_TOKEN" not in module_src
    assert "assert TEST_CHAT_ID" not in module_src
    assert "requires_telegram_bot" in inspect.getsource(messaging)


def test_argo_probe_call_sites_use_authoritative_namespace_and_helper():
    for module in (quality_monitor, manual_optimization):
        source = inspect.getsource(
            module.require_kubectl_cluster
            if module is quality_monitor
            else module.require_argo_workflows
        )
        # property, not byte-layout: the formatter may wrap this call
        assert "argo_workflow_controller_probe_command(" in source
        assert "ARGO_NAMESPACE" in source
        assert "argo_workflow_controller_probe_failure_message(" in source
        assert "namespace=NAMESPACE" not in source


def test_no_e2e_module_selects_the_workflow_controller_outside_the_helper():
    """Every controller probe must route through the shared helper.

    A literal selector here re-introduces the class twice seen: the wrong
    label (component=workflow-controller) and the right label queried in the
    wrong namespace, which silently returns empty and makes the guarded
    assertion unreachable.
    """
    e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
    offenders = []
    for path in sorted(e2e_dir.glob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "workflow-controller" in line and "workflow-submitter" not in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "e2e modules must select the Argo controller via "
        "argo_workflow_controller_probe_command(); found literal selectors:\n"
        + "\n".join(offenders)
    )


def _e2e_env_loader_block() -> str:
    """The runner's secret-loading block, executed verbatim by the tests below."""
    text = (
        Path(__file__).resolve().parents[3] / "scripts" / "run_e2e_batched.sh"
    ).read_text()
    start = text.index("# >>> e2e-env-loader")
    end = text.index("# <<< e2e-env-loader")
    return text[start:end]


def _run_loader(repo_root: Path) -> str:
    script = (
        "set -euo pipefail\n"
        + _e2e_env_loader_block()
        + '\nprintf "%s|%s" "${TELEGRAM_BOT_TOKEN:-}" "${TELEGRAM_TEST_CHAT_ID:-}"\n'
    )
    done = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "REPO_ROOT": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"loader failed: {done.stderr!r}"
    return done.stdout


def test_loader_exports_secrets_from_a_per_key_env_directory(tmp_path):
    """The shipped layout: `.env/` holding one bare-value file per secret."""
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text("123456:AAHbotToken\n")
    (env_dir / "TELEGRAM_TEST_CHAT_ID.env").write_text("-1001234567890\n")

    assert _run_loader(tmp_path) == "123456:AAHbotToken|-1001234567890"


def test_loader_exports_secrets_from_a_single_env_file(tmp_path):
    (tmp_path / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:AAHbotToken\nTELEGRAM_TEST_CHAT_ID=-1001234567890\n"
    )

    assert _run_loader(tmp_path) == "123456:AAHbotToken|-1001234567890"


def test_loader_tolerates_a_key_file_written_as_key_equals_value(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text(
        "# provisioned by hand\nTELEGRAM_BOT_TOKEN=123456:AAHbotToken\n"
    )

    assert _run_loader(tmp_path) == "123456:AAHbotToken|"


def test_expected_initial_trust_pins_every_kind_and_derivation_pair():
    """The e2e trust expectation, written out so drift is visible in review."""
    from cogniverse_core.memory.provenance import DerivationKind
    from tests.e2e.conftest import expected_initial_trust

    pairs = {
        ("entity_fact", DerivationKind.DIRECT_INGEST): 0.60,
        ("entity_fact", DerivationKind.AGENT_INFERENCE): 0.35,
        ("entity_fact", DerivationKind.EXTRACTION): 0.50,
        ("external_doc", DerivationKind.DIRECT_INGEST): 0.84,
        ("external_doc", DerivationKind.SYNTHESIS): 0.595,
        ("session_scratch", DerivationKind.SUMMARIZATION): 0.27,
        ("learned_strategy", DerivationKind.USER_ASSERT): 0.66,
        # 0.95 x 1.20 = 1.14, clamped to the [0.0, 1.0] range the product uses.
        ("tenant_instruction", DerivationKind.DIRECT_INGEST): 1.0,
    }
    assert {
        key: round(expected_initial_trust(*key), 10) for key in pairs
    } == pytest.approx(pairs)


def test_expected_initial_trust_rejects_an_unknown_schema_kind():
    from cogniverse_core.memory.provenance import DerivationKind
    from tests.e2e.conftest import expected_initial_trust

    with pytest.raises(KeyError):
        expected_initial_trust("not_a_schema_kind", DerivationKind.DIRECT_INGEST)


def test_telegram_gate_resolves_secrets_from_a_per_key_env_directory(
    monkeypatch, tmp_path
):
    """A secret provisioned the documented way must satisfy the gate.

    ``.env`` is a directory of bare-value ``<VAR>.env`` files; reading
    os.environ instead of read_secret ignored it and deselected the suite.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TEST_CHAT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text("123456:AAHbotToken\n")
    (env_dir / "TELEGRAM_TEST_CHAT_ID.env").write_text("-1001234567890\n")
    item = _FakeItem(
        "tests/e2e/test_messaging_e2e.py::TestTelegramRealFlow::test_bot_can_send_message",
        {"requires_telegram_bot"},
    )

    assert _telegram_real_flow_deselections([item]) == ([], None)


def test_telegram_gate_names_only_the_key_that_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TEST_CHAT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text("123456:AAHbotToken\n")
    item = _FakeItem(
        "tests/e2e/test_messaging_e2e.py::TestTelegramRealFlow::test_bot_can_send_message",
        {"requires_telegram_bot"},
    )

    deselected, reason = _telegram_real_flow_deselections([item])

    assert deselected == [item]
    assert reason == "missing TELEGRAM_TEST_CHAT_ID"


_LOCK_START = "# >>> e2e-run-lock"
_LOCK_END = "# <<< e2e-run-lock"


def _e2e_run_lock_block() -> str:
    """The runner's single-run lock, executed verbatim by the tests below."""
    text = (
        Path(__file__).resolve().parents[3] / "scripts" / "run_e2e_batched.sh"
    ).read_text()
    assert _LOCK_START in text and _LOCK_END in text, (
        "run_e2e_batched.sh carries no e2e-run-lock block: concurrent e2e runs "
        "are unenforced"
    )
    return text[text.index(_LOCK_START) : text.index(_LOCK_END)]


_INFERENCE_TEARDOWN_START = "# >>> e2e-inference-teardown"
_INFERENCE_TEARDOWN_END = "# <<< e2e-inference-teardown"


def _e2e_inference_teardown_block() -> str:
    text = _run_e2e_batched_script()
    assert _INFERENCE_TEARDOWN_START in text and _INFERENCE_TEARDOWN_END in text, (
        "run_e2e_batched.sh carries no inference teardown block: the EXIT trap "
        "does not own the model scale-down path"
    )
    return text[
        text.index(_INFERENCE_TEARDOWN_START) : text.index(_INFERENCE_TEARDOWN_END)
    ]


def _e2e_cleanup_body() -> str:
    block = _e2e_inference_teardown_block()
    match = re.search(
        r"_e2e_cleanup\(\) \{\n(?P<body>.*?)\n\}\n\ntrap _e2e_cleanup EXIT",
        block,
        re.S,
    )
    assert match, "run_e2e_batched.sh carries no cleanup function body"
    return match.group("body")


def _run_lock(lock_file, scan_pattern, trailer='echo "ACQUIRED $$"'):
    script = "set -euo pipefail\n" + _e2e_run_lock_block() + "\n" + trailer + "\n"
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "E2E_LOCK_FILE": str(lock_file),
            "E2E_LOCK_SCAN_PATTERN": scan_pattern,
        },
        capture_output=True,
        text=True,
    )


@contextmanager
def _detached(argv):
    """A process in its OWN process group - what a second run really looks like."""
    proc = subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _never_matches() -> str:
    return f"nomatch{uuid.uuid4().hex}"


def test_run_lock_refuses_to_start_while_a_live_holder_owns_the_lock(tmp_path):
    lock = tmp_path / "e2e.lock"
    with _detached(["sleep", "120"]) as holder:
        lock.write_text(f"{holder.pid}\n")

        done = _run_lock(lock, _never_matches())

    assert done.returncode == 3, f"expected refusal, got {done.returncode}: {done!r}"
    assert f"pid {holder.pid}" in done.stderr, done.stderr
    assert "ACQUIRED" not in done.stdout, done.stdout
    assert lock.read_text() == f"{holder.pid}\n", "refusal must not steal the lock"


def test_run_lock_takes_over_a_lock_whose_holder_is_dead(tmp_path):
    lock = tmp_path / "e2e.lock"
    with _detached(["sleep", "120"]) as corpse:
        dead_pid = corpse.pid
    lock.write_text(f"{dead_pid}\n")

    done = _run_lock(
        lock, _never_matches(), trailer='printf "%s %s" "$(cat "$E2E_LOCK_FILE")" "$$"'
    )

    assert done.returncode == 0, f"stale lock must not block a run: {done.stderr!r}"
    assert "stale" in done.stderr.lower(), done.stderr
    recorded, runner_pid = done.stdout.split()
    assert recorded == runner_pid != str(dead_pid), done.stdout


def test_run_lock_refuses_when_a_detached_e2e_pytest_is_already_running(tmp_path):
    token = f"lockprobe{uuid.uuid4().hex[:10]}"
    fake = tmp_path / "pytest"
    fake.write_text("#!/usr/bin/env bash\nsleep 120\n")
    fake.chmod(0o755)

    with _detached([str(fake), f"tests/e2e/{token}.py"]) as other:
        done = _run_lock(tmp_path / "e2e.lock", token)

    assert done.returncode == 3, f"expected refusal, got {done.returncode}: {done!r}"
    assert f"pid {other.pid}" in done.stderr, done.stderr
    assert not (tmp_path / "e2e.lock").exists(), "refusal must not create a lock"


def test_run_lock_ignores_a_matching_process_in_its_own_process_group(tmp_path):
    """The scan must not refuse because its OWN command line mentions the pattern."""
    token = f"lockprobe{uuid.uuid4().hex[:10]}"

    done = _run_lock(
        tmp_path / "e2e.lock", token, trailer=f'# {token}\necho "ACQUIRED $$"'
    )

    assert done.returncode == 0, f"self-match refusal: {done.stderr!r}"
    assert done.stdout.startswith("ACQUIRED "), done.stdout


def test_run_lock_writes_its_own_pid_and_releases_the_lock_on_exit(tmp_path):
    lock = tmp_path / "e2e.lock"

    done = _run_lock(
        lock, _never_matches(), trailer='printf "%s %s" "$(cat "$E2E_LOCK_FILE")" "$$"'
    )

    assert done.returncode == 0, done.stderr
    recorded, runner_pid = done.stdout.split()
    assert recorded == runner_pid, f"lock records {recorded}, runner is {runner_pid}"
    assert not lock.exists(), "the lock outlived the run that took it"


def test_run_lock_default_scan_pattern_matches_a_real_e2e_pytest_command_line():
    """A scan pattern that matches nothing is a gate that verifies nothing."""
    block = _e2e_run_lock_block()
    found = re.search(r'E2E_LOCK_SCAN_PATTERN:-(.*?)\}"', block)
    assert found, f"no default scan pattern in the lock block: {block!r}"
    pattern = found.group(1)

    def matches(command_line: str) -> bool:
        return (
            subprocess.run(
                ["awk", "-v", f"pat={pattern}", "$0 ~ pat {found=1} END {exit !found}"],
                input=command_line,
                text=True,
                capture_output=True,
            ).returncode
            == 0
        )

    assert matches(
        "/home/a/src/cogniverse/.venv/bin/python -m pytest tests/e2e/test_api_e2e.py -x"
    )
    assert matches("uv run pytest tests/e2e/ -k optimization --tb=long")
    assert not matches("uv run pytest tests/cli/unit/test_e2e_contracts.py")
    assert not matches("/usr/bin/vim tests/e2e/test_api_e2e.py")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


_HARDCODED_INFERENCE_DEPLOYMENTS = {
    "cogniverse-clap-embed",
    "cogniverse-colbert-pylate",
    "cogniverse-denseon",
    "cogniverse-gliner",
    "cogniverse-vllm-asr",
    "cogniverse-vllm-colpali",
    "cogniverse-vllm-llm-student",
    "cogniverse-vllm-llm-teacher",
}


def test_run_e2e_batched_script_derives_inference_deployments_from_live_labels():
    block = _e2e_inference_teardown_block()

    assert 'kubectl --context "$KUBECTL_CONTEXT" get deploy -n "$NS"' in block
    assert ".metadata.labels.app\\.kubernetes\\.io/component" in block
    assert "while IFS=$'\\t' read -r component name replicas;" in block
    assert '[[ "$component" == inference-* ]] || continue' in block
    assert 'E2E_INFERENCE_DEPLOYMENTS_TO_TEAR_DOWN+=("$name")' in block
    assert block.count('--context "$KUBECTL_CONTEXT"') == 2
    assert _HARDCODED_INFERENCE_DEPLOYMENTS.isdisjoint(
        set(re.findall(r"cogniverse-[a-z0-9-]+", block))
    ), block


def test_run_e2e_batched_script_tears_down_only_cold_inference_deployments_and_keeps_exit_code(
    tmp_path,
):
    scaled = tmp_path / "scaled.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "kubectl",
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "--context" ]]; then
  shift 2
fi
case "${{1:-}} ${{2:-}}" in
  "get deploy")
    printf '%b\n' \
      'inference-alpha\\tcogniverse-alpha\\t0' \
      'inference-beta\\tcogniverse-beta\\t1' \
      'runtime\\tcogniverse-runtime\\t1'
    ;;
  "scale deployment")
    printf '%s\n' "$3" >> "{scaled}"
    ;;
esac
exit 0
""",
    )
    _write_executable(
        bin_dir / "cat",
        """#!/usr/bin/env bash
set -euo pipefail
printf '1073741824\n'
""",
    )
    _write_executable(
        bin_dir / "sleep",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )

    lock_file = tmp_path / "e2e.lock"
    script = (
        "set -euo pipefail\n"
        f'export PATH="{bin_dir}:$PATH"\n'
        "export NS=cogniverse\n"
        f"export KUBECTL_CONTEXT={KUBECTL_CONTEXT!r}\n"
        f'_e2e_release_lock() {{ printf "LOCK_RELEASED\\n" > {str(lock_file)!r}; }}\n'
        + _e2e_inference_teardown_block()
        + "\n"
        + "exit 17\n"
    )
    done = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "E2E_LOCK_SCAN_PATTERN": _never_matches(),
            "KUBECTL_CONTEXT": KUBECTL_CONTEXT,
        },
        capture_output=True,
        text=True,
    )

    assert done.returncode == 17, done.stderr
    assert scaled.read_text().splitlines() == ["cogniverse-alpha"], scaled.read_text()
    assert lock_file.read_text() == "LOCK_RELEASED\n"
    assert "teardown rc=17" in done.stdout, done.stdout
    assert "scaled=1" in done.stdout, done.stdout


def test_run_e2e_batched_script_teardown_traps_exit_and_reports_gtt_from_card_glob():
    block = _e2e_inference_teardown_block()

    assert "trap _e2e_cleanup EXIT" in block
    assert "trap 'E2E_EXIT_CODE=130; exit 130' INT" in block
    assert "trap 'E2E_EXIT_CODE=143; exit 143' TERM" in block
    assert "local rc=${E2E_EXIT_CODE:-$?}" in block
    assert 'exit "$rc"' in block
    assert "/sys/class/drm/card*/device/mem_info_gtt_used" in block
    assert not re.search(r"/sys/class/drm/card[0-9]+/device/mem_info_gtt_used", block)


def test_run_e2e_batched_script_exit_cleanup_releases_the_lock():
    cleanup = _e2e_cleanup_body()

    assert "trap - EXIT INT TERM" in cleanup
    assert "_e2e_teardown_inference_deployments" in cleanup
    assert "_e2e_release_lock" in cleanup
    assert (
        cleanup.index("_e2e_teardown_inference_deployments")
        < cleanup.index("_e2e_release_lock")
        < cleanup.index('exit "$rc"')
    )


_E2E_BATCH_EXCLUSIONS_START = "# >>> e2e-batch-exclusions"
_E2E_BATCH_EXCLUSIONS_END = "# <<< e2e-batch-exclusions"
_E2E_BATCH_UNCOVERED_START = "# >>> e2e-batch-uncovered"
_E2E_BATCH_UNCOVERED_END = "# <<< e2e-batch-uncovered"
_E2E_BATCH_MANIFEST_ENTRY = re.compile(
    r'^\s*"(?P<path>tests/e2e/[a-z0-9_/]+\.py)\|(?P<reason>[^"]+)"\s*$',
    re.M,
)

EXPECTED_E2E_BATCH_UNCOVERED = {
    "tests/e2e/test_annotation_feedback_e2e.py",
    "tests/e2e/test_asr_sidecar_e2e.py",
    "tests/e2e/test_cronworkflow_execution_e2e.py",
    "tests/e2e/test_inbound_dspy_span_e2e.py",
    "tests/e2e/test_inbound_lm_output_approximations.py",
    "tests/e2e/test_inbound_redis_replay_e2e.py",
    "tests/e2e/test_ingestion_pipeline_telemetry.py",
    "tests/e2e/test_ingestion_upload_e2e.py",
    "tests/e2e/test_manual_optimization_e2e.py",
    "tests/e2e/test_messaging_gateway_e2e.py",
    "tests/e2e/test_multimodal_report_e2e.py",
    "tests/e2e/test_optimizer_persistence_e2e.py",
    "tests/e2e/test_orchestrator_inbound_e2e.py",
    "tests/e2e/test_quality_monitor_e2e.py",
}


def test_derived_kubectl_context_is_refused_when_empty() -> None:
    """An empty derived context is worse than none: kubectl treats
    ``--context ""`` as the CURRENT context, so a failed derivation would
    silently act on whatever cluster happens to be selected while still
    satisfying the --context guard."""
    script = _run_e2e_batched_script()
    derivation = script.index('KUBECTL_CONTEXT="$(')
    first_kubectl = script.index("kubectl --context")
    guard_window = script[derivation:first_kubectl]
    assert '[[ -z "$KUBECTL_CONTEXT" ]]' in guard_window, (
        "run_e2e_batched.sh derives KUBECTL_CONTEXT but never refuses an "
        "empty value before its first kubectl call"
    )
    assert "exit 2" in guard_window, (
        "an empty derived context must abort the run, not warn"
    )


def _run_e2e_batched_script() -> str:
    return (
        Path(__file__).resolve().parents[3] / "scripts" / "run_e2e_batched.sh"
    ).read_text()


def _script_array_entries(script_text: str, array_name: str) -> set[str]:
    block = re.search(rf"^{re.escape(array_name)}=\((.*?)^\)", script_text, re.S | re.M)
    assert block, f"missing {array_name} array"
    return set(re.findall(r"tests/e2e/[a-z0-9_/]+\.py", block.group(1)))


def _script_manifest_entries(
    script_text: str, start_marker: str, end_marker: str, *, kind: str
) -> dict[str, str]:
    start = script_text.index(start_marker)
    end = script_text.index(end_marker)
    entries = _E2E_BATCH_MANIFEST_ENTRY.findall(script_text[start:end])
    assert entries, f"missing {kind} entries"
    manifest: dict[str, str] = {}
    for path, reason in entries:
        assert path not in manifest, f"duplicate {kind} entry for {path}"
        reason = reason.strip()
        assert reason and "\n" not in reason, f"bad {kind} reason for {path}"
        manifest[path] = reason
    return manifest


def test_run_e2e_batched_script_covers_every_e2e_test_file():
    script_text = _run_e2e_batched_script()
    batched_files = _script_array_entries(
        script_text, "BATCH1"
    ) | _script_array_entries(script_text, "BATCH2")
    exclusion_reasons = _script_manifest_entries(
        script_text,
        _E2E_BATCH_EXCLUSIONS_START,
        _E2E_BATCH_EXCLUSIONS_END,
        kind="exclusion",
    )
    uncovered_reasons = _script_manifest_entries(
        script_text,
        _E2E_BATCH_UNCOVERED_START,
        _E2E_BATCH_UNCOVERED_END,
        kind="uncovered",
    )
    exclusion_files = set(exclusion_reasons)
    uncovered_files = set(uncovered_reasons)
    e2e_dir = Path(__file__).resolve().parents[3] / "tests" / "e2e"
    filesystem_files = {f"tests/e2e/{path.name}" for path in e2e_dir.glob("test_*.py")}

    assert exclusion_files == {
        "tests/e2e/test_dashboard_e2e.py",
        "tests/e2e/test_modal_inference_e2e.py",
        "tests/e2e/test_cronworkflow_execution_heavy_e2e.py",
    }, sorted(exclusion_files)
    assert exclusion_reasons["tests/e2e/test_dashboard_e2e.py"] == (
        "browser lane via pytest.mark.browser"
    )
    assert exclusion_reasons["tests/e2e/test_modal_inference_e2e.py"] == (
        "requires_modal_inference collection gate"
    )
    assert exclusion_reasons["tests/e2e/test_cronworkflow_execution_heavy_e2e.py"] == (
        "e2e_heavy marker"
    )

    assert batched_files.isdisjoint(exclusion_files), (
        "files may not appear in both batches and exclusions: "
        f"{sorted(batched_files & exclusion_files)}"
    )
    assert batched_files.isdisjoint(uncovered_files), (
        "files may not appear in both batches and uncovered: "
        f"{sorted(batched_files & uncovered_files)}"
    )
    assert exclusion_files.isdisjoint(uncovered_files), (
        "files may not appear in both exclusions and uncovered: "
        f"{sorted(exclusion_files & uncovered_files)}"
    )

    covered_files = batched_files | exclusion_files | uncovered_files
    missing = filesystem_files - covered_files
    extra = covered_files - filesystem_files
    assert missing == set() and extra == set(), (
        "every tests/e2e/test_*.py file must be in a batch, an explicit "
        f"script exclusion, or the uncovered ratchet; missing={sorted(missing)} "
        f"extra={sorted(extra)}"
    )
    assert uncovered_files <= EXPECTED_E2E_BATCH_UNCOVERED, (
        "E2E_BATCH_UNCOVERED may only shrink; added="
        f"{sorted(uncovered_files - EXPECTED_E2E_BATCH_UNCOVERED)}"
    )


_REPO = Path(__file__).resolve().parents[3]


def _run_lock_module():
    path = _REPO / "tests" / "e2e" / "run_lock.py"
    assert path.exists(), (
        "tests/e2e/run_lock.py is missing: an e2e session takes no run lock, so a "
        "lane can start a second run against the cluster"
    )
    return importlib.import_module("tests.e2e.run_lock")


def _acquire_in_child(lock_file, *, wait_for=None, hold=0.0):
    """Acquire the lock in a separate process; print WON / INHERITED / REFUSED."""
    source = f"""
import pathlib, sys, time, importlib.util
spec = importlib.util.spec_from_file_location(
    "e2e_run_lock", {str(_REPO / "tests" / "e2e" / "run_lock.py")!r}
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
gate = {str(wait_for) if wait_for else None!r}
if gate:
    while not pathlib.Path(gate).exists():
        time.sleep(0.005)
try:
    owned = mod.acquire({str(lock_file)!r})
    print("WON" if owned else "INHERITED", flush=True)
    time.sleep({hold})
except mod.E2ERunLockError as exc:
    print("REFUSED", exc, flush=True)
"""
    return subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_e2e_run_lock_refuses_a_session_while_a_foreign_process_holds_it(tmp_path):
    run_lock = _run_lock_module()
    lock = tmp_path / "e2e.lock"
    with _detached(["sleep", "120"]) as foreign:
        lock.write_text(f"{foreign.pid}\n")

        with pytest.raises(run_lock.E2ERunLockError) as raised:
            run_lock.acquire(lock)

    assert f"pid {foreign.pid}" in str(raised.value), str(raised.value)
    assert lock.read_text() == f"{foreign.pid}\n", "a refused session rewrote the lock"


def test_e2e_run_lock_inherits_the_lock_from_its_own_launcher(tmp_path):
    """The batched runner takes the lock, then launches pytest under it."""
    lock = tmp_path / "e2e.lock"
    lock.write_text(f"{os.getpid()}\n")

    child = _acquire_in_child(lock)
    stdout, stderr = child.communicate(timeout=60)

    assert stdout.strip() == "INHERITED", f"{stdout!r} {stderr!r}"
    assert lock.read_text() == f"{os.getpid()}\n", "the child stole its launcher's lock"


def test_e2e_run_lock_takes_over_a_lock_whose_holder_is_dead(tmp_path):
    run_lock = _run_lock_module()
    lock = tmp_path / "e2e.lock"
    with _detached(["sleep", "120"]) as corpse:
        dead_pid = corpse.pid
    lock.write_text(f"{dead_pid}\n")

    assert run_lock.acquire(lock) is True
    assert lock.read_text().strip() == str(os.getpid())
    run_lock.release(lock)


def test_e2e_run_lock_grants_exactly_one_winner_under_concurrent_acquisition(tmp_path):
    lock = tmp_path / "e2e.lock"
    gate = tmp_path / "go"

    children = [_acquire_in_child(lock, wait_for=gate, hold=3.0) for _ in range(8)]
    gate.write_text("go")
    verdicts = [c.communicate(timeout=90)[0].split()[0] for c in children]

    assert verdicts.count("WON") == 1, verdicts
    assert verdicts.count("REFUSED") == 7, verdicts


def test_e2e_run_lock_fails_closed_when_the_lock_cannot_be_written(tmp_path):
    run_lock = _run_lock_module()
    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o500)

    with pytest.raises(run_lock.E2ERunLockError) as raised:
        run_lock.acquire(sealed / "e2e.lock")

    assert "e2e.lock" in str(raised.value), str(raised.value)


def test_e2e_run_lock_release_leaves_a_foreign_lock_intact(tmp_path):
    run_lock = _run_lock_module()
    lock = tmp_path / "e2e.lock"
    with _detached(["sleep", "120"]) as foreign:
        lock.write_text(f"{foreign.pid}\n")

        run_lock.release(lock)

    assert lock.read_text() == f"{foreign.pid}\n"


def test_e2e_stack_fixture_acquires_the_run_lock_before_touching_the_cluster():
    e2e_conftest = importlib.import_module("tests.e2e.conftest")
    source = inspect.getsource(e2e_conftest.e2e_stack.__wrapped__)
    acquire_at = source.index("run_lock.acquire(")
    assert acquire_at < source.index("_ensure_host_sandbox_gateway(")
    assert acquire_at < source.index("_e2e_cluster_state(")
