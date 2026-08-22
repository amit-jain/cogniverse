from __future__ import annotations

import importlib
import inspect
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.e2e.test_manual_optimization_e2e as manual_optimization
import tests.e2e.test_messaging_e2e as messaging
import tests.e2e.test_quality_monitor_e2e as quality_monitor
from tests.e2e.conftest import _telegram_real_flow_deselections


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
