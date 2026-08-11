from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

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


def test_telegram_real_flow_is_deselected_without_required_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TEST_CHAT_ID", raising=False)
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
