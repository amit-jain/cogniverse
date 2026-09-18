"""The egress drift check resolves each service URL to the port it connects to."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cogniverse_foundation.config.utils as config_utils
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:acme"
MODAL_LM = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run"


def _dispatcher(monkeypatch, api_base, denseon_url):
    system_config = SimpleNamespace(
        backend_url="http://vespa.internal",
        backend_port=8080,
        inference_service_urls={"denseon": denseon_url},
    )
    llm_config = SimpleNamespace(primary=SimpleNamespace(api_base=api_base))
    monkeypatch.setattr(
        config_utils,
        "get_config",
        lambda tenant_id, config_manager: SimpleNamespace(
            get_llm_config=lambda: llm_config
        ),
    )
    d = object.__new__(AgentDispatcher)
    d._config_manager = SimpleNamespace(get_system_config=lambda: system_config)
    return d


@pytest.mark.parametrize(
    ("api_base", "denseon_url", "llm_port", "denseon_port"),
    [
        (MODAL_LM, "https://denseon.internal", 443, 443),
        (f"{MODAL_LM}/v1", "http://denseon.internal/embed", 443, 80),
        (
            "http://amit-jain--cogniverse-vllm-llm-student-inference.modal.run",
            "denseon.internal",
            80,
            8000,
        ),
        (
            "amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1",
            "denseon.internal:9100",
            11434,
            9100,
        ),
        (f"{MODAL_LM}:8443/v1", "https://denseon.internal:8001", 8443, 8001),
    ],
)
def test_port_comes_from_the_url_then_its_scheme_then_the_service_default(
    monkeypatch, api_base, denseon_url, llm_port, denseon_port
):
    dispatcher = _dispatcher(monkeypatch, api_base, denseon_url)

    assert dispatcher._system_endpoints(TENANT) == {
        "vespa": {"host": "vespa.internal", "port": 8080, "protocol": "tcp"},
        "llm": {
            "host": "amit-jain--cogniverse-vllm-llm-student-inference.modal.run",
            "port": llm_port,
            "protocol": "tcp",
        },
        "denseon": {
            "host": "denseon.internal",
            "port": denseon_port,
            "protocol": "tcp",
        },
    }


def test_https_lm_on_its_allowlisted_port_is_not_drift(monkeypatch):
    dispatcher = _dispatcher(monkeypatch, f"{MODAL_LM}/v1", "http://denseon:8000")
    host = "amit-jain--cogniverse-vllm-llm-student-inference.modal.run"
    dispatcher._sandbox_manager = SimpleNamespace(
        get_policy=lambda agent_name: {
            "egress": {"allow": [{"host": host, "port": 443, "protocol": "tcp"}]}
        }
    )
    dispatcher._egress_kinds_for = lambda agent_name: frozenset({"llm"})

    assert dispatcher._verify_egress("orchestrator_agent", TENANT) == []

    dispatcher._sandbox_manager = SimpleNamespace(
        get_policy=lambda agent_name: {
            "egress": {"allow": [{"host": host, "port": 11434, "protocol": "tcp"}]}
        }
    )
    assert dispatcher._verify_egress("orchestrator_agent", TENANT) == [
        {
            "host": host,
            "port": 443,
            "protocol": "tcp",
            "reason": (
                f"host={host} port=443 protocol=tcp "
                "not in egress allowlist for agent=orchestrator_agent"
            ),
        }
    ]
