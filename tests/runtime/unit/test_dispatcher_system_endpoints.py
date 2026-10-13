"""The egress drift check resolves each service URL to the port it connects to."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import cogniverse_foundation.config.utils as config_utils
from cogniverse_foundation.config.unified_config import (
    SemanticRouterConfig,
    SystemConfig,
)
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.sandbox_http import (
    DEFAULT_LLM_ADDRESS,
    default_service_addresses,
)

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


REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_POLICIES = {
    path.stem: yaml.safe_load(path.read_text())
    for path in sorted((REPO_ROOT / "configs" / "agent_policies").glob("*.yaml"))
}
SHIPPED_AGENTS = json.loads((REPO_ROOT / "configs" / "config.json").read_text())[
    "agents"
]
# A cluster deployment: the backend and the LLM are served at service
# addresses, not at the defaults the policy rules name them by.
DEPLOYED_VESPA = {"host": "cogniverse-vespa", "port": 8080, "protocol": "tcp"}
ROUTER_URL = "http://cogniverse-semantic-router-envoy:8801/v1"
DEPLOYED_ROUTER = {
    "host": "cogniverse-semantic-router-envoy",
    "port": 8801,
    "protocol": "tcp",
}
DEPLOYED = {"vespa": DEPLOYED_VESPA, "llm": DEPLOYED_ROUTER}


def _deployed_dispatcher(monkeypatch, *, router_enabled=True, policies=None):
    system_config = SimpleNamespace(
        backend_url="http://cogniverse-vespa",
        backend_port=8080,
        inference_service_urls={},
    )
    llm_config = SimpleNamespace(primary=SimpleNamespace(api_base=f"{MODAL_LM}/v1"))
    router = SemanticRouterConfig(
        enabled=router_enabled, semantic_router_url=ROUTER_URL
    )
    monkeypatch.setattr(
        config_utils,
        "get_config",
        lambda tenant_id, config_manager: SimpleNamespace(
            get_llm_config=lambda: llm_config,
            get_semantic_router=lambda: router,
        ),
    )
    policies = SHIPPED_POLICIES if policies is None else policies
    d = object.__new__(AgentDispatcher)
    d._config_manager = SimpleNamespace(get_system_config=lambda: system_config)
    d._sandbox_manager = SimpleNamespace(get_policy=policies.get)
    d._registry = SimpleNamespace(
        get_agent=lambda name: (
            SimpleNamespace(capabilities=SHIPPED_AGENTS[name].get("capabilities"))
            if name in SHIPPED_AGENTS
            else None
        )
    )
    return d


def _rule_addresses(policy) -> set:
    return {
        (rule["host"], rule["port"]) for rule in policy["network_policies"]["egress"]
    }


def _undeclared_kinds(dispatcher, agent_name) -> list:
    """Endpoint kinds the agent reaches whose default address its shipped
    policy does not name, in the order the dispatcher checks them."""
    declared = _rule_addresses(SHIPPED_POLICIES[agent_name])
    reached = dispatcher._egress_kinds_for(agent_name)
    return [
        kind
        for kind, address in default_service_addresses().items()
        if kind in reached and address not in declared
    ]


def _violation(endpoint, agent_name):
    return {
        **endpoint,
        "reason": (
            f"host={endpoint['host']} port={endpoint['port']} protocol=tcp "
            f"not in egress allowlist for agent={agent_name}"
        ),
    }


def _drift_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records if "DRIFT" in r.getMessage()]


def test_policy_rules_name_services_at_these_default_addresses():
    default = SystemConfig()
    assert default_service_addresses() == {
        "vespa": (
            default.backend_url.removeprefix("http://"),
            default.backend_port,
        ),
        "llm": DEFAULT_LLM_ADDRESS,
    }
    assert DEFAULT_LLM_ADDRESS == ("localhost", 11434)


def test_the_llm_endpoint_is_the_router_when_calls_are_routed(monkeypatch):
    routed = _deployed_dispatcher(monkeypatch, router_enabled=True)
    assert routed._system_endpoints(TENANT) == DEPLOYED

    direct = _deployed_dispatcher(monkeypatch, router_enabled=False)
    assert direct._system_endpoints(TENANT) == {
        "vespa": DEPLOYED_VESPA,
        "llm": {
            "host": "amit-jain--cogniverse-vllm-llm-student-inference.modal.run",
            "port": 443,
            "protocol": "tcp",
        },
    }


@pytest.mark.parametrize("router_enabled", [True, False])
@pytest.mark.parametrize("agent_name", sorted(SHIPPED_POLICIES))
def test_a_deployment_drifts_only_where_a_shipped_policy_omits_a_service(
    monkeypatch, caplog, agent_name, router_enabled
):
    """A rule for a service's default address admits the address the
    deployment serves it at; a service the policy does not name is drift."""
    dispatcher = _deployed_dispatcher(monkeypatch, router_enabled=router_enabled)
    deployed = dispatcher._system_endpoints(TENANT)
    undeclared = _undeclared_kinds(dispatcher, agent_name)
    expected = [_violation(deployed[kind], agent_name) for kind in undeclared]

    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"):
        violations = dispatcher._verify_egress(agent_name, TENANT)

    assert violations == expected
    assert _drift_lines(caplog) == (
        [
            f"egress policy DRIFT for {agent_name}: {len(expected)} endpoint(s) "
            f"outside allowlist; first violation: {expected[0]['reason']}"
        ]
        if expected
        else []
    )


def test_the_agents_the_fast_path_dispatches_do_not_drift(monkeypatch, caplog):
    dispatcher = _deployed_dispatcher(monkeypatch)
    fast_path = ["routing_agent", "search_agent"]

    assert [_undeclared_kinds(dispatcher, name) for name in fast_path] == [[], []]
    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"):
        assert [dispatcher._verify_egress(name, TENANT) for name in fast_path] == [
            [],
            [],
        ]
    assert _drift_lines(caplog) == []


def test_a_policy_without_the_vespa_rule_drifts_on_the_deployed_backend(
    monkeypatch, caplog
):
    policy = yaml.safe_load(yaml.safe_dump(SHIPPED_POLICIES["search_agent"]))
    vespa_default = default_service_addresses()["vespa"]
    policy["network_policies"]["egress"] = [
        rule
        for rule in policy["network_policies"]["egress"]
        if (rule["host"], rule["port"]) != vespa_default
    ]
    dispatcher = _deployed_dispatcher(monkeypatch, policies={"search_agent": policy})

    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.agent_dispatcher"):
        violations = dispatcher._verify_egress("search_agent", TENANT)

    assert violations == [_violation(DEPLOYED_VESPA, "search_agent")]
    assert _drift_lines(caplog) == [
        "egress policy DRIFT for search_agent: 1 endpoint(s) outside allowlist; "
        "first violation: host=cogniverse-vespa port=8080 protocol=tcp not in "
        "egress allowlist for agent=search_agent"
    ]


def test_a_binding_admits_only_the_bound_address_and_protocol(monkeypatch):
    dispatcher = _deployed_dispatcher(monkeypatch)
    bindings = {("localhost", 8080): ("cogniverse-vespa", 8080)}

    assert dispatcher.validate_dispatch_endpoints(
        "search_agent",
        [
            {"host": "cogniverse-vespa", "port": 8080},
            {"host": "localhost", "port": 8080},
            {"host": "cogniverse-vespa", "port": 19071},
            {"host": "cogniverse-vespa", "port": 8080, "protocol": "udp"},
            {"host": "other-vespa", "port": 8080},
        ],
        endpoint_bindings=bindings,
    ) == [
        _violation({"host": "cogniverse-vespa", "port": 19071}, "search_agent")
        | {
            "reason": "host=cogniverse-vespa port=19071 protocol=tcp not in "
            "egress allowlist for agent=search_agent"
        },
        {
            "host": "cogniverse-vespa",
            "port": 8080,
            "protocol": "udp",
            "reason": "host=cogniverse-vespa port=8080 protocol=udp not in "
            "egress allowlist for agent=search_agent",
        },
        _violation({"host": "other-vespa", "port": 8080}, "search_agent"),
    ]
