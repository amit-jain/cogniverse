"""A tenant's stored tier steers the deployed router on the production path.

Set the tier through the deployed runtime's admin route, send one real query
through the dispatch route, and reconcile what the deployed semantic router
recorded: the per-decision counters it exports against the routed calls its own
log attributes to this tenant. The runtime resolves the tier itself from its
config store, so this exercises the whole chain the cluster runs: admin write
-> the runtime's per-tenant reader (and its in-process invalidation) -> the
router's decision.

One summarizer dispatch makes one routed LM call per DSPy signature it runs -
the search-query rewrite and the summary - so the counter moves by that many,
derived from those signatures rather than restated. A decision whose only rule
is an authz role matches exactly the calls carrying that role, so every
counter movement in the window is attributable, this tenant's and anyone
else's, and a concurrent tenant cannot move a pin.

Each leg carries its own query text. Identical messages with identical routing
headers hit the runtime's DSPy cache, which answers without reaching the router
at all; the last leg repeats the previous leg's query verbatim to pin that -
zero routed calls, and no movement in the router's own response-cache counter,
which is a different layer.

On this chart both catalog models resolve to the same upstream and the same
``provider_model_id`` (``charts/cogniverse/files/semantic-router/config.yaml``
lines 25-44), so promoting a tenant does NOT change which model answers. What
it changes is the decision the router matches and the reasoning flag it sends,
which is what this reads.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import yaml

from cogniverse_agents.search_agent import SearchOptimizationSignature
from cogniverse_agents.summarizer_agent import SummaryGenerationSignature
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import ROUTER_TIERS
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    RUNTIME,
    register_tenant_and_wait,
    unique_id,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

# The release namespace the e2e cluster deploys into, as every other kubectl
# read in tests/e2e/conftest.py spells it.
NAMESPACE = "cogniverse"
_ROUTER_SVC = "cogniverse-semantic-router"
_ROUTER_DEPLOY = f"deploy/{_ROUTER_SVC}"
_ROUTER_METRICS_PORT = 9190
CHART_ROUTER_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "charts"
    / "cogniverse"
    / "files"
    / "semantic-router"
    / "config.yaml"
)
QUERY = "summarise what this tenant has ingested"
AGENT = "summarizer_agent"

# The input-field sets of the DSPy signatures one summarizer dispatch runs, as
# the served modules declare them. The router logs each request's rendered user
# message, whose ``[[ ## field ## ]]`` markers are exactly the signature's
# input fields, so this is what identifies a routed call and how many there are.
DISPATCH_SIGNATURE_INPUTS = frozenset(
    {
        frozenset(SearchOptimizationSignature.input_fields),
        frozenset(SummaryGenerationSignature.input_fields),
    }
)

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[0-9.eE+-]+)$"
)
_AUTHZ_MATCHED = re.compile(
    r'^\[Authz Signal\] Matched \d+ roles for user "(?P<user>[^"]*)": '
    r"\[(?P<roles>[^\]]*)\]$"
)
_CACHE_UPDATED = re.compile(
    r"^Cache updated for request ID: (?P<request_id>[0-9a-fA-F-]+)$"
)
_PROMPT_FIELD = re.compile(r"\[\[ ## ([a-zA-Z0-9_]+) ## \]\]")


@dataclass(frozen=True)
class RouterPolicy:
    """What the chart's routing section binds, keyed the way the router reports."""

    role_by_tier: dict[str, str]
    tier_by_decision: dict[str, str]
    model_by_decision: dict[str, str]
    default_decision_by_tier: dict[str, str]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _router_policy() -> RouterPolicy:
    """Read the chart's own role bindings and decisions.

    The Go templating in the model blocks is irrelevant to the routing section,
    which is literal.
    """
    text = "\n".join(
        line
        for line in CHART_ROUTER_CONFIG.read_text().splitlines()
        if "{{" not in line
    )
    routing = yaml.safe_load(text)["routing"]
    tier_by_role = {
        binding["role"]: subject["name"]
        for binding in routing["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }
    tier_by_decision: dict[str, str] = {}
    model_by_decision: dict[str, str] = {}
    default_decision_by_tier: dict[str, str] = {}
    for decision in routing["decisions"]:
        conditions = decision["rules"]["conditions"]
        roles = {c["name"] for c in conditions if c["type"] == "authz"}
        if not roles:
            continue
        (role,) = roles
        name = decision["name"]
        tier_by_decision[name] = tier_by_role[role]
        model_by_decision[name] = decision["modelRefs"][0]["model"]
        if len(conditions) == 1:
            default_decision_by_tier[tier_by_role[role]] = name
    return RouterPolicy(
        role_by_tier={tier: role for role, tier in tier_by_role.items()},
        tier_by_decision=tier_by_decision,
        model_by_decision=model_by_decision,
        default_decision_by_tier=default_decision_by_tier,
    )


@pytest.fixture(scope="module")
def router_metrics_url():
    """kubectl port-forward to the deployed router's metrics endpoint."""
    local = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "port-forward",
            f"svc/{_ROUTER_SVC}",
            f"{local}:{_ROUTER_METRICS_PORT}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{local}/metrics"
    deadline = time.time() + 60
    try:
        while True:
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    break
            except (OSError, httpx.HTTPError):
                pass
            if time.time() >= deadline:
                proc.terminate()
                pytest.fail(
                    f"port-forward to {_ROUTER_SVC}:{_ROUTER_METRICS_PORT} did not "
                    f"serve /metrics in 60s"
                )
            time.sleep(2)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _samples(metrics_url: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    body = httpx.get(metrics_url, timeout=30).text
    out: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        labels: tuple[tuple[str, str], ...] = ()
        if match.group("labels"):
            labels = tuple(
                sorted(
                    (key.strip(), value.strip().strip('"'))
                    for key, _, value in (
                        item.partition("=") for item in match.group("labels").split(",")
                    )
                )
            )
        out[(match.group("name"), labels)] = float(match.group("value"))
    return out


def _counter(samples: dict, name: str, **labels: str) -> int:
    return int(samples.get((name, tuple(sorted(labels.items()))), 0.0))


def _router_entries(start: datetime, end: datetime) -> list[dict]:
    """The router's own JSON log lines stamped inside ``[start, end]``."""
    since = (start - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    proc = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "logs",
            _ROUTER_DEPLOY,
            f"--since-time={since}",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"reading {_ROUTER_DEPLOY} logs since {since} failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()[:400]}"
        )
    entries = []
    for raw in proc.stdout.splitlines():
        if not raw.startswith("{"):
            continue
        try:
            entry = json.loads(raw)
            stamp = datetime.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if start <= stamp <= end:
            entries.append(entry)
    return entries


@dataclass(frozen=True)
class RoutedTraffic:
    """Every routed call the router logged in one window, by who made it."""

    roles: list[tuple[str, str]]
    prompt_by_request: dict[str, str]
    call_by_request: dict[str, tuple[str, str]]
    cache_hits: int


def _routed_traffic(entries: list[dict]) -> RoutedTraffic:
    roles: list[tuple[str, str]] = []
    prompt_by_request: dict[str, str] = {}
    call_by_request: dict[str, tuple[str, str]] = {}
    cache_hits = 0
    pending_prompt: str | None = None
    for entry in entries:
        event = entry.get("event")
        message = entry.get("msg", "")
        if event == "cache_entry_added":
            pending_prompt = entry.get("query", "")
            continue
        if event == "routing_decision":
            call_by_request[entry["request_id"]] = (
                entry["decision"],
                entry["selected_model"],
            )
            continue
        if event == "llm_usage" and entry.get("cache_hit"):
            cache_hits += 1
            continue
        updated = _CACHE_UPDATED.match(message)
        if updated is not None:
            if pending_prompt is not None:
                prompt_by_request[updated.group("request_id")] = pending_prompt
            pending_prompt = None
            continue
        matched = _AUTHZ_MATCHED.match(message)
        if matched is not None:
            for role in matched.group("roles").split(","):
                roles.append((matched.group("user"), role.strip()))
    return RoutedTraffic(roles, prompt_by_request, call_by_request, cache_hits)


def _set_tier(tenant_id: str, tier: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.put(
            f"{RUNTIME}/admin/tenants/{tenant_id}/tier", json={"tier": tier}
        )
    assert response.status_code == 200, response.text
    return response.json()


def _dispatch_one_query(tenant_id: str, query: str) -> None:
    """One real query through the production dispatch route.

    Body shape is the route's ``AgentTask``: agent_name + query, tenant inside
    ``context``.
    """
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"{RUNTIME}/agents/{AGENT}/process",
            json={
                "agent_name": AGENT,
                "query": query,
                "context": {"tenant_id": tenant_id},
            },
        )
    assert response.status_code == 200, response.text[:600]


@dataclass(frozen=True)
class Leg:
    """One dispatch, as the router's counters and its log both describe it."""

    deltas: dict[str, int]
    expected_deltas: dict[str, int]
    cache_hit_delta: int
    logged_cache_hits: int
    mine_by_role: Counter
    decisions: list[str]
    selected_models: list[str]
    prompt_fields: set[frozenset[str]]

    def __str__(self) -> str:
        return (
            f"counter deltas {self.deltas} vs {self.expected_deltas} derived from "
            f"the log; this tenant's routed calls {self.mine_by_role} -> "
            f"decisions {self.decisions} on models {self.selected_models} with "
            f"signature inputs {sorted(sorted(f) for f in self.prompt_fields)}; "
            f"router response-cache hits {self.cache_hit_delta} counted / "
            f"{self.logged_cache_hits} logged"
        )


def _run_leg(
    metrics_url: str,
    tenant_id: str,
    canonical: str,
    query: str,
    policy: RouterPolicy,
) -> Leg:
    tiers = sorted(ROUTER_TIERS)
    start = datetime.now(timezone.utc)
    before = _samples(metrics_url)
    _dispatch_one_query(tenant_id, query)
    after = _samples(metrics_url)
    end = datetime.now(timezone.utc)

    traffic = _routed_traffic(_router_entries(start, end))
    mine = {
        request_id
        for request_id, prompt in traffic.prompt_by_request.items()
        if query in prompt
    }
    calls = [
        traffic.call_by_request[r] for r in sorted(mine) if r in traffic.call_by_request
    ]
    role_counts = Counter(role for _, role in traffic.roles)

    def decision(tier: str) -> str:
        return policy.default_decision_by_tier[tier]

    return Leg(
        deltas={
            tier: _counter(
                after, "llm_decision_match_total", decision_name=decision(tier)
            )
            - _counter(before, "llm_decision_match_total", decision_name=decision(tier))
            for tier in tiers
        },
        expected_deltas={
            tier: role_counts[policy.role_by_tier[tier]] for tier in tiers
        },
        cache_hit_delta=sum(
            _counter(
                after,
                "llm_cache_plugin_hits_total",
                decision_name=decision(tier),
                plugin_type="response_cache",
            )
            - _counter(
                before,
                "llm_cache_plugin_hits_total",
                decision_name=decision(tier),
                plugin_type="response_cache",
            )
            for tier in tiers
        ),
        logged_cache_hits=traffic.cache_hits,
        mine_by_role=Counter(role for user, role in traffic.roles if user == canonical),
        decisions=[name for name, _ in calls],
        selected_models=[model for _, model in calls],
        prompt_fields={
            frozenset(_PROMPT_FIELD.findall(traffic.prompt_by_request[r])) for r in mine
        },
    )


def test_the_stored_tier_steers_the_deployed_router(router_metrics_url):
    """Each tier set through the admin route changes the decision the deployed
    router matches for the next query on the production dispatch path, and a
    repeated query never reaches the router at all."""
    policy = _router_policy()
    assert set(policy.default_decision_by_tier) == set(ROUTER_TIERS)

    tenant_id = unique_id("tier")
    register_tenant_and_wait(tenant_id)
    canonical = canonical_tenant_id(tenant_id)

    walk = sorted(ROUTER_TIERS)
    walk.append(walk[0])
    legs = [(tier, f"{QUERY} ({tenant_id} leg {i})") for i, tier in enumerate(walk)]
    legs.append(legs[-1])

    routed = len(DISPATCH_SIGNATURE_INPUTS)
    for index, (tier, query) in enumerate(legs):
        repeat = index == len(legs) - 1
        assert _set_tier(tenant_id, tier) == {
            "tenant_id": canonical,
            "tier": tier,
        }
        leg = _run_leg(router_metrics_url, tenant_id, canonical, query, policy)
        where = f"leg {index} on tier {tier!r} (repeat={repeat}): {leg}"

        assert leg.mine_by_role == (
            Counter() if repeat else Counter({policy.role_by_tier[tier]: routed})
        ), where
        assert [policy.tier_by_decision[name] for name in leg.decisions] == (
            [] if repeat else [tier] * routed
        ), where
        assert leg.selected_models == [
            policy.model_by_decision[name] for name in leg.decisions
        ], where
        assert leg.prompt_fields == (
            set() if repeat else set(DISPATCH_SIGNATURE_INPUTS)
        ), where
        assert leg.deltas == leg.expected_deltas, where
        assert leg.cache_hit_delta == leg.logged_cache_hits, where
