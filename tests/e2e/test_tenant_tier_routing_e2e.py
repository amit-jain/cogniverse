"""A tenant's stored tier steers the deployed router on the production path.

Set the tier through the deployed runtime's admin route, send one real query
through the dispatch route, and reconcile what the deployed stack recorded:
the router's per-decision counters against its own routed-call log, Envoy's
access log for which cluster served each call, and the runtime's spans for
which model answered. The runtime resolves the tier itself from its config
store, so this exercises the whole chain the cluster runs: admin write -> the
runtime's per-tenant reader (and its in-process invalidation) -> the router's
decision -> the backend Envoy dials -> the model the backend reports.

One summarizer dispatch makes one routed LM call per DSPy signature it runs -
the search-query rewrite and the summary - so the routed count is derived from
those signatures rather than restated. The rewrite is a bounded call and enters
on the classification entrypoint, whose decisions serve basic-chat for every
tier; the summary enters on the auto alias, where a pro tenant's decision
serves pro-reasoning. Every counter movement in the window is attributable per
decision, this tenant's and anyone else's, so a concurrent tenant cannot move
a pin.

The deployed chart binds each catalog model to a backend - basic-chat to the
student, pro-reasoning to the teacher - on their own Envoy clusters. Beyond
the decision name, a tier change is observed as the cluster Envoy logs for the
call and the served model the runtime stamps on its span from the completion's
own ``model`` field. Both expectations are read from the ConfigMaps the
cluster runs, never restated here.

Each leg carries its own query text. Identical messages with identical routing
headers hit the runtime's DSPy cache, which answers without reaching the router
at all; the last leg repeats the previous leg's query verbatim to pin that -
zero routed calls and no movement in the router's own response-cache counter,
which is a different layer - while the replayed completion still names the
model that produced it.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import httpx
import pytest
import yaml

from cogniverse_agents.search_agent import SearchOptimizationSignature
from cogniverse_agents.summarizer_agent import SummaryGenerationSignature
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.span_contract import (
    LLM_SERVED_MODEL_ATTRIBUTE,
    read_span_attributes,
)
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    RUNTIME,
    SEEDED_TENANT_TIER,
    TENANT_ID,
    bootstrap_seeded_tenant_tier,
    register_tenant_and_wait,
    unique_id,
)
from tests.e2e.loop_probe import LoopProbe, assert_loop_served

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

# The release namespace the e2e cluster deploys into, as every other kubectl
# read in tests/e2e/conftest.py spells it.
NAMESPACE = "cogniverse"
_ROUTER_SVC = "cogniverse-semantic-router"
_ROUTER_DEPLOY = f"deploy/{_ROUTER_SVC}"
_ENVOY_DEPLOY = f"deploy/{_ROUTER_SVC}-envoy"
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
# Spans reach Phoenix through the runtime's batch exporter; measured on this
# cluster the served-model attribute is readable 4-9 s after the dispatch
# returns, so the read polls up to this long before it reports what it saw.
SERVED_MODEL_READ_BUDGET_S = 90.0

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
    """What the chart's routing section binds, keyed the way the router reports.

    Every decision the router can name - the auto alias's and every recipe's.
    """

    role_by_tier: dict[str, str]
    tier_by_decision: dict[str, str]
    model_by_decision: dict[str, str]
    metric_label_by_decision: dict[str, str]
    decisions: tuple[str, ...]
    conditions_by_decision: dict[str, frozenset[tuple[str, str]]]
    priority_by_decision: dict[str, int]
    section_by_decision: dict[str, str]


@dataclass(frozen=True)
class DeployedRouting:
    """What the ConfigMaps the cluster runs bind each catalog model to."""

    cluster_by_model: dict[str, str]
    default_cluster: str
    served_model_by_catalog: dict[str, str]

    def cluster_for(self, catalog_model: str) -> str:
        return self.cluster_by_model.get(catalog_model, self.default_cluster)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _router_policy() -> RouterPolicy:
    """Read the chart's own role bindings and decisions, recipes included.

    The Go templating in the model blocks is irrelevant to the routing
    sections, which are literal.
    """
    text = "\n".join(
        line
        for line in CHART_ROUTER_CONFIG.read_text().splitlines()
        if "{{" not in line
    )
    document = yaml.safe_load(text)
    routing = document["routing"]
    tier_by_role = {
        binding["role"]: subject["name"]
        for binding in routing["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }
    # The router's counters label a recipe's decision as ``<recipe>::<name>``
    # and the auto alias's decisions by bare name.
    decisions = [(decision, decision["name"], "") for decision in routing["decisions"]]
    for recipe in document.get("recipes", []):
        decisions += [
            (decision, f"{recipe['name']}::{decision['name']}", recipe["name"])
            for decision in recipe["routing"]["decisions"]
        ]
    tier_by_decision: dict[str, str] = {}
    model_by_decision: dict[str, str] = {}
    metric_label_by_decision: dict[str, str] = {}
    conditions_by_decision: dict[str, frozenset[tuple[str, str]]] = {}
    priority_by_decision: dict[str, int] = {}
    section_by_decision: dict[str, str] = {}
    for decision, metric_label, section in decisions:
        name = decision["name"]
        # _matched_decisions relies on AND rules and pure priority selection.
        if decision["rules"]["operator"] != "AND" or "tier" in decision:
            raise AssertionError(
                f"decision {name} is not an AND rule under priority selection"
            )
        conditions = decision["rules"]["conditions"]
        roles = {c["name"] for c in conditions if c["type"] == "authz"}
        (role,) = roles
        tier_by_decision[name] = tier_by_role[role]
        (ref,) = decision["modelRefs"]
        model_by_decision[name] = ref["model"]
        metric_label_by_decision[name] = metric_label
        conditions_by_decision[name] = frozenset(
            (c["type"], c["name"]) for c in conditions
        )
        priority_by_decision[name] = decision["priority"]
        section_by_decision[name] = section
    return RouterPolicy(
        role_by_tier={tier: role for role, tier in tier_by_role.items()},
        tier_by_decision=tier_by_decision,
        model_by_decision=model_by_decision,
        metric_label_by_decision=metric_label_by_decision,
        decisions=tuple(tier_by_decision),
        conditions_by_decision=conditions_by_decision,
        priority_by_decision=priority_by_decision,
        section_by_decision=section_by_decision,
    )


def _kubectl(*args: str, timeout: int = 60) -> str:
    proc = subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "-n", NAMESPACE, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"kubectl {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    return proc.stdout


def _deployed_configmap(name: str, key: str) -> str:
    return _kubectl(
        "get",
        "configmap",
        name,
        "-o",
        "jsonpath={.data." + key.replace(".", r"\.") + "}",
    )


def _deployed_routing() -> DeployedRouting:
    """The clusters and served models the cluster's own ConfigMaps bind."""
    envoy = yaml.safe_load(_deployed_configmap(f"{_ROUTER_SVC}-envoy", "envoy.yaml"))
    listener = envoy["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    (vhost,) = hcm["route_config"]["virtual_hosts"]
    cluster_by_model: dict[str, str] = {}
    catch_all: list[str] = []
    for route in vhost["routes"]:
        headers = route["match"].get("headers", [])
        if not headers:
            catch_all.append(route["route"]["cluster"])
            continue
        (header,) = headers
        cluster_by_model[header["string_match"]["exact"]] = route["route"]["cluster"]
    (default_cluster,) = catch_all
    config = yaml.safe_load(_deployed_configmap(f"{_ROUTER_SVC}-config", "config.yaml"))
    return DeployedRouting(
        cluster_by_model=cluster_by_model,
        default_cluster=default_cluster,
        served_model_by_catalog={
            model["name"]: model["provider_model_id"]
            for model in config["providers"]["models"]
        },
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


def _router_stamp(entry: dict) -> datetime:
    return datetime.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=timezone.utc
    )


def _envoy_stamp(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))


def _json_log_entries(
    deploy: str, start: datetime, end: datetime, stamp: Callable[[dict], datetime]
) -> list[dict]:
    """A deployment's JSON log lines stamped inside ``[start, end]``."""
    since = (start - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stdout = _kubectl("logs", deploy, f"--since-time={since}", timeout=180)
    entries = []
    for raw in stdout.splitlines():
        if not raw.startswith("{"):
            continue
        try:
            entry = json.loads(raw)
            at = stamp(entry)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if start <= at <= end:
            entries.append(entry)
    return entries


@dataclass(frozen=True)
class RoutedTraffic:
    """Every routed call the router logged in one window, by who made it.

    The router logs a request's authz match (which names the user) and its
    routing decision (which names the request id) as adjacent lines on one
    goroutine, so the user of each request is the last authz match before
    its decision. That attributes every attempt to its tenant, including one
    the client gave up on before it completed.
    """

    roles: list[tuple[str, str]]
    user_by_request: dict[str, str]
    prompt_by_request: dict[str, str]
    call_by_request: dict[str, tuple[str, str]]
    latency_ms_by_request: dict[str, int]
    cache_hits: int
    roles_by_request: dict[str, frozenset[str]]


def _routed_traffic(entries: list[dict]) -> RoutedTraffic:
    roles: list[tuple[str, str]] = []
    user_by_request: dict[str, str] = {}
    prompt_by_request: dict[str, str] = {}
    call_by_request: dict[str, tuple[str, str]] = {}
    roles_by_request: dict[str, frozenset[str]] = {}
    latency_ms_by_request: dict[str, int] = {}
    cache_hits = 0
    pending_prompt: str | None = None
    pending_user: str | None = None
    pending_roles: frozenset[str] = frozenset()
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
            roles_by_request[entry["request_id"]] = pending_roles
            if pending_user is not None:
                user_by_request[entry["request_id"]] = pending_user
            pending_user = None
            pending_roles = frozenset()
            continue
        if event == "llm_usage":
            if entry.get("cache_hit"):
                cache_hits += 1
            elif "completion_latency_ms" in entry:
                latency_ms_by_request[entry["request_id"]] = int(
                    entry["completion_latency_ms"]
                )
            continue
        updated = _CACHE_UPDATED.match(message)
        if updated is not None:
            if pending_prompt is not None:
                prompt_by_request[updated.group("request_id")] = pending_prompt
            pending_prompt = None
            continue
        matched = _AUTHZ_MATCHED.match(message)
        if matched is not None:
            pending_user = matched.group("user")
            pending_roles = frozenset(
                role.strip() for role in matched.group("roles").split(",")
            ) - {""}
            for role in matched.group("roles").split(","):
                roles.append((matched.group("user"), role.strip()))
    return RoutedTraffic(
        roles,
        user_by_request,
        prompt_by_request,
        call_by_request,
        latency_ms_by_request,
        cache_hits,
        roles_by_request,
    )


def _matched_decisions(
    policy: RouterPolicy, selected: str, roles: frozenset[str]
) -> set[str]:
    """Every decision the router matched on a call that selected ``selected``.

    The router counts each decision whose rules hold, then serves the matched
    one with the highest priority. Within the selected decision's routing
    section, a decision matched when its conditions are among the selected
    decision's and the caller's roles, and did not when it names a role the
    caller lacks or outranks the selected decision. The log settles nothing
    else, so anything else is refused.
    """
    selected_conditions = policy.conditions_by_decision[selected]
    held_roles = roles | {
        value for kind, value in selected_conditions if kind == "authz"
    }
    known = selected_conditions | {("authz", role) for role in held_roles}
    priority = policy.priority_by_decision[selected]
    matched: set[str] = set()
    for name in policy.decisions:
        if policy.section_by_decision[name] != policy.section_by_decision[selected]:
            continue
        conditions = policy.conditions_by_decision[name]
        outranks = policy.priority_by_decision[name] > priority
        if name == selected or (conditions <= known and not outranks):
            matched.add(name)
        elif any(
            kind == "authz" and value not in held_roles for kind, value in conditions
        ):
            continue
        elif outranks and not conditions <= known:
            continue
        else:
            raise AssertionError(
                f"the router log cannot settle whether {name} matched a call "
                f"that selected {selected}"
            )
    return matched


def _expected_match_deltas(
    policy: RouterPolicy, traffic: RoutedTraffic
) -> dict[str, int]:
    """The decision-match counter deltas the logged calls account for."""
    counts: Counter = Counter()
    for request_id, (decision, _) in traffic.call_by_request.items():
        counts.update(
            _matched_decisions(policy, decision, traffic.roles_by_request[request_id])
        )
    return {decision: counts[decision] for decision in policy.decisions}


@dataclass(frozen=True)
class EnvoyCall:
    cluster: str
    duration_ms: int
    flags: str
    status: int


def _envoy_calls(entries: list[dict]) -> dict[str, EnvoyCall]:
    """request_id -> what Envoy logged for every completion it proxied."""
    calls: dict[str, EnvoyCall] = {}
    for entry in entries:
        if not str(entry.get("path", "")).endswith("/chat/completions"):
            continue
        request_id = entry.get("request_id")
        if not request_id or not entry.get("cluster"):
            continue
        calls[request_id] = EnvoyCall(
            cluster=entry["cluster"],
            duration_ms=int(entry["duration_ms"]),
            flags=str(entry.get("flags", "")),
            status=int(entry.get("status", 0)),
        )
    return calls


# Envoy writes a request's access line when its stream ends, which for the
# dispatch's last LM call is within the same second the dispatch returns;
# the read retries for this long before it reports a call as unlogged.
ENVOY_ACCESS_LOG_BUDGET_S = 15.0


def _envoy_calls_for(
    request_ids: list[str], start: datetime, end: datetime
) -> dict[str, EnvoyCall]:
    deadline = time.monotonic() + ENVOY_ACCESS_LOG_BUDGET_S
    while True:
        calls = _envoy_calls(_json_log_entries(_ENVOY_DEPLOY, start, end, _envoy_stamp))
        if all(r in calls for r in request_ids) or time.monotonic() >= deadline:
            return calls
        time.sleep(1)


def _served_models(
    phoenix_client, project: str, start: datetime, end: datetime, expected: set[str]
) -> tuple[set[str], float]:
    """The served-model values stamped on this tenant's spans in the window.

    Polls until ``expected`` is observed or the read budget lapses, and
    returns what it saw with the seconds it took; a Phoenix read that keeps
    failing raises with the last error rather than reading as no spans.
    """
    started = time.monotonic()
    deadline = started + SERVED_MODEL_READ_BUDGET_S
    found: set[str] = set()
    last_error: Exception | None = None
    while True:
        try:
            frame = phoenix_client.spans.get_spans_dataframe(
                project_identifier=project,
                start_time=start,
                end_time=end + timedelta(seconds=SERVED_MODEL_READ_BUDGET_S),
                timeout=30,
            )
            last_error = None
            # Phoenix flattens only the OpenInference llm.* keys into columns;
            # this one arrives inside the nested ``attributes.llm`` column.
            spans = [] if frame is None else [row for _, row in frame.iterrows()]
            found = {
                str(attributes[LLM_SERVED_MODEL_ATTRIBUTE])
                for attributes in map(read_span_attributes, spans)
                if LLM_SERVED_MODEL_ATTRIBUTE in attributes
            }
        except Exception as exc:  # noqa: BLE001 - re-raised at the deadline
            last_error = exc
        if found == expected or time.monotonic() >= deadline:
            break
        time.sleep(2)
    if last_error is not None:
        raise AssertionError(
            f"Phoenix read of {project!r} kept failing for "
            f"{SERVED_MODEL_READ_BUDGET_S}s: {type(last_error).__name__}: {last_error}"
        )
    return found, time.monotonic() - started


def _set_tier(tenant_id: str, tier: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.put(
            f"{RUNTIME}/admin/tenants/{tenant_id}/tier", json={"tier": tier}
        )
    assert response.status_code == 200, response.text
    return response.json()


def _get_tier(tenant_id: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(f"{RUNTIME}/admin/tenants/{tenant_id}/tier")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_bootstrap_declares_the_seeded_tenant_tier():
    """The session bootstrap leaves the seeded tenant on SEEDED_TENANT_TIER;
    re-running the bootstrap's tier step restores it after an operator moved
    it, and a tenant the bootstrap never touched reads the default."""
    canonical = canonical_tenant_id(TENANT_ID)
    declared = {"tenant_id": canonical, "tier": SEEDED_TENANT_TIER}
    assert _get_tier(TENANT_ID) == declared

    (moved,) = sorted(ROUTER_TIERS - {SEEDED_TENANT_TIER, DEFAULT_ROUTER_TIER})
    assert _set_tier(TENANT_ID, moved) == {"tenant_id": canonical, "tier": moved}
    assert _get_tier(TENANT_ID) == {"tenant_id": canonical, "tier": moved}
    assert bootstrap_seeded_tenant_tier() == declared
    assert _get_tier(TENANT_ID) == declared
    assert bootstrap_seeded_tenant_tier() == declared
    assert _get_tier(TENANT_ID) == declared

    fresh = unique_id("tier")
    register_tenant_and_wait(fresh)
    assert _get_tier(fresh) == {
        "tenant_id": canonical_tenant_id(fresh),
        "tier": DEFAULT_ROUTER_TIER,
    }


def _dispatch_one_query(tenant_id: str, query: str) -> float:
    """One real query through the production dispatch route; seconds it took.

    Body shape is the route's ``AgentTask``: agent_name + query, tenant inside
    ``context``.
    """
    started = time.monotonic()
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
    return time.monotonic() - started


@dataclass(frozen=True)
class Leg:
    """One dispatch, as the router, Envoy and the runtime's spans describe it."""

    deltas: dict[str, int]
    expected_deltas: dict[str, int]
    cache_hit_delta: int
    logged_cache_hits: int
    mine_by_role: Counter
    decisions: list[str]
    selected_models: list[str]
    clusters: list[str]
    timed_out: int
    served_models: set[str]
    prompt_fields: set[frozenset[str]]
    timings: dict

    def __str__(self) -> str:
        return (
            f"counter deltas {self.deltas} vs {self.expected_deltas} derived from "
            f"the log; this tenant's routed calls {self.mine_by_role} -> "
            f"decisions {self.decisions} on models {self.selected_models} via "
            f"clusters {self.clusters} ({self.timed_out} cut by the client), "
            f"served models {sorted(self.served_models)}, "
            f"signature inputs {sorted(sorted(f) for f in self.prompt_fields)}; "
            f"router response-cache hits {self.cache_hit_delta} counted / "
            f"{self.logged_cache_hits} logged; timings {self.timings}"
        )


def _run_leg(
    metrics_url: str,
    phoenix_client,
    tenant_id: str,
    canonical: str,
    query: str,
    policy: RouterPolicy,
    routing: DeployedRouting,
    expected_served: set[str] | None,
) -> Leg:
    start = datetime.now(timezone.utc)
    before = _samples(metrics_url)
    dispatch_s = _dispatch_one_query(tenant_id, query)
    after = _samples(metrics_url)
    end = datetime.now(timezone.utc)

    traffic = _routed_traffic(
        _json_log_entries(_ROUTER_DEPLOY, start, end, _router_stamp)
    )
    mine = [r for r, user in traffic.user_by_request.items() if user == canonical]
    envoy = _envoy_calls_for(mine, start, end)
    calls = [traffic.call_by_request[r] for r in mine]
    selected_models = [model for _, model in calls]
    if expected_served is None:
        expected_served = {routing.served_model_by_catalog[m] for m in selected_models}
    project = TelemetryConfig().get_project_name(canonical)
    served, served_read_s = _served_models(
        phoenix_client, project, start, end, expected_served
    )

    return Leg(
        deltas={
            decision: _counter(
                after,
                "llm_decision_match_total",
                decision_name=policy.metric_label_by_decision[decision],
            )
            - _counter(
                before,
                "llm_decision_match_total",
                decision_name=policy.metric_label_by_decision[decision],
            )
            for decision in policy.decisions
        },
        expected_deltas=_expected_match_deltas(policy, traffic),
        cache_hit_delta=sum(
            _counter(
                after,
                "llm_cache_plugin_hits_total",
                decision_name=policy.metric_label_by_decision[decision],
                plugin_type="response_cache",
            )
            - _counter(
                before,
                "llm_cache_plugin_hits_total",
                decision_name=policy.metric_label_by_decision[decision],
                plugin_type="response_cache",
            )
            for decision in policy.decisions
        ),
        logged_cache_hits=traffic.cache_hits,
        mine_by_role=Counter(role for user, role in traffic.roles if user == canonical),
        decisions=[name for name, _ in calls],
        selected_models=selected_models,
        clusters=[envoy[r].cluster for r in mine if r in envoy],
        timed_out=sum(1 for r in mine if r in envoy and "DC" in envoy[r].flags),
        served_models=served,
        prompt_fields={
            frozenset(_PROMPT_FIELD.findall(traffic.prompt_by_request[r]))
            for r in mine
            if r in traffic.prompt_by_request
        },
        timings={
            "dispatch_s": round(dispatch_s, 2),
            "backend_completion_ms": [
                traffic.latency_ms_by_request.get(r) for r in mine
            ],
            "envoy_duration_ms": [
                envoy[r].duration_ms if r in envoy else None for r in mine
            ],
            "envoy_flags": [envoy[r].flags if r in envoy else None for r in mine],
            "served_model_read_s": round(served_read_s, 2),
        },
    )


def _shown(value):
    """``value`` with every set as a sorted list, so a verdict reads the same
    whatever the interpreter's hash seed."""
    if isinstance(value, (set, frozenset)):
        return sorted((_shown(item) for item in value), key=repr)
    return value


def _leg_verdict(
    leg: Leg,
    *,
    tier: str,
    repeat: bool,
    policy: RouterPolicy,
    routing: DeployedRouting,
    routed: int,
    expected_served: set[str],
) -> list[str]:
    """Every way the leg departs from what the tier and the deployed routing
    require; empty when the leg is exactly as expected."""
    problems: list[str] = []

    def check(name: str, actual, expected) -> None:
        if actual != expected:
            problems.append(f"{name}: {_shown(actual)!r} != {_shown(expected)!r}")

    check("calls cut by the client's timeout", leg.timed_out, 0)
    check(
        "this tenant's authz matches",
        leg.mine_by_role,
        Counter() if repeat else Counter({policy.role_by_tier[tier]: routed}),
    )
    check(
        "decision tiers",
        [policy.tier_by_decision[name] for name in leg.decisions],
        [] if repeat else [tier] * routed,
    )
    check(
        "selected models",
        leg.selected_models,
        [policy.model_by_decision[name] for name in leg.decisions],
    )
    check(
        "clusters",
        leg.clusters,
        [routing.cluster_for(model) for model in leg.selected_models],
    )
    check("served models", leg.served_models, expected_served)
    check(
        "signature inputs",
        leg.prompt_fields,
        set() if repeat else set(DISPATCH_SIGNATURE_INPUTS),
    )
    check("counter deltas", leg.deltas, leg.expected_deltas)
    check("response-cache hits", leg.cache_hit_delta, leg.logged_cache_hits)
    return problems


def test_the_stored_tier_steers_the_deployed_router(
    router_metrics_url, phoenix_client_session
):
    """Each tier set through the admin route changes the decision the deployed
    router matches, the cluster Envoy dials and the model that answers, for
    the next query on the production dispatch path; a repeated query never
    reaches the router at all."""
    policy = _router_policy()
    routing = _deployed_routing()
    assert set(policy.tier_by_decision.values()) == set(ROUTER_TIERS)
    assert set(routing.cluster_by_model) < set(routing.served_model_by_catalog)
    assert len(set(routing.served_model_by_catalog.values())) == len(
        routing.served_model_by_catalog
    )

    tenant_id = unique_id("tier")
    register_tenant_and_wait(tenant_id)
    canonical = canonical_tenant_id(tenant_id)

    walk = sorted(ROUTER_TIERS)
    walk.append(walk[0])
    legs = [(tier, f"{QUERY} ({tenant_id} leg {i})") for i, tier in enumerate(walk)]
    legs.append(legs[-1])

    routed = len(DISPATCH_SIGNATURE_INPUTS)
    timings: list[dict] = []
    previous: Leg | None = None
    for index, (tier, query) in enumerate(legs):
        repeat = index == len(legs) - 1
        assert _set_tier(tenant_id, tier) == {
            "tenant_id": canonical,
            "tier": tier,
        }
        expected_served = previous.served_models if repeat and previous else None
        leg = _run_leg(
            router_metrics_url,
            phoenix_client_session,
            tenant_id,
            canonical,
            query,
            policy,
            routing,
            expected_served,
        )
        timings.append({"leg": index, "tier": tier, "repeat": repeat, **leg.timings})
        print(f"\nTIER LEG TIMING {json.dumps(timings[-1])}")
        verdict = _leg_verdict(
            leg,
            tier=tier,
            repeat=repeat,
            policy=policy,
            routing=routing,
            routed=routed,
            expected_served=(
                expected_served
                if expected_served is not None
                else {routing.served_model_by_catalog[m] for m in leg.selected_models}
            ),
        )
        assert verdict == [], f"leg {index} on tier {tier!r} (repeat={repeat}): {leg}"
        if not repeat:
            assert leg.served_models == {
                routing.served_model_by_catalog[policy.model_by_decision[name]]
                for name in leg.decisions
            }, f"leg {index}: {leg}"
        previous = leg
    print(f"\nTIER LEG TIMINGS {json.dumps(timings)}")


_SYNTHETIC_POLICY = RouterPolicy(
    role_by_tier={"pro": "pro_tier", "default": "base_tier"},
    tier_by_decision={
        "pro-technical": "pro",
        "pro-default": "pro",
        "classification-pro": "pro",
        "base-default": "default",
        "classification-base": "default",
    },
    model_by_decision={
        "pro-technical": "pro-reasoning",
        "pro-default": "pro-reasoning",
        "classification-pro": "basic-chat",
        "base-default": "basic-chat",
        "classification-base": "basic-chat",
    },
    metric_label_by_decision={
        "pro-technical": "pro-technical",
        "pro-default": "pro-default",
        "classification-pro": "classification::classification-pro",
        "base-default": "base-default",
        "classification-base": "classification::classification-base",
    },
    decisions=(
        "pro-technical",
        "pro-default",
        "classification-pro",
        "base-default",
        "classification-base",
    ),
    conditions_by_decision={
        "pro-technical": frozenset({("authz", "pro_tier"), ("domain", "technical")}),
        "pro-default": frozenset({("authz", "pro_tier")}),
        "classification-pro": frozenset({("authz", "pro_tier")}),
        "base-default": frozenset({("authz", "base_tier")}),
        "classification-base": frozenset({("authz", "base_tier")}),
    },
    priority_by_decision={
        "pro-technical": 250,
        "pro-default": 200,
        "classification-pro": 200,
        "base-default": 50,
        "classification-base": 50,
    },
    section_by_decision={
        "pro-technical": "",
        "pro-default": "",
        "classification-pro": "classification",
        "base-default": "",
        "classification-base": "classification",
    },
)
_SYNTHETIC_ROUTING = DeployedRouting(
    cluster_by_model={"pro-reasoning": "llm_teacher"},
    default_cluster="llm_upstream",
    served_model_by_catalog={
        "basic-chat": "student-model",
        "pro-reasoning": "teacher-model",
    },
)


def _synthetic_pro_leg() -> Leg:
    """A pro leg exactly as the deployed routing must produce it."""
    deltas = {
        "pro-default": 1,
        "classification-pro": 1,
        "base-default": 0,
        "classification-base": 0,
    }
    return Leg(
        deltas=deltas,
        expected_deltas=dict(deltas),
        cache_hit_delta=0,
        logged_cache_hits=0,
        mine_by_role=Counter({"pro_tier": 2}),
        decisions=["classification-pro", "pro-default"],
        selected_models=["basic-chat", "pro-reasoning"],
        clusters=["llm_upstream", "llm_teacher"],
        timed_out=0,
        served_models={"student-model", "teacher-model"},
        prompt_fields=set(DISPATCH_SIGNATURE_INPUTS),
        timings={},
    )


def _synthetic_verdict(leg: Leg) -> list[str]:
    return _leg_verdict(
        leg,
        tier="pro",
        repeat=False,
        policy=_SYNTHETIC_POLICY,
        routing=_SYNTHETIC_ROUTING,
        routed=2,
        expected_served={"student-model", "teacher-model"},
    )


def test_the_verdict_accepts_a_pro_leg_served_as_the_chart_binds():
    assert _synthetic_verdict(_synthetic_pro_leg()) == []


def test_the_verdict_rejects_a_pro_call_dialled_to_the_student_cluster():
    leg = replace(_synthetic_pro_leg(), clusters=["llm_upstream", "llm_upstream"])
    assert _synthetic_verdict(leg) == [
        "clusters: ['llm_upstream', 'llm_upstream'] != ['llm_upstream', 'llm_teacher']"
    ]


def test_the_verdict_rejects_a_pro_call_answered_by_the_student_model():
    leg = replace(_synthetic_pro_leg(), served_models={"student-model"})
    assert _synthetic_verdict(leg) == [
        "served models: ['student-model'] != ['student-model', 'teacher-model']"
    ]


def test_the_verdict_rejects_a_call_the_client_cut_before_it_completed():
    leg = replace(_synthetic_pro_leg(), timed_out=1)
    assert _synthetic_verdict(leg) == ["calls cut by the client's timeout: 1 != 0"]


def test_a_selected_decision_also_counts_the_catch_all_beneath_it():
    assert _matched_decisions(
        _SYNTHETIC_POLICY, "pro-technical", frozenset({"pro_tier"})
    ) == {"pro-technical", "pro-default"}


def test_a_selected_catch_all_counts_only_itself():
    assert _matched_decisions(
        _SYNTHETIC_POLICY, "pro-default", frozenset({"pro_tier"})
    ) == {"pro-default"}


def test_a_recipe_selection_counts_only_its_own_section():
    assert _matched_decisions(
        _SYNTHETIC_POLICY, "classification-pro", frozenset({"pro_tier"})
    ) == {"classification-pro"}


def test_a_match_the_log_cannot_settle_is_refused():
    policy = replace(
        _SYNTHETIC_POLICY,
        decisions=(*_SYNTHETIC_POLICY.decisions, "pro-keyword"),
        conditions_by_decision={
            **_SYNTHETIC_POLICY.conditions_by_decision,
            "pro-keyword": frozenset({("authz", "pro_tier"), ("keyword", "technical")}),
        },
        priority_by_decision={
            **_SYNTHETIC_POLICY.priority_by_decision,
            "pro-keyword": 220,
        },
        section_by_decision={
            **_SYNTHETIC_POLICY.section_by_decision,
            "pro-keyword": "",
        },
    )
    with pytest.raises(AssertionError) as raised:
        _matched_decisions(policy, "pro-technical", frozenset({"pro_tier"}))
    assert str(raised.value) == (
        "the router log cannot settle whether pro-keyword matched a call "
        "that selected pro-technical"
    )


def test_the_counter_deltas_count_every_decision_each_logged_call_matched():
    authz = '[Authz Signal] Matched 1 roles for user "t:t": [pro_tier]'
    entries = [
        {"msg": authz},
        {
            "event": "routing_decision",
            "request_id": "r1",
            "decision": "pro-technical",
            "selected_model": "pro-reasoning",
        },
        {"msg": authz},
        {
            "event": "routing_decision",
            "request_id": "r2",
            "decision": "classification-pro",
            "selected_model": "basic-chat",
        },
    ]
    assert _expected_match_deltas(_SYNTHETIC_POLICY, _routed_traffic(entries)) == {
        "pro-technical": 1,
        "pro-default": 1,
        "classification-pro": 1,
        "base-default": 0,
        "classification-base": 0,
    }


def test_the_verdict_rejects_a_decision_the_counters_did_not_record():
    leg = replace(
        _synthetic_pro_leg(), deltas={**_synthetic_pro_leg().deltas, "pro-default": 0}
    )
    assert _synthetic_verdict(leg) == [
        "counter deltas: {'pro-default': 0, 'classification-pro': 1, "
        "'base-default': 0, 'classification-base': 0} != {'pro-default': 1, "
        "'classification-pro': 1, 'base-default': 0, 'classification-base': 0}"
    ]


def test_the_tier_refresh_does_not_stall_the_event_loop(request):
    """The refresh the admin write forces is paid off the serving loop.

    Writing the tier drops it from every reader in the replica, so the next
    dispatch resolves it again out of the config store — a Vespa read with
    retries and backoff. That read is the whole of this test's window: a
    second client polls liveness for the dispatch's duration and every poll
    must be answered.
    """
    request.addfinalizer(bootstrap_seeded_tenant_tier)
    canonical = canonical_tenant_id(TENANT_ID)
    declared = {"tenant_id": canonical, "tier": SEEDED_TENANT_TIER}
    assert _set_tier(TENANT_ID, SEEDED_TENANT_TIER) == declared

    with LoopProbe() as probe:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{RUNTIME}/agents/{AGENT}/process",
                json={
                    "agent_name": AGENT,
                    "query": QUERY,
                    "context": {"tenant_id": TENANT_ID},
                },
            )
        served = probe.stop()

    assert response.status_code == 200, response.text[:600]
    body = response.json()
    assert body["status"] == "success", body
    assert body["agent"] == AGENT, body
    assert body["message"] == f"Generated summary for '{QUERY}'", body["message"]
    assert_loop_served(served)
    # The refresh resolved the tier the admin write stored, so the offload
    # cannot be achieved by skipping the read.
    assert _get_tier(TENANT_ID) == declared
