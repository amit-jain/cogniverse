"""A tenant's stored tier steers the deployed router on the production path.

Set the tier through the deployed runtime's admin route, send one real query
through the dispatch route, and reconcile what the deployed stack recorded:
the router's per-decision counters against its own routed-call log, Envoy's
access log for which cluster served each call, and the runtime's spans for
which model answered. The runtime resolves the tier itself from its config
store, so this exercises the whole chain the cluster runs: admin write -> the
runtime's per-tenant reader (and its in-process invalidation) -> the router's
decision -> the backend Envoy dials -> the model the backend reports.

One summarizer dispatch makes one routed LM call per call site it runs - the
search-query rewrite and the summary - so the routed count and the router
entrypoints are derived from those call sites rather than restated. The rewrite
is a bounded call and enters on the classification entrypoint, whose decisions
serve basic-chat for every tier; the summary enters on the auto alias, where a
pro tenant's decision serves pro-reasoning. Every counter movement in the
window is attributable per decision, this tenant's and anyone else's, so a
concurrent tenant cannot move a pin.

The deployed chart binds each catalog model to a backend - basic-chat to the
student, pro-reasoning to the teacher - on their own Envoy clusters. Beyond
the decision name, a tier change is observed as the cluster Envoy logs for the
call and the served model the runtime stamps on its span from the completion's
own ``model`` field. Both expectations are read from the ConfigMaps the
cluster runs, never restated here.

Each leg carries its own query text. The last leg repeats the previous leg's
query verbatim, and the repeat never reaches the upstream model. The runtime's
LM response cache is per worker process, so which cache answers depends on the
worker the repeat lands on, and the leg names it: the worker that answered the
original replays it without routing anything; any other worker routes every
call site's call again on the previous leg's tier, and the router answers each
from its response cache - one counted and logged hit per call, the original
decisions counted again, no decision logged, no Envoy upstream and no backend
completion. Either way the replayed completion still names the model that
produced it.
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

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.semantic_router import routed_model_for
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
    SemanticRouterConfig,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.span_contract import (
    LLM_SERVED_MODEL_ATTRIBUTE,
    read_span_attributes,
)
from tests.e2e.cluster import KUBECTL_CONTEXT, RUNTIME, TENANT_ID
from tests.e2e.conftest import SEEDED_TENANT_TIER, bootstrap_seeded_tenant_tier
from tests.e2e.loop_probe import LoopProbe, assert_loop_served
from tests.e2e.tenants import register_tenant_and_wait, unique_id

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
TIER_REFRESH_AGENT = "search_agent"
# Spans reach Phoenix through the runtime's batch exporter; measured on this
# cluster the served-model attribute is readable 4-9 s after the dispatch
# returns, so the read polls up to this long before it reports what it saw.
SERVED_MODEL_READ_BUDGET_S = 90.0

# The call sites of the LM calls one summarizer dispatch routes: the search
# agent's query rewrite and the summary. Each sends the router model name its
# call site maps to, which the router logs as the decision's ``original_model``
# without the litellm provider prefix, so this is what identifies a routed
# call and how many there are.
DISPATCH_CALL_SITES = (TIER_REFRESH_AGENT, AGENT)
DISPATCH_ENTRYPOINTS = sorted(
    routed_model_for(SemanticRouterConfig(), call_site).split("/", 1)[1]
    for call_site in DISPATCH_CALL_SITES
)

# What answered a repeated query: the serving worker's own LM response cache,
# or - on any other worker - the router's response cache.
WORKER_CACHE = "the worker's response cache"
ROUTER_CACHE = "the router's response cache"

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
    entrypoint_by_request: dict[str, str]
    cache_written: frozenset[str]
    call_by_request: dict[str, tuple[str, str]]
    latency_ms_by_request: dict[str, int]
    cache_hits: int
    roles_by_request: dict[str, frozenset[str]]


def _routed_traffic(entries: list[dict]) -> RoutedTraffic:
    roles: list[tuple[str, str]] = []
    user_by_request: dict[str, str] = {}
    entrypoint_by_request: dict[str, str] = {}
    cache_written: set[str] = set()
    call_by_request: dict[str, tuple[str, str]] = {}
    roles_by_request: dict[str, frozenset[str]] = {}
    latency_ms_by_request: dict[str, int] = {}
    cache_hits = 0
    pending_user: str | None = None
    pending_roles: frozenset[str] = frozenset()
    for entry in entries:
        event = entry.get("event")
        message = entry.get("msg", "")
        if event == "routing_decision":
            call_by_request[entry["request_id"]] = (
                entry["decision"],
                entry["selected_model"],
            )
            entrypoint_by_request[entry["request_id"]] = entry["original_model"]
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
            cache_written.add(updated.group("request_id"))
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
        entrypoint_by_request,
        frozenset(cache_written),
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
    entrypoints: list[str]
    cache_writes: int
    backend_completions: int
    timings: dict

    def __str__(self) -> str:
        return (
            f"counter deltas {self.deltas} vs {self.expected_deltas} derived from "
            f"the log; this tenant's routed calls {self.mine_by_role} -> "
            f"decisions {self.decisions} on models {self.selected_models} via "
            f"clusters {self.clusters} ({self.timed_out} cut by the client), "
            f"served models {sorted(self.served_models)}, "
            f"entrypoints {self.entrypoints} with {self.cache_writes} response-"
            "cache writes; "
            f"{self.backend_completions} backend completions in the window; "
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
        entrypoints=sorted(traffic.entrypoint_by_request[r] for r in mine),
        cache_writes=sum(1 for r in mine if r in traffic.cache_written),
        backend_completions=len(traffic.latency_ms_by_request),
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


def _repeat_branch(leg: Leg, routed: int) -> str:
    """Which cache answered a repeated query, read off its routed calls.

    The runtime's LM response cache is per worker process: a repeat served by
    the worker that answered the original routes nothing, and one served by
    any other worker routes every call site's call again, each answered from
    the router's response cache.
    """
    calls = sum(leg.mine_by_role.values())
    if calls == 0:
        return WORKER_CACHE
    if calls == routed:
        return ROUTER_CACHE
    return f"neither cache: {calls} routed calls"


def _leg_verdict(
    leg: Leg,
    *,
    tier: str,
    repeat: bool,
    policy: RouterPolicy,
    routing: DeployedRouting,
    routed: int,
    expected_served: set[str],
    previous: Leg | None = None,
) -> list[str]:
    """Every way the leg departs from what the tier and the deployed routing
    require; empty when the leg is exactly as expected.

    A repeat (``repeat=True``) needs ``previous``, the leg it repeats, and is
    held to whichever branch ``_repeat_branch`` reads off it."""
    problems: list[str] = []

    def check(name: str, actual, expected) -> None:
        if actual != expected:
            problems.append(f"{name}: {_shown(actual)!r} != {_shown(expected)!r}")

    check("calls cut by the client's timeout", leg.timed_out, 0)
    if repeat:
        branch = _repeat_branch(leg, routed)
        if branch not in (WORKER_CACHE, ROUTER_CACHE):
            problems.append(f"repeat answered by {branch}, not 0 or {routed}")
        hits = 0 if branch == WORKER_CACHE else routed
        check(
            "this tenant's authz matches",
            leg.mine_by_role,
            Counter({policy.role_by_tier[tier]: hits}) if hits else Counter(),
        )
        # A response-cache hit is answered before a routing decision is
        # logged or an upstream is dialled.
        check("decision tiers", [policy.tier_by_decision[n] for n in leg.decisions], [])
        check("clusters", leg.clusters, [])
        check("entrypoints", leg.entrypoints, [])
        check("response-cache writes", leg.cache_writes, 0)
        check("backend completions", leg.backend_completions, 0)
        check("response-cache hits counted", leg.cache_hit_delta, hits)
        check("response-cache hits logged", leg.logged_cache_hits, hits)
        # Each hit still counts the decisions the original call matched.
        check(
            "counter deltas",
            leg.deltas,
            {
                decision: leg.expected_deltas[decision]
                + (previous.deltas[decision] if hits else 0)
                for decision in policy.decisions
            },
        )
        check("served models", leg.served_models, expected_served)
        return problems

    check(
        "this tenant's authz matches",
        leg.mine_by_role,
        Counter({policy.role_by_tier[tier]: routed}),
    )
    check(
        "decision tiers",
        [policy.tier_by_decision[name] for name in leg.decisions],
        [tier] * routed,
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
    check("entrypoints", leg.entrypoints, DISPATCH_ENTRYPOINTS)
    # Every routed call is a miss the router writes to its response cache.
    check("response-cache writes", leg.cache_writes, len(leg.decisions))
    check("counter deltas", leg.deltas, leg.expected_deltas)
    check("response-cache hits", leg.cache_hit_delta, leg.logged_cache_hits)
    return problems


def test_the_stored_tier_steers_the_deployed_router(
    router_metrics_url, phoenix_client_session
):
    """Each tier set through the admin route changes the decision the deployed
    router matches, the cluster Envoy dials and the model that answers, for
    the next query on the production dispatch path; a repeated query never
    reaches the upstream model: the worker that answered it replays it without
    routing, and any other worker's routed calls are each answered from the
    router's response cache on the previous leg's tier."""
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

    routed = len(DISPATCH_CALL_SITES)
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
        branch = _repeat_branch(leg, routed) if repeat else None
        timings.append(
            {
                "leg": index,
                "tier": tier,
                "repeat": repeat,
                "repeat_branch": branch,
                **leg.timings,
            }
        )
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
            previous=previous,
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
        entrypoints=list(DISPATCH_ENTRYPOINTS),
        cache_writes=2,
        backend_completions=2,
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
            "original_model": "auto",
        },
        {"msg": authz},
        {
            "event": "routing_decision",
            "request_id": "r2",
            "decision": "classification-pro",
            "selected_model": "basic-chat",
            "original_model": "cogniverse-classification",
        },
    ]
    assert _expected_match_deltas(_SYNTHETIC_POLICY, _routed_traffic(entries)) == {
        "pro-technical": 1,
        "pro-default": 1,
        "classification-pro": 1,
        "base-default": 0,
        "classification-base": 0,
    }


# One routed call as the pinned router logs it, verbatim: the caller's authz
# match, the routing decision, its usage line and the response-cache write.
_PINNED_ROUTER_CALL_LOG = [
    r'{"level":"info","ts":"2026-09-27T11:34:06.970","caller":"authz_classifier.go:174","msg":"[Authz Signal] Matched 1 roles for user \"rc_hit_de8d6d5bx\": [base_tier]"}',
    r'{"level":"info","ts":"2026-09-27T11:34:06.971","caller":"logging.go:264","msg":"routing_decision","reasoning_enabled":false,"component":"extproc","original_model":"cogniverse-classification","reason_code":"entrypoint_routing","selected_model":"basic-chat","decision":"classification-base","routing_latency_ms":1,"event":"routing_decision","request_id":"b9801ef9-201f-497f-8f45-827a29f69f7b","reasoning_effort":""}',
    r'{"level":"info","ts":"2026-09-27T11:34:06.974","caller":"logging.go:160","msg":"llm_usage","pricing":"not_configured","request_id":"b9801ef9-201f-497f-8f45-827a29f69f7b","model":"basic-chat","cache_write_tokens":0,"currency":"unknown","cached_prompt_tokens":0,"completion_latency_ms":3,"total_tokens":20,"event":"llm_usage","prompt_tokens":8,"completion_tokens":12,"cost":0}',
    r'{"level":"info","ts":"2026-09-27T11:34:06.974","caller":"processor_res_cache.go:73","msg":"Cache updated for request ID: b9801ef9-201f-497f-8f45-827a29f69f7b"}',
]


def test_the_traffic_reads_each_call_off_the_pinned_router_log():
    request_id = "b9801ef9-201f-497f-8f45-827a29f69f7b"
    traffic = _routed_traffic([json.loads(line) for line in _PINNED_ROUTER_CALL_LOG])
    assert traffic == RoutedTraffic(
        roles=[("rc_hit_de8d6d5bx", "base_tier")],
        user_by_request={request_id: "rc_hit_de8d6d5bx"},
        entrypoint_by_request={request_id: "cogniverse-classification"},
        cache_written=frozenset({request_id}),
        call_by_request={request_id: ("classification-base", "basic-chat")},
        latency_ms_by_request={request_id: 3},
        cache_hits=0,
        roles_by_request={request_id: frozenset({"base_tier"})},
    )


def test_the_dispatch_enters_on_the_classification_and_auto_entrypoints():
    assert DISPATCH_ENTRYPOINTS == ["auto", "cogniverse-classification"]


def test_the_verdict_rejects_a_call_sent_to_the_wrong_entrypoint():
    leg = replace(_synthetic_pro_leg(), entrypoints=["auto", "auto"])
    assert _synthetic_verdict(leg) == [
        "entrypoints: ['auto', 'auto'] != ['auto', 'cogniverse-classification']"
    ]


def test_the_verdict_rejects_a_routed_call_the_router_did_not_cache():
    leg = replace(_synthetic_pro_leg(), cache_writes=1)
    assert _synthetic_verdict(leg) == ["response-cache writes: 1 != 2"]


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

    The dispatch is a search: it resolves the tier to bind the routed LM for
    its query rewrite, and a rewrite the LM does not complete degrades to the
    query as written, so the request's outcome is the search's, whatever the
    LM writes.
    """
    request.addfinalizer(bootstrap_seeded_tenant_tier)
    canonical = canonical_tenant_id(TENANT_ID)
    declared = {"tenant_id": canonical, "tier": SEEDED_TENANT_TIER}
    assert _set_tier(TENANT_ID, SEEDED_TENANT_TIER) == declared

    with LoopProbe() as probe:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{RUNTIME}/agents/{TIER_REFRESH_AGENT}/process",
                json={
                    "agent_name": TIER_REFRESH_AGENT,
                    "query": QUERY,
                    "context": {"tenant_id": TENANT_ID},
                },
            )
        served = probe.stop()

    assert response.status_code == 200, response.text[:600]
    body = response.json()
    assert body["status"] == "success", body
    assert body["agent"] == TIER_REFRESH_AGENT, body
    assert body["results_count"] == len(body["results"]), body
    assert body["message"] == (
        f"Found {body['results_count']} results for '{QUERY}'"
        if body["results_count"]
        else f"No results found for '{QUERY}'"
    ), body["message"]
    assert_loop_served(served)
    # The refresh resolved the tier the admin write stored, so the offload
    # cannot be achieved by skipping the read.
    assert _get_tier(TENANT_ID) == declared


# The deployed run's last two legs, as it reported them: leg 3 on the default
# tier, and its repeat, which landed on another worker. The repeat's line is
# the failure message that run printed, verbatim; leg 3 passed, so it is the
# leg its timing line and the verdict then in force pin.
_LIVE_LEG_3_TIMING = (
    'TIER LEG TIMING {"leg": 3, "tier": "default", "repeat": false, '
    '"dispatch_s": 3.81, "backend_completion_ms": [661, 1572], '
    '"envoy_duration_ms": [663, 1574], "envoy_flags": ["-", "-"], '
    '"served_model_read_s": 0.02}'
)
_LIVE_LEG_4_REPORTED = (
    "counter deltas {'pro-technical-keyword': 0, 'pro-technical-domain': 0, "
    "'pro-default': 0, 'free-default': 0, 'base-default': 1, "
    "'classification-pro': 0, 'classification-free': 0, "
    "'classification-base': 1, 'vision-pro': 0, 'vision-free': 0, "
    "'vision-base': 0, 'short-reasoning-pro': 0, 'short-reasoning-free': 0, "
    "'short-reasoning-base': 0} vs {'pro-technical-keyword': 0, "
    "'pro-technical-domain': 0, 'pro-default': 0, 'free-default': 0, "
    "'base-default': 0, 'classification-pro': 0, 'classification-free': 0, "
    "'classification-base': 0, 'vision-pro': 0, 'vision-free': 0, "
    "'vision-base': 0, 'short-reasoning-pro': 0, 'short-reasoning-free': 0, "
    "'short-reasoning-base': 0} derived from the log; this tenant's routed "
    "calls Counter({'base_tier': 2}) -> decisions [] on models [] via "
    "clusters [] (0 cut by the client), served models "
    "['google/gemma-4-e4b-it'], entrypoints [] with 0 response-cache writes; "
    "router response-cache hits 2 counted / 2 logged; timings "
    "{'dispatch_s': 1.13, 'backend_completion_ms': [], "
    "'envoy_duration_ms': [], 'envoy_flags': [], 'served_model_read_s': 0.02}"
)
_LIVE_SERVED = {"google/gemma-4-e4b-it"}


def _live_deltas(**moved: int) -> dict[str, int]:
    return {
        decision: moved.get(decision.replace("-", "_"), 0)
        for decision in _router_policy().decisions
    }


def _live_leg_3() -> Leg:
    timings = json.loads(_LIVE_LEG_3_TIMING.split(" ", 3)[3])
    deltas = _live_deltas(classification_base=1, base_default=1)
    return Leg(
        deltas=deltas,
        expected_deltas=dict(deltas),
        cache_hit_delta=0,
        logged_cache_hits=0,
        mine_by_role=Counter({"base_tier": 2}),
        decisions=["classification-base", "base-default"],
        selected_models=["basic-chat", "basic-chat"],
        clusters=["llm_upstream", "llm_upstream"],
        timed_out=0,
        served_models=set(_LIVE_SERVED),
        entrypoints=list(DISPATCH_ENTRYPOINTS),
        cache_writes=2,
        backend_completions=len(timings["backend_completion_ms"]),
        timings={k: timings[k] for k in timings if k not in ("leg", "tier", "repeat")},
    )


def _live_leg_4() -> Leg:
    return Leg(
        deltas=_live_deltas(classification_base=1, base_default=1),
        expected_deltas=_live_deltas(),
        cache_hit_delta=2,
        logged_cache_hits=2,
        mine_by_role=Counter({"base_tier": 2}),
        decisions=[],
        selected_models=[],
        clusters=[],
        timed_out=0,
        served_models=set(_LIVE_SERVED),
        entrypoints=[],
        cache_writes=0,
        backend_completions=0,
        timings={
            "dispatch_s": 1.13,
            "backend_completion_ms": [],
            "envoy_duration_ms": [],
            "envoy_flags": [],
            "served_model_read_s": 0.02,
        },
    )


def _live_repeat_verdict(leg: Leg) -> list[str]:
    return _leg_verdict(
        leg,
        tier=DEFAULT_ROUTER_TIER,
        repeat=True,
        policy=_router_policy(),
        routing=_SYNTHETIC_ROUTING,
        routed=len(DISPATCH_CALL_SITES),
        expected_served=_live_leg_3().served_models,
        previous=_live_leg_3(),
    )


def test_the_live_legs_are_the_ones_the_run_reported():
    assert str(_live_leg_4()).replace("0 backend completions in the window; ", "") == (
        _LIVE_LEG_4_REPORTED
    )
    assert _live_leg_3().timings == {
        "dispatch_s": 3.81,
        "backend_completion_ms": [661, 1572],
        "envoy_duration_ms": [663, 1574],
        "envoy_flags": ["-", "-"],
        "served_model_read_s": 0.02,
    }
    assert _live_leg_3().backend_completions == 2


def test_the_live_repeat_was_answered_by_the_router_cache_and_is_accepted():
    leg = _live_leg_4()
    assert _repeat_branch(leg, len(DISPATCH_CALL_SITES)) == ROUTER_CACHE
    assert _live_repeat_verdict(leg) == []


def test_a_repeat_the_worker_cache_answered_is_accepted():
    leg = replace(
        _live_leg_4(),
        deltas=_live_deltas(),
        cache_hit_delta=0,
        logged_cache_hits=0,
        mine_by_role=Counter(),
    )
    assert _repeat_branch(leg, len(DISPATCH_CALL_SITES)) == WORKER_CACHE
    assert _live_repeat_verdict(leg) == []


def test_a_repeat_that_dialled_upstream_is_rejected():
    """The repeat's calls routed as misses: decided, dialled, completed and
    written to the cache, exactly as the original leg was."""
    leg = replace(
        _live_leg_4(),
        expected_deltas=_live_deltas(classification_base=1, base_default=1),
        cache_hit_delta=0,
        logged_cache_hits=0,
        decisions=["classification-base", "base-default"],
        selected_models=["basic-chat", "basic-chat"],
        clusters=["llm_upstream", "llm_upstream"],
        entrypoints=list(DISPATCH_ENTRYPOINTS),
        cache_writes=2,
        backend_completions=2,
    )
    assert _repeat_branch(leg, len(DISPATCH_CALL_SITES)) == ROUTER_CACHE
    assert _live_repeat_verdict(leg) == [
        "decision tiers: ['default', 'default'] != []",
        "clusters: ['llm_upstream', 'llm_upstream'] != []",
        "entrypoints: ['auto', 'cogniverse-classification'] != []",
        "response-cache writes: 2 != 0",
        "backend completions: 2 != 0",
        "response-cache hits counted: 0 != 2",
        "response-cache hits logged: 0 != 2",
        f"counter deltas: "
        f"{_live_deltas(classification_base=1, base_default=1)!r} != "
        f"{_live_deltas(classification_base=2, base_default=2)!r}",
    ]


def test_a_repeat_routed_on_another_tier_is_rejected():
    leg = replace(_live_leg_4(), mine_by_role=Counter({"free_tier": 2}))
    assert _live_repeat_verdict(leg) == [
        "this tenant's authz matches: Counter({'free_tier': 2}) != "
        "Counter({'base_tier': 2})"
    ]


def test_a_repeat_routing_only_one_call_is_neither_branch():
    leg = replace(
        _live_leg_4(),
        mine_by_role=Counter({"base_tier": 1}),
        cache_hit_delta=1,
        logged_cache_hits=1,
        deltas=_live_deltas(base_default=1),
    )
    assert _repeat_branch(leg, len(DISPATCH_CALL_SITES)) == (
        "neither cache: 1 routed calls"
    )
    assert _live_repeat_verdict(leg) == [
        "repeat answered by neither cache: 1 routed calls, not 0 or 2",
        "this tenant's authz matches: Counter({'base_tier': 1}) != "
        "Counter({'base_tier': 2})",
        "response-cache hits counted: 1 != 2",
        "response-cache hits logged: 1 != 2",
        f"counter deltas: {_live_deltas(base_default=1)!r} != "
        f"{_live_deltas(classification_base=1, base_default=1)!r}",
    ]
