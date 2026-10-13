"""The search agent's profile fan-out encodes each query as ``POST /search`` does.

Real Vespa holds every shipped profile's schema for one tenant. Each inference
service the shipped profiles name is a recording HTTP server answering in the
service's production response shape, so a search shows which service was asked
to encode the query and with which model. Every shipped profile is searched
through ``SearchAgent._search_ensemble`` and through the runtime search route,
once per ranking strategy that reads query embeddings; both must put the same
encode requests on the wire.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import threading
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_core.common.models import semantic_embedder
from cogniverse_core.query.encoders import SERVICE_TEXT_ENCODERS, QueryEncoderFactory
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.inference_specs import get_inference_service_spec
from cogniverse_vespa.query_inputs import query_input_encodings
from cogniverse_vespa.ranking_strategy_extractor import RankingStrategyExtractor
from tests.utils.vespa_test_helpers import make_config_manager, serve_search_route

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

SHIPPED = json.loads(Path("configs/config.json").read_text())["backend"]
SHIPPED_PROFILES: dict[str, dict] = SHIPPED["profiles"]
ACTIVE_PROFILE = SHIPPED["default_profiles"]["video"]["profile"]
DENSE_SERVICE = "denseon"
TENANT = f"fanout{uuid.uuid4().hex[:8]}:unit"
QUERY = "red kayak river"


def _is_dense(profile: dict) -> bool:
    return profile.get("embedding_type") == "dense" or profile.get("encoder") == (
        "denseon"
    )


def _encode_service(profile: dict) -> str:
    """The inference service that embeds the profile's own embedding field."""
    if _is_dense(profile):
        return DENSE_SERVICE
    return profile["inference_services"]["embedding"]


def _query_model(profile: dict) -> str:
    return profile.get("semantic_model") or profile["embedding_model"]


SERVICES = sorted(
    {_encode_service(profile) for profile in SHIPPED_PROFILES.values()}
    | set(SERVICE_TEXT_ENCODERS)
)


def _embedding_strategies(profile: dict) -> dict:
    """The profile's ranking strategies that read query embeddings."""
    strategies = RankingStrategyExtractor().extract_from_schema(
        Path("configs/schemas") / f"{profile['schema_name']}_schema.json"
    )
    return {name: info for name, info in strategies.items() if info.inputs}


def _expected_encodes(profile: dict, info) -> set[tuple[str, str | None]]:
    """The (service, model) encode requests one search with the strategy
    sends. A service that answers one vector per text takes no model in the
    request: the X-CLIP sidecar and the field text encoders."""
    encodings = query_input_encodings(
        profile, {"inputs": info.inputs, "input_fields": info.input_fields}
    )
    expected = set()
    for encoding in encodings.values():
        if encoding.service is not None:
            expected.add((encoding.service, None))
        elif profile.get("model_loader") == "xclip":
            expected.add((_encode_service(profile), None))
        else:
            expected.add((_encode_service(profile), _query_model(profile)))
    return expected


CASES = sorted(
    (name, strategy)
    for name, profile in SHIPPED_PROFILES.items()
    for strategy in _embedding_strategies(profile)
)


class _Sidecar(BaseHTTPRequestHandler):
    """One inference service: records each encode request and answers in the
    production shape of whichever route was called."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        dim = self.server.dim
        with self.server.lock:
            self.server.calls.append(
                (self.server.service, self.path, body.get("model"))
            )
        if self.path == "/pooling":
            out = {"data": [{"data": [[0.01] * dim] * 3}]}
        elif self.path == "/embed/text":
            out = {"vec": [0.01] * dim}
        elif self.path == "/v1/embeddings":
            out = {
                "data": [
                    {"index": i, "embedding": [0.01] * dim}
                    for i in range(len(body["input"]))
                ]
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def _service_dim(service: str) -> int:
    """The width the service's vectors must have for the schemas it feeds."""
    dims = {
        profile["schema_config"]["embedding_dim"]
        for profile in SHIPPED_PROFILES.values()
        if not _is_dense(profile) and _encode_service(profile) == service
    }
    if len(dims) == 1:
        return dims.pop()
    return get_inference_service_spec(service).output_dimension


class _Sidecars:
    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []
        self.lock = threading.Lock()
        self.servers: dict[str, ThreadingHTTPServer] = {}
        for service in SERVICES:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Sidecar)
            server.service = service
            server.dim = _service_dim(service)
            server.calls = self.calls
            server.lock = self.lock
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers[service] = server

    @property
    def urls(self) -> dict[str, str]:
        return {
            service: f"http://127.0.0.1:{server.server_address[1]}"
            for service, server in self.servers.items()
        }

    def drain(self) -> list[tuple[str, str, str | None]]:
        with self.lock:
            calls = list(self.calls)
            self.calls.clear()
        return calls

    def close(self) -> None:
        for server in self.servers.values():
            server.shutdown()
            server.server_close()


@pytest.fixture(scope="module")
def deployment(shared_vespa):
    """Every shipped profile's schema deployed for one tenant, and every
    inference service the profiles name served by a recording sidecar."""
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()
    sidecars = _Sidecars()
    urls = sidecars.urls
    config_manager = make_config_manager(shared_vespa, inference_service_urls=urls)
    for name, profile in SHIPPED_PROFILES.items():
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(name, profile), tenant_id=TENANT
        )
    previous_dense = (
        semantic_embedder._CONFIGURED_REMOTE_URL,
        semantic_embedder._CONFIGURED_MODEL_NAME,
    )
    semantic_embedder.configure_semantic_embedder_defaults(
        remote_url=urls[DENSE_SERVICE], model_name=None
    )
    registry = BackendRegistry.get_instance()
    backend = registry.get_ingestion_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "port": shared_vespa["http_port"],
                "config_port": shared_vespa["config_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        tenant_id=TENANT,
    )
    backend.schema_registry.deploy_schemas(
        tenant_id=TENANT,
        base_schema_names=sorted(
            {profile["schema_name"] for profile in SHIPPED_PROFILES.values()}
        ),
    )
    BackendRegistry.clear_instances()
    with serve_search_route(config_manager, tenants=[TENANT]) as route_url:
        yield {
            "vespa": shared_vespa,
            "config_manager": config_manager,
            "sidecars": sidecars,
            "route_url": route_url,
        }
    semantic_embedder.configure_semantic_embedder_defaults(
        remote_url=previous_dense[0], model_name=previous_dense[1]
    )
    sidecars.close()
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(autouse=True)
def _telemetry_disabled():
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if telemetry_manager_module._telemetry_manager is installed:
        telemetry_manager_module._telemetry_manager = None


@pytest.fixture
def agent(deployment):
    vespa = deployment["vespa"]
    deployment["sidecars"].drain()
    return SearchAgent(
        deps=SearchAgentDeps(
            profile=ACTIVE_PROFILE,
            tenant_id=TENANT,
            backend_url="http://localhost",
            backend_port=vespa["http_port"],
            backend_config_port=vespa["config_port"],
            auto_create_memory_schema=False,
        ),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        config_manager=deployment["config_manager"],
        port=8062,
    )


def _fan_out(agent, profiles, modality="video", ranking=None):
    return asyncio.run(
        agent._search_ensemble(
            QUERY,
            tenant_id=TENANT,
            profiles=list(profiles),
            modality=modality,
            ranking=ranking,
        )
    )


def _route_search(deployment, profile: str, strategy: str) -> requests.Response:
    return requests.post(
        f"{deployment['route_url']}/search/",
        json={
            "query": QUERY,
            "profile": profile,
            "strategy": strategy,
            "tenant_id": TENANT,
            "top_k": 10,
        },
        timeout=60,
    )


def _encodes(calls) -> Counter:
    return Counter((service, model) for service, _path, model in calls)


def test_every_shipped_profile_has_an_embedding_strategy():
    assert sorted({profile for profile, _strategy in CASES}) == sorted(SHIPPED_PROFILES)


@pytest.mark.parametrize(("profile", "strategy"), CASES)
def test_a_fan_out_leg_encodes_as_the_search_route_does(
    deployment, agent, profile, strategy
):
    shipped = SHIPPED_PROFILES[profile]
    expected = _expected_encodes(shipped, _embedding_strategies(shipped)[strategy])
    sidecars = deployment["sidecars"]

    response = _route_search(deployment, profile, strategy)
    route_calls = sidecars.drain()
    assert response.status_code == 200, response.text
    assert response.json()["results"] == []
    assert _encodes(route_calls) == Counter(expected)

    outcome = _fan_out(agent, [profile], modality=shipped["type"], ranking=strategy)
    agent_calls = sidecars.drain()

    assert (outcome.results, outcome.searched, outcome.degraded) == (
        [],
        (profile,),
        (),
    )
    assert Counter(agent_calls) == Counter(route_calls)


def _default_strategy(profile: dict) -> str:
    """The strategy the backend runs when a search names none."""
    available = list(_embedding_strategies(profile))
    schema_strategies = list(
        RankingStrategyExtractor().extract_from_schema(
            Path("configs/schemas") / f"{profile['schema_name']}_schema.json"
        )
    )
    chosen = next(
        (s for s in ("default", "hybrid") if s in schema_strategies),
        schema_strategies[0],
    )
    assert chosen in available
    return chosen


def _default_encodes(profile: dict) -> set[tuple[str, str | None]]:
    return _expected_encodes(
        profile, _embedding_strategies(profile)[_default_strategy(profile)]
    )


def test_one_fan_out_over_every_shipped_profile_degrades_no_leg(deployment, agent):
    profiles = sorted(SHIPPED_PROFILES)

    outcome = _fan_out(agent, profiles)
    calls = deployment["sidecars"].drain()

    assert outcome.degraded == ()
    assert outcome.searched == tuple(profiles)
    assert set(_encodes(calls)) == set().union(
        *(_default_encodes(SHIPPED_PROFILES[p]) for p in profiles)
    )


def _encodes_per_fan_out(profiles) -> Counter:
    """One encode per distinct encoder among the non-active profiles, plus
    the active profile's own."""
    units = {
        (
            _is_dense(SHIPPED_PROFILES[p]),
            _query_model(SHIPPED_PROFILES[p]),
            SHIPPED_PROFILES[p].get("schema_config", {}).get("embedding_dim"),
            *sorted(_default_encodes(SHIPPED_PROFILES[p]), key=str),
        )
        for p in profiles
        if p != ACTIVE_PROFILE
    }
    per_request = Counter(encode for unit in units for encode in unit[3:])
    if ACTIVE_PROFILE in profiles:
        per_request.update(_default_encodes(SHIPPED_PROFILES[ACTIVE_PROFILE]))
    return per_request


def test_concurrent_fan_outs_encode_once_per_encoder_per_request(deployment, agent):
    profiles = sorted(SHIPPED_PROFILES)
    requests_count = 6
    barrier = threading.Barrier(requests_count)
    outcomes = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        outcome = _fan_out(agent, profiles)
        with lock:
            outcomes.append((outcome.searched, outcome.degraded))

    threads = [threading.Thread(target=worker) for _ in range(requests_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    calls = deployment["sidecars"].drain()

    assert outcomes == [(tuple(profiles), ())] * requests_count
    assert _encodes(calls) == Counter(
        {
            encode: count * requests_count
            for encode, count in _encodes_per_fan_out(profiles).items()
        }
    )


def test_a_dense_encoder_outage_degrades_only_the_dense_legs(deployment, agent):
    config_manager = deployment["config_manager"]
    dense = sorted(p for p, body in SHIPPED_PROFILES.items() if _is_dense(body))
    others = sorted(set(SHIPPED_PROFILES) - set(dense))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    healthy = semantic_embedder._CONFIGURED_REMOTE_URL
    system = config_manager.get_system_config()
    semantic_embedder.configure_semantic_embedder_defaults(
        remote_url=closed, model_name=None
    )
    config_manager.set_system_config(
        dataclasses.replace(
            system,
            inference_service_urls={
                **system.inference_service_urls,
                DENSE_SERVICE: closed,
            },
        )
    )
    try:
        outcome = _fan_out(agent, sorted(SHIPPED_PROFILES))
    finally:
        semantic_embedder.configure_semantic_embedder_defaults(
            remote_url=healthy, model_name=None
        )
        config_manager.set_system_config(system)

    recovered = _fan_out(agent, sorted(SHIPPED_PROFILES))

    assert len(dense) == 1
    assert outcome.searched == tuple(others)
    assert outcome.degraded == tuple((profile, "encode_failed") for profile in dense)
    assert (recovered.searched, recovered.degraded) == (
        tuple(sorted(SHIPPED_PROFILES)),
        (),
    )
