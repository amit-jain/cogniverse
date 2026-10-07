"""Each query tensor input is filled by an encoder whose output fits it.

A rank profile input scored against a field embedded by a service of its own
(``inference_services`` keyed by the field) takes that service's text
encoder; every other input takes the profile's query encoder. The audio
``hybrid_acoustic_bm25`` profile binds a 512-dim CLAP vector to
``query(acoustic_query)``; it used to get the profile's LateOn per-token
vectors, which no single-vector input can bind.
"""

from __future__ import annotations

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_agents.graph.graph_manager import _GRAPH_BASE_SCHEMA, GraphManager
from cogniverse_core.memory.manager import (
    MEMORY_BASE_SCHEMA,
    MEMORY_EMBEDDING_DIMS,
    build_memory_profile,
)
from cogniverse_core.query import encoders as encoders_module
from cogniverse_core.query.encoders import (
    ClapTextQueryEncoder,
    EncoderNotConfiguredError,
    EncoderUnavailableError,
    QueryEncoderFactory,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.query_inputs import (
    QueryInputEncoding,
    encoding_fits,
    profile_encoder_encoding,
    query_input_encodings,
)
from cogniverse_vespa.ranking_strategy_extractor import RankingStrategyExtractor
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.memory_store import InMemoryConfigStore

SCHEMAS_DIR = Path("configs/schemas")
SHIPPED_PROFILES = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
]
AUDIO_PROFILE = "audio_clap_semantic"
CLAP_VECTOR = [round(0.001 * i, 3) for i in range(512)]
LATEON_TOKENS = [[0.01] * 128] * 4


def _rank_config(info) -> dict:
    return {"inputs": info.inputs, "input_fields": info.input_fields}


def _graph_profile() -> dict:
    """GraphManager encodes knowledge_graph queries itself, with its default
    ColBERT model; that model's width is the one a shipped profile declares."""
    model = inspect.signature(GraphManager.__init__).parameters["colbert_model"].default
    (width,) = {
        p["schema_config"]["embedding_dim"]
        for p in SHIPPED_PROFILES.values()
        if p.get("model_loader") == "colbert"
        and (p.get("semantic_model") or p.get("embedding_model")) == model
    }
    return {"embedding_type": "multi_vector", "schema_config": {"embedding_dim": width}}


def _encoders_by_schema() -> dict[str, dict[str, dict]]:
    """Every encoder that fills a schema's query inputs, by schema: the
    shipped search profiles, the agent-memory profile and GraphManager."""
    by_schema: dict[str, dict[str, dict]] = {}
    for name, profile in SHIPPED_PROFILES.items():
        by_schema.setdefault(profile["schema_name"], {})[name] = profile
    by_schema.setdefault(MEMORY_BASE_SCHEMA, {})["memory"] = build_memory_profile(
        MEMORY_BASE_SCHEMA, MEMORY_EMBEDDING_DIMS
    )
    by_schema.setdefault(_GRAPH_BASE_SCHEMA, {})["graph"] = _graph_profile()
    return by_schema


def _schemas_with_inputs() -> dict[str, dict]:
    found = {}
    for path in sorted(SCHEMAS_DIR.glob("*_schema.json")):
        strategies = RankingStrategyExtractor().extract_from_schema(path)
        with_inputs = {n: i for n, i in strategies.items() if i.inputs}
        if with_inputs:
            found[path.name.removesuffix("_schema.json")] = with_inputs
    return found


class TestEveryQueryInputFitsItsEncoder:
    def test_every_schema_with_query_inputs_has_an_encoder(self):
        encoders = _encoders_by_schema()

        assert sorted(s for s in _schemas_with_inputs() if not encoders.get(s)) == []

    def test_every_query_input_fits_the_encoder_that_fills_it(self):
        encoders = _encoders_by_schema()
        misfits = []
        for base, strategies in _schemas_with_inputs().items():
            for profile_name, profile in encoders[base].items():
                for strategy, info in strategies.items():
                    encodings = query_input_encodings(profile, _rank_config(info))
                    for input_name, input_type in info.inputs.items():
                        if not encoding_fits(encodings[input_name], input_type):
                            misfits.append(
                                (
                                    base,
                                    profile_name,
                                    strategy,
                                    input_name,
                                    input_type,
                                    encodings[input_name],
                                )
                            )

        assert misfits == []

    def test_acoustic_inputs_take_the_clap_text_encoder(self):
        strategies = _schemas_with_inputs()["audio_content"]
        profile = SHIPPED_PROFILES[AUDIO_PROFILE]

        assert {
            strategy: query_input_encodings(profile, _rank_config(info))
            for strategy, info in strategies.items()
            if "acoustic_query" in info.inputs
        } == {
            "acoustic_similarity": {
                "acoustic_query": QueryInputEncoding("clap_embed", False, 512)
            },
            "hybrid_acoustic_bm25": {
                "acoustic_query": QueryInputEncoding("clap_embed", False, 512)
            },
        }

    def test_the_profile_encoder_does_not_fit_the_acoustic_input(self):
        """LateOn's per-token vectors, which the acoustic input used to get,
        are rejected by the fit rule."""
        lateon = profile_encoder_encoding(SHIPPED_PROFILES[AUDIO_PROFILE])

        assert lateon == QueryInputEncoding(None, True, 128)
        assert not encoding_fits(lateon, "tensor<float>(v[512])")
        assert encoding_fits(lateon, "tensor<float>(querytoken{}, v[128])")
        assert encoding_fits(lateon, "tensor<int8>(querytoken{}, v[16])")

    @pytest.mark.parametrize(
        ("encoding", "input_type", "fits"),
        [
            (QueryInputEncoding(None, False, 768), "tensor<float>(v[768])", True),
            (QueryInputEncoding(None, False, 768), "tensor<int8>(v[96])", True),
            (QueryInputEncoding(None, False, 768), "tensor<float>(d0[768])", True),
            (QueryInputEncoding(None, False, 768), "tensor<float>(v[512])", False),
            (QueryInputEncoding(None, False, 768), "tensor<int8>(v[768])", False),
            (
                QueryInputEncoding(None, False, 128),
                "tensor<float>(querytoken{}, v[128])",
                False,
            ),
            (QueryInputEncoding(None, True, 320), "tensor<float>(v[320])", False),
            (QueryInputEncoding(None, True, None), "tensor<float>(v[320])", False),
        ],
    )
    def test_fit_rule(self, encoding, input_type, fits):
        assert encoding_fits(encoding, input_type) is fits


class _Sidecars(BaseHTTPRequestHandler):
    """colbert_pylate ``/pooling`` and clap_embed ``/embed/text`` with their
    production response shapes; a 5xx when the server is told to fail."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(self.path)
        if self.server.failing:
            self.send_response(503)
            self.end_headers()
            return
        if self.path == "/pooling":
            out = {"data": [{"data": LATEON_TOKENS} for _ in body["input"]]}
        elif self.path == "/embed/text":
            out = {"vec": CLAP_VECTOR}
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


@pytest.fixture
def sidecars():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Sidecars)
    server.calls = []
    server.failing = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def clean_encoder_cache():
    QueryEncoderFactory._encoder_cache.clear()
    yield
    QueryEncoderFactory._encoder_cache.clear()


def _system_config(**urls) -> SystemConfig:
    return SystemConfig(
        backend_url="http://127.0.0.1", backend_port=9, inference_service_urls=urls
    )


class TestServiceEncoderFactory:
    def test_clap_service_builds_the_clap_text_encoder(
        self, sidecars, clean_encoder_cache
    ):
        url = f"http://127.0.0.1:{sidecars.server_port}"
        encoder = QueryEncoderFactory.create_service_encoder(
            AUDIO_PROFILE, "clap_embed", _system_config(clap_embed=url)
        )

        assert isinstance(encoder, ClapTextQueryEncoder)
        assert encoder.encode("fire").tolist() == pytest.approx(CLAP_VECTOR)
        assert encoder.get_embedding_dim() == 512
        assert sidecars.calls == ["/embed/text"]

    def test_unconfigured_service_url_is_a_configuration_error(
        self, clean_encoder_cache
    ):
        with pytest.raises(EncoderNotConfiguredError, match="clap_embed") as info:
            QueryEncoderFactory.create_service_encoder(
                AUDIO_PROFILE, "clap_embed", _system_config()
            )
        assert info.value.profile == AUDIO_PROFILE

    def test_service_without_a_text_encoder_is_a_configuration_error(
        self, clean_encoder_cache
    ):
        with pytest.raises(EncoderNotConfiguredError, match="face_embed"):
            QueryEncoderFactory.create_service_encoder(
                AUDIO_PROFILE, "face_embed", _system_config(face_embed="http://x")
            )

    def test_concurrent_first_builds_share_one_encoder(
        self, monkeypatch, clean_encoder_cache
    ):
        built = []
        release = threading.Barrier(16)

        class _Counting(ClapTextQueryEncoder):
            def __init__(self, url):
                built.append(url)
                super().__init__(url)

        monkeypatch.setitem(
            encoders_module.SERVICE_TEXT_ENCODERS, "clap_embed", _Counting
        )
        config = _system_config(clap_embed="http://127.0.0.1:9")

        def build(_):
            release.wait(timeout=30)
            return QueryEncoderFactory.create_service_encoder(
                AUDIO_PROFILE, "clap_embed", config
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            encoders = list(pool.map(build, range(16)))

        assert built == ["http://127.0.0.1:9"]
        assert {id(e) for e in encoders} == {id(encoders[0])}


def _backend(sidecars) -> tuple[VespaSearchBackend, list]:
    url = f"http://127.0.0.1:{sidecars.server_port}"
    config_manager = ConfigManager(store=InMemoryConfigStore())
    config_manager.set_system_config(_system_config(colbert_pylate=url, clap_embed=url))
    backend = VespaSearchBackend(
        config={
            "url": "http://127.0.0.1",
            "port": 9,
            "profiles": {AUDIO_PROFILE: SHIPPED_PROFILES[AUDIO_PROFILE]},
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=lambda tenant, base: True,
        enable_connection_pool=False,
    )
    built = []
    build_query = backend._build_query

    def recording_build_query(*args, **kwargs):
        built.append(build_query(*args, **kwargs))
        raise _QueryBuilt

    backend._build_query = recording_build_query
    return backend, built


class _QueryBuilt(Exception):
    """Stops the search once its Vespa query is built."""


def _search(backend, strategy: str):
    with pytest.raises(_QueryBuilt):
        backend.search(
            {
                "query": "fire",
                "type": "audio",
                "profile": AUDIO_PROFILE,
                "strategy": strategy,
                "tenant_id": "acoustic:unit",
                "top_k": 5,
            }
        )


@pytest.mark.usefixtures("telemetry_manager_without_phoenix", "clean_encoder_cache")
class TestOnDemandQueryEncoding:
    def test_hybrid_acoustic_binds_the_clap_text_vector(self, sidecars):
        backend, built = _backend(sidecars)

        _search(backend, "hybrid_acoustic_bm25")

        (params,) = built
        assert params["input.query(acoustic_query)"] == pytest.approx(CLAP_VECTOR)
        assert [k for k in params if k.startswith("input.query(")] == [
            "input.query(acoustic_query)"
        ]
        assert "nearestNeighbor(acoustic_embedding, acoustic_query)" in params["yql"]
        assert params["userQuery"] == "fire"
        assert sidecars.calls == ["/embed/text"]

    def test_semantic_inputs_still_take_the_profile_encoder(self, sidecars):
        backend, built = _backend(sidecars)

        _search(backend, "phased_semantic")

        (params,) = built
        assert params["input.query(qt)"] == {
            str(i): row for i, row in enumerate(np.float32(LATEON_TOKENS).tolist())
        }
        assert set(params["input.query(qtb)"]) == {"0", "1", "2", "3"}
        assert sidecars.calls == ["/pooling"]

    def test_clap_outage_is_an_unavailable_encoder_naming_its_service(self, sidecars):
        backend, built = _backend(sidecars)
        sidecars.failing = True

        with pytest.raises(EncoderUnavailableError) as info:
            backend.search(
                {
                    "query": "fire",
                    "type": "audio",
                    "profile": AUDIO_PROFILE,
                    "strategy": "hybrid_acoustic_bm25",
                    "tenant_id": "acoustic:unit",
                    "top_k": 5,
                }
            )

        assert (info.value.profile, info.value.service, info.value.endpoint) == (
            AUDIO_PROFILE,
            "clap_embed",
            f"http://127.0.0.1:{sidecars.server_port}",
        )
        assert built == []
        assert sidecars.calls == ["/embed/text"]
