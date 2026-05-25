"""Content-schema back-ref integration: KG IDs landing on content docs.

After per-segment KG extraction, ``_extract_graph_per_segment`` PATCHes
``entity_ids`` / ``relation_ids`` / ``claim_ids`` onto the corresponding
content documents in Vespa. This file locks the following:

- ``entity_ids`` on a known frame doc sort byte-equal to the golden.
- ``relation_ids`` byte-equal to the SHA1-16 prefixes computed from
  ``Edge.edge_id`` against the deterministic inputs.
- ``claim_ids == relation_ids`` (every relation IS a claim).
- Each modality (video frame, video chunk, document, code, audio) has
  its full content document byte-equal to the per-modality golden.
- YQL join ``entity_ids contains "marie_curie"`` returns the locked
  doc_id set, sorted.

Tests run against the project-wide shared_memory_vespa container and a
real ColBERT/pylate sidecar (in-process fallback when
INFERENCE_SERVICE_URLS is absent). File-level skip fires when neither
service is reachable.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import Edge, normalize_name
from tests.utils.vespa_test_helpers import schema_full_name

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"


# --------------------------------------------------------------------- #
# Golden-file harness                                                   #
# --------------------------------------------------------------------- #


def assert_golden(actual, name: str):
    path = GOLDEN_DIR / name
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, (
        f"Golden {name} mismatch.\n"
        f"--- expected ---\n{expected}\n--- actual ---\n{actual_json}"
    )


# --------------------------------------------------------------------- #
# Availability probes for the file-level skip                           #
# --------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _vespa_running() -> bool:
    """Probe whether the docker daemon is up so the shared-vespa fixture
    can spawn a container, OR whether an existing shared Vespa is already
    reachable on its default config-API port (the in-cluster Vespa exposed
    via k3d serverlb).
    """
    # 1. Try the docker socket directly — that's what the shared_memory_vespa
    #    fixture uses to spin up its container.
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # 2. Fall back to a live Vespa on the default config port — k3d exposes
    #    cogniverse-vespa at localhost:19071 in dev environments.
    try:
        resp = requests.get("http://localhost:19071/ApplicationStatus", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _colbert_endpoint_from_env() -> Optional[str]:
    raw = os.environ.get("INFERENCE_SERVICE_URLS")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    url = parsed.get("colbert_pylate")
    if not url:
        return None
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=2)
        if resp.status_code == 200:
            return url
    except requests.RequestException:
        return None
    return None


def _pylate_sidecar_module_importable() -> bool:
    sidecar_path = Path("deploy/pylate/server.py")
    if not sidecar_path.exists():
        return False
    try:
        spec = importlib.util.spec_from_file_location(
            "pylate_server_probe_backrefs", str(sidecar_path)
        )
    except Exception:
        return False
    return spec is not None and spec.loader is not None


def _colbert_available() -> bool:
    return (
        _colbert_endpoint_from_env() is not None or _pylate_sidecar_module_importable()
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _vespa_running(),
        reason=(
            "Content-schema back-ref tests need a Vespa container the suite "
            "can spin up — Docker is unavailable in this environment."
        ),
    ),
    pytest.mark.skipif(
        not _colbert_available(),
        reason=(
            "Content-schema back-ref tests need a real ColBERT endpoint — "
            "set INFERENCE_SERVICE_URLS with a live colbert_pylate URL or "
            "make deploy/pylate/server.py importable so the in-process "
            "fallback can spawn it."
        ),
    ),
]


# --------------------------------------------------------------------- #
# Marie Curie shared fixture                                            #
# --------------------------------------------------------------------- #

TENANT_ID = "test"
VIDEO_ID = "marie_curie_30s"
GRAPH_BASE_SCHEMA = "knowledge_graph"
# Deployed name mirrors SchemaRegistry.deploy_schema: tenant_id is
# canonicalized ("test" -> "test:test") then ":" -> "_", so the suffix
# is doubled ("..._test_test").
GRAPH_TENANT_SCHEMA = schema_full_name(GRAPH_BASE_SCHEMA, TENANT_ID)

CONTENT_BASE_FRAME = "video_colpali_smol500_mv_frame"
CONTENT_TENANT_FRAME = schema_full_name(CONTENT_BASE_FRAME, TENANT_ID)

# The content doc this test PATCHes.
FRAME_DOC_ID = f"{VIDEO_ID}__seg_3"

TRANSCRIPT_TEXT = "Marie Curie discovered radium in 1898 at the Sorbonne."


# --------------------------------------------------------------------- #
# Per-modality seed-document field map                                  #
# --------------------------------------------------------------------- #
#
# Each schema declares a different set of required-ish fields. The
# per-modality parametrized test deploys one schema at a time and seeds
# a placeholder document the back-ref PATCH can target. We populate
# ONLY fields the target schema declares — feeding ``video_id`` into
# ``document_text`` (which has no such field) returns HTTP 400. The
# universal back-ref arrays (``entity_ids`` / ``relation_ids`` /
# ``claim_ids``) are added by the seed helper itself; this map covers
# everything else.
_MODALITY_FIELD_MAP: Dict[str, Dict[str, Any]] = {
    "video_frame": {
        "video_id": VIDEO_ID,
        "video_title": "Marie Curie 30s",
        "segment_id": 3,
        "start_time": 12.0,
        "end_time": 18.5,
        "segment_description": "",
        "audio_transcript": TRANSCRIPT_TEXT,
    },
    "video_chunk": {
        "video_id": VIDEO_ID,
        "video_title": "Marie Curie 30s",
        "segment_id": 3,
        "start_time": 12.0,
        "end_time": 18.5,
        "audio_transcript": TRANSCRIPT_TEXT,
    },
    "document": {
        "document_id": VIDEO_ID,
        "document_title": "Marie Curie 30s",
        "document_type": "transcript",
        "document_path": f"/tmp/{VIDEO_ID}.txt",
        "page_count": 1,
        "full_text": TRANSCRIPT_TEXT,
        "section_headings": "",
    },
    "code": {
        "code_id": VIDEO_ID,
        "file_path": f"/tmp/{VIDEO_ID}.py",
        "chunk_name": "marie_curie_chunk",
        "chunk_type": "function",
        "language": "python",
        "signature": "def marie_curie() -> None",
        "line_start": 1,
        "line_end": 10,
        "source_code": TRANSCRIPT_TEXT,
    },
    "audio": {
        "audio_id": VIDEO_ID,
        "audio_title": "Marie Curie 30s",
        "audio_transcript": TRANSCRIPT_TEXT,
        "audio_path": f"/tmp/{VIDEO_ID}.wav",
        "audio_duration": 30.0,
        "audio_language": "en",
    },
}


# Hand-curated SPO claims the ClaimExtractor is expected to produce for
# the Marie Curie transcript. Confidences are locked at 0.92, 0.88, and
# 0.87 — same values used in goldens/test_per_segment_kg_provenance.
_EXPECTED_CLAIMS = [
    {
        "subject": "Marie Curie",
        "predicate": "discovered",
        "object": "radium",
        "evidence_span": TRANSCRIPT_TEXT,
        "confidence": 0.92,
    },
    {
        "subject": "Marie Curie",
        "predicate": "worked_at",
        "object": "Sorbonne",
        "evidence_span": TRANSCRIPT_TEXT,
        "confidence": 0.88,
    },
    {
        "subject": "Marie Curie",
        "predicate": "discovered_in",
        "object": "1898",
        "evidence_span": TRANSCRIPT_TEXT,
        "confidence": 0.87,
    },
]


def _edge_id_for(subject: str, predicate: str, obj: str) -> str:
    """Compute the SHA1-16 ``edge_id`` the runtime would produce."""
    edge = Edge(
        tenant_id=TENANT_ID,
        source=subject,
        target=obj,
        relation=predicate,
        evidence_span=TRANSCRIPT_TEXT,
        segment_id="seg_3",
        ts_start=12.0,
        ts_end=18.5,
        modality="transcript",
        provenance="EXTRACTED",
        source_doc_id=VIDEO_ID,
        confidence=0.0,
    )
    return edge.edge_id


# --------------------------------------------------------------------- #
# Deterministic ClaimExtractor stub for the back-ref fixtures            #
# --------------------------------------------------------------------- #


class _DeterministicClaimExtractor:
    """A drop-in for ClaimExtractor that returns hand-curated edges.

    The full DSPy/RLM ClaimExtractor is exercised by
    ``test_claim_extractor_dspy.py``. For the back-ref test we need the
    edge IDs PATCHed onto content docs to be deterministic so the
    golden can pin them — using the real extractor here would couple
    this test to LLM endpoint availability and drift.
    """

    def extract(
        self,
        *,
        text: str,
        entity_hints: List[str],
        modality_hint: str,
        segment_anchor,
        tenant_id: str,
        source_doc_id: str,
    ) -> List[Edge]:
        if text.strip() != TRANSCRIPT_TEXT.strip():
            return []
        return [
            Edge(
                tenant_id=tenant_id,
                source=claim["subject"],
                target=claim["object"],
                relation=claim["predicate"],
                evidence_span=claim["evidence_span"],
                segment_id=segment_anchor.segment_id,
                ts_start=segment_anchor.ts_start,
                ts_end=segment_anchor.ts_end,
                modality=segment_anchor.modality,
                provenance="EXTRACTED",
                source_doc_id=source_doc_id,
                confidence=claim["confidence"],
            )
            for claim in _EXPECTED_CLAIMS
        ]


# --------------------------------------------------------------------- #
# ColBERT endpoint fixture (shared with cross-modal linker)              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def colbert_endpoint():
    env_url = _colbert_endpoint_from_env()
    if env_url is not None:
        yield env_url
        return

    import uvicorn  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "pylate_server_under_test_backrefs", "deploy/pylate/server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    app = mod.build_app(model_name="lightonai/LateOn", device="cpu", mode="colbert")
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("pylate /health did not come up within 180s — model load failed")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# --------------------------------------------------------------------- #
# Schema-deploy fixture                                                 #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def content_backref_env(shared_memory_vespa, colbert_endpoint):
    """Deploy the KG schema + the content frame schema, wire factories.

    Returns a dict with everything the back-ref tests need:
    - ``http_port``: shared_vespa data port
    - ``base_url``: ``f"http://localhost:{http_port}"``
    - ``graph_schema``: tenant-scoped KG schema name
    - ``content_frame_schema``: tenant-scoped video frame schema name
    """
    from tests.utils.vespa_test_helpers import deploy_tenant_schema  # noqa: PLC0415

    http_port = shared_memory_vespa["http_port"]
    base_url = f"http://localhost:{http_port}"

    deploy_tenant_schema(
        shared_memory_vespa,
        tenant_id=TENANT_ID,
        base_schema_name=GRAPH_BASE_SCHEMA,
        config_manager=shared_memory_vespa["config_manager"],
    )
    deploy_tenant_schema(
        shared_memory_vespa,
        tenant_id=TENANT_ID,
        base_schema_name=CONTENT_BASE_FRAME,
        config_manager=shared_memory_vespa["config_manager"],
    )

    # The runtime reads ``VESPA_URL`` to decide where to PATCH back-refs.
    prior_vespa_url = os.environ.get("VESPA_URL")
    prior_inference_urls = os.environ.get("INFERENCE_SERVICE_URLS")
    os.environ["VESPA_URL"] = base_url
    os.environ["INFERENCE_SERVICE_URLS"] = json.dumps(
        {"colbert_pylate": colbert_endpoint}
    )

    # Wire a GraphManager factory into the graph router so the ingestion
    # path can resolve a per-tenant manager.
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,  # noqa: PLC0415
    )
    from cogniverse_core.schemas.filesystem_loader import (
        FilesystemSchemaLoader,  # noqa: PLC0415
    )
    from cogniverse_foundation.config.manager import ConfigManager  # noqa: PLC0415
    from cogniverse_foundation.config.unified_config import (
        SystemConfig,  # noqa: PLC0415
    )
    from cogniverse_runtime.routers import graph as graph_router  # noqa: PLC0415
    from cogniverse_vespa.config.config_store import VespaConfigStore  # noqa: PLC0415

    BackendRegistry._backend_instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost", backend_port=http_port
    )
    cm = ConfigManager(store=config_store)
    # Empty telemetry endpoints so _lookup_artifact_manager returns None
    # (no Phoenix init) — the ClaimExtractor here is stubbed anyway.
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
            telemetry_url="",
            telemetry_collector_endpoint="",
        )
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    def _factory(tenant_id: str) -> GraphManager:
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id=tenant_id,
            config={
                "backend": {
                    "url": "http://localhost",
                    "config_port": shared_memory_vespa["config_port"],
                    "port": http_port,
                }
            },
            config_manager=cm,
            schema_loader=schema_loader,
        )
        return GraphManager(
            backend=backend,
            tenant_id=tenant_id,
            schema_name=schema_full_name(GRAPH_BASE_SCHEMA, tenant_id),
            colbert_endpoint_url=colbert_endpoint,
        )

    prior_factory = graph_router._graph_manager_factory
    graph_router.set_graph_manager_factory(_factory)

    # Point the ConfigManager singleton (which DocExtractor consults for the
    # GLiNER endpoint) at this fixture's config_manager. Its empty
    # inference_service_urls make _discover_gliner_url return None so GLiNER
    # loads locally instead of hitting the unresolvable in-cluster sidecar
    # URL — otherwise entity extraction silently degrades and entity_ids drift.
    import cogniverse_foundation.config.utils as _cfg_utils

    _prev_singleton = _cfg_utils._config_manager_singleton
    _cfg_utils._config_manager_singleton = cm

    yield {
        "http_port": http_port,
        "base_url": base_url,
        "graph_schema": GRAPH_TENANT_SCHEMA,
        "content_frame_schema": CONTENT_TENANT_FRAME,
        "config_manager": cm,
        "schema_loader": schema_loader,
        "config_port": shared_memory_vespa["config_port"],
    }

    _cfg_utils._config_manager_singleton = _prev_singleton
    graph_router._graph_manager_factory = prior_factory
    if prior_vespa_url is None:
        os.environ.pop("VESPA_URL", None)
    else:
        os.environ["VESPA_URL"] = prior_vespa_url
    if prior_inference_urls is None:
        os.environ.pop("INFERENCE_SERVICE_URLS", None)
    else:
        os.environ["INFERENCE_SERVICE_URLS"] = prior_inference_urls
    BackendRegistry._backend_instances.clear()


# --------------------------------------------------------------------- #
# Helpers — feed empty content docs, build processing_results, ingest    #
# --------------------------------------------------------------------- #


def _content_doc_url(base_url: str, schema: str, doc_id: str) -> str:
    # Content docs live under the ``content`` namespace (the embedding feed
    # writes ``id:content:<schema>::<id>``; see cogniverse_vespa
    # ingestion_client/backend). The runtime back-ref PATCH targets
    # ``/document/v1/content/<schema>/docid/<id>`` — seed + read here under
    # the same namespace so the PATCH lands on the doc the test inspects.
    return f"{base_url}/document/v1/content/{schema}/docid/{doc_id}"


def _feed_empty_content_doc(
    base_url: str,
    schema: str,
    doc_id: str,
    modality: str = "video_frame",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a placeholder content document the back-ref PATCH can target.

    Vespa partial updates against a missing doc would 404 without
    ``?create=true``; the runtime PATCH doesn't pass that flag, so we
    pre-create the doc with empty back-ref arrays.

    ``modality`` selects which entry of ``_MODALITY_FIELD_MAP`` we use
    to populate non-back-ref fields. Feeding fields the target schema
    doesn't declare returns HTTP 400 from Vespa, so each modality has
    its own field set.
    """
    if modality not in _MODALITY_FIELD_MAP:
        raise KeyError(
            f"unknown modality {modality!r}; expected one of "
            f"{sorted(_MODALITY_FIELD_MAP)}"
        )
    fields: Dict[str, Any] = dict(_MODALITY_FIELD_MAP[modality])
    fields.update(
        {
            "entity_ids": [],
            "relation_ids": [],
            "claim_ids": [],
        }
    )
    if extra_fields:
        fields.update(extra_fields)

    resp = requests.post(
        _content_doc_url(base_url, schema, doc_id),
        json={"fields": fields},
        timeout=10,
    )
    assert resp.status_code in (200, 201), (
        f"failed to seed content doc {doc_id}: {resp.status_code} {resp.text[:500]}"
    )


def _get_content_doc(base_url: str, schema: str, doc_id: str) -> Optional[Dict]:
    for _ in range(20):
        try:
            resp = requests.get(_content_doc_url(base_url, schema, doc_id), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


def _build_processing_results(*, schema: str, doc_id: str) -> Dict[str, Any]:
    """Construct the ``processing_results`` shape ``_extract_graph_per_segment`` expects."""
    return {
        "transcript": {
            "segments": [
                # Pad seg_0..seg_2 so the seg_3 index matches the fixture anchor.
                {"text": "", "start": 0.0, "end": 0.0},
                {"text": "", "start": 0.0, "end": 0.0},
                {"text": "", "start": 0.0, "end": 0.0},
                {
                    "text": TRANSCRIPT_TEXT,
                    "start": 12.0,
                    "end": 18.5,
                },
            ]
        },
        "keyframes": {"keyframes": []},
        "descriptions": {"descriptions": {}},
        "document_files": [],
        "fed_documents": [
            {
                "schema": schema,
                "doc_id": doc_id,
                "segment_id": "seg_3",
            }
        ],
    }


async def _run_extract(
    processing_results: Dict[str, Any], config_manager: Any
) -> Dict[str, Any]:
    """Invoke ``_extract_graph_per_segment`` with a deterministic ClaimExtractor.

    Monkey-patches ``ClaimExtractor`` inside the routers module so the
    SPO edges PATCHed onto content docs are deterministic — see the
    docstring on ``_DeterministicClaimExtractor``.
    """
    import cogniverse_runtime.routers.ingestion as ingestion_router  # noqa: PLC0415
    from cogniverse_agents.graph import claim_extractor as ce_module  # noqa: PLC0415

    real_cls = ce_module.ClaimExtractor

    class _StubFactory:
        def __init__(self, *_, **__):
            pass

        def extract(self, **kwargs):
            return _DeterministicClaimExtractor().extract(**kwargs)

    ce_module.ClaimExtractor = _StubFactory
    try:
        result = await ingestion_router._extract_graph_per_segment(
            processing_results=processing_results,
            source_doc_id=VIDEO_ID,
            tenant_id=TENANT_ID,
            config_manager=config_manager,
        )
    finally:
        ce_module.ClaimExtractor = real_cls
    return result


# --------------------------------------------------------------------- #
# entity_ids on the seg_3 frame doc, sorted                              #
# --------------------------------------------------------------------- #


class TestContentBackrefsEntityIds:
    """``entity_ids`` on the seg_3 video frame doc is byte-equal."""

    def test_entity_ids_byte_equal(self, content_backref_env):
        env = content_backref_env
        _feed_empty_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )

        asyncio.run(
            _run_extract(
                _build_processing_results(
                    schema=env["content_frame_schema"], doc_id=FRAME_DOC_ID
                ),
                env["config_manager"],
            )
        )

        doc = _get_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )
        assert doc is not None, f"frame doc {FRAME_DOC_ID} not retrievable after PATCH"
        entity_ids = sorted(doc.get("fields", {}).get("entity_ids", []))
        assert_golden(entity_ids, "content_backrefs_entity_ids.json")


# --------------------------------------------------------------------- #
# relation_ids equal to the deterministic SHA1-16 prefixes               #
# --------------------------------------------------------------------- #


class TestContentBackrefsRelationIds:
    """``relation_ids`` byte-equal to ``[edge_id(...), edge_id(...), ...]``.

    Each value is the SHA1-16 prefix of the normalized triple — the
    ``Edge.edge_id`` property's contract. The list is sorted before
    asserting so insertion order from the dict doesn't leak into the
    golden.
    """

    def test_relation_ids_byte_equal(self, content_backref_env):
        env = content_backref_env
        _feed_empty_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )

        asyncio.run(
            _run_extract(
                _build_processing_results(
                    schema=env["content_frame_schema"], doc_id=FRAME_DOC_ID
                ),
                env["config_manager"],
            )
        )

        doc = _get_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )
        assert doc is not None
        relation_ids = sorted(doc.get("fields", {}).get("relation_ids", []))

        expected_sorted = sorted(
            _edge_id_for(c["subject"], c["predicate"], c["object"])
            for c in _EXPECTED_CLAIMS
        )
        assert len(relation_ids) == 3, (
            f"expected 3 relation_ids, got {len(relation_ids)}: {relation_ids}"
        )
        assert relation_ids == expected_sorted, (
            f"runtime relation_ids {relation_ids} != computed edge_ids "
            f"{expected_sorted}"
        )
        assert_golden(relation_ids, "content_backrefs_relation_ids.json")


# --------------------------------------------------------------------- #
# claim_ids == relation_ids                                              #
# --------------------------------------------------------------------- #


class TestContentBackrefsClaimIdsEqualRelationIds:
    """Every relation IS a claim — exact list equality including order."""

    def test_claim_ids_equal_relation_ids(self, content_backref_env):
        env = content_backref_env
        _feed_empty_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )

        asyncio.run(
            _run_extract(
                _build_processing_results(
                    schema=env["content_frame_schema"], doc_id=FRAME_DOC_ID
                ),
                env["config_manager"],
            )
        )

        doc = _get_content_doc(
            env["base_url"], env["content_frame_schema"], FRAME_DOC_ID
        )
        assert doc is not None
        fields = doc.get("fields", {})
        relation_ids = fields.get("relation_ids", [])
        claim_ids = fields.get("claim_ids", [])
        assert claim_ids == relation_ids, (
            f"claim_ids and relation_ids must be identical (including order); "
            f"got claim_ids={claim_ids} relation_ids={relation_ids}"
        )


# --------------------------------------------------------------------- #
# Per-modality coverage                                                  #
# --------------------------------------------------------------------- #


_MODALITY_FIXTURES = [
    pytest.param(
        "video_chunk",
        "video_videoprism_base_mv_chunk_30s",
        "content_backrefs_video_chunk.json",
        id="video_chunk",
    ),
    pytest.param(
        "document",
        "document_text",
        "content_backrefs_document.json",
        id="document",
    ),
    pytest.param(
        "code",
        "code_lateon_mv",
        "content_backrefs_code.json",
        id="code",
    ),
    pytest.param(
        "audio",
        "audio_content",
        "content_backrefs_audio.json",
        id="audio",
    ),
]


class TestContentBackrefsPerModalityCoverage:
    """``entity_ids`` populated on each modality's content document.

    Spec: ingest one fixture per modality and lock the resulting
    document's ``entity_ids`` array against the per-modality golden.
    We deliberately do NOT lock the full document dict here — Vespa
    Document v1 reads include attribute fields whose presence depends
    on the deployed schema variant; the contract under test is the
    KG back-refs subset.
    """

    @pytest.mark.parametrize("modality_label,base_schema,golden", _MODALITY_FIXTURES)
    def test_modality_entity_ids_byte_equal(
        self, content_backref_env, modality_label, base_schema, golden
    ):
        env = content_backref_env

        from tests.utils.vespa_test_helpers import (  # noqa: PLC0415
            deploy_tenant_schema,
        )

        deploy_tenant_schema(
            {
                "http_port": env["http_port"],
                "config_port": env["config_port"],
            },
            tenant_id=TENANT_ID,
            base_schema_name=base_schema,
            config_manager=env["config_manager"],
        )

        tenant_schema = schema_full_name(base_schema, TENANT_ID)
        doc_id = f"{VIDEO_ID}__seg_3__{modality_label}"

        # Seed the placeholder with ONLY fields the per-modality schema
        # declares (see ``_MODALITY_FIELD_MAP``). Feeding e.g.
        # ``video_id`` into ``document_text`` returns HTTP 400. The
        # back-ref PATCH only touches the array fields.
        _feed_empty_content_doc(
            env["base_url"],
            tenant_schema,
            doc_id,
            modality=modality_label,
        )

        asyncio.run(
            _run_extract(
                _build_processing_results(schema=tenant_schema, doc_id=doc_id),
                env["config_manager"],
            )
        )

        doc = _get_content_doc(env["base_url"], tenant_schema, doc_id)
        assert doc is not None, (
            f"{modality_label} doc {doc_id} not retrievable after PATCH"
        )
        entity_ids = sorted(doc.get("fields", {}).get("entity_ids", []))
        assert_golden(entity_ids, golden)


# --------------------------------------------------------------------- #
# YQL join correctness: entity_ids contains "marie_curie"                #
# --------------------------------------------------------------------- #


class TestContentBackrefsYqlJoin:
    """``select * from sources <frame_schema> where entity_ids contains "marie_curie"``
    returns the locked set of doc_ids (sorted)."""

    def test_yql_join_returns_locked_doc_ids(self, content_backref_env):
        env = content_backref_env

        # Two seg_3 docs under different "videos" both carrying the
        # marie_curie entity back-ref; the YQL join must surface both.
        doc_id_primary = FRAME_DOC_ID
        doc_id_secondary = "marie_curie_60s__seg_3"

        for doc_id in (doc_id_primary, doc_id_secondary):
            _feed_empty_content_doc(
                env["base_url"], env["content_frame_schema"], doc_id
            )
            asyncio.run(
                _run_extract(
                    _build_processing_results(
                        schema=env["content_frame_schema"], doc_id=doc_id
                    ),
                    env["config_manager"],
                )
            )

        # Vespa needs a beat between feed/PATCH and visibility on a
        # YQL search; the shared container's indexer is async.
        time.sleep(2)

        # ``select *`` makes Vespa surface the full document summary
        # (including ``documentid``) on each hit. Selecting an explicit
        # field list with a non-existent ``doc_id`` column suppresses
        # the summary fetch and the hit's top-level ``id`` collapses to
        # the internal ``index:...`` identifier.
        yql = (
            f"select * from sources {env['content_frame_schema']} where "
            f'entity_ids contains "{normalize_name("Marie Curie")}"'
        )
        resp = requests.post(
            f"{env['base_url']}/search/",
            json={"yql": yql, "hits": 100},
            timeout=15,
        )
        assert resp.status_code == 200, (
            f"YQL query failed: {resp.status_code} {resp.text[:500]}"
        )
        data = resp.json()
        hits = (data.get("root") or {}).get("children") or []
        # Each hit's top-level ``id`` is the Vespa document id of the
        # form ``id:<namespace>:<doctype>::<user_doc_id>``. The suffix
        # after ``::`` is the user-facing doc_id we seeded.
        doc_ids = []
        for h in hits:
            raw_id = h.get("id") or ""
            if "::" not in raw_id:
                continue
            doc_ids.append(raw_id.split("::")[-1])
        doc_ids = sorted(d for d in doc_ids if d)

        assert doc_id_primary in doc_ids, (
            f"YQL join missed primary doc {doc_id_primary}; got {doc_ids} "
            f"(raw hits: {[h.get('id') for h in hits]})"
        )
        assert doc_id_secondary in doc_ids, (
            f"YQL join missed secondary doc {doc_id_secondary}; got {doc_ids} "
            f"(raw hits: {[h.get('id') for h in hits]})"
        )
        assert_golden(doc_ids, "content_backrefs_yql_join.json")
