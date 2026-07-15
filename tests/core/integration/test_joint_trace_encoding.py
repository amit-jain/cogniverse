"""Integration tests for the joint-trace ColBERT query encoding.

Exercises ``ColBERTQueryEncoder.encode(query, trace=...)`` against the
real vLLM ``lightonai/LateOn`` sidecar the ``pylate_server`` fixture
provisions. The sidecar is self-provisioned, so these run on a fresh
checkout without any pre-started service.

The four cases lock:
  - Empty-trace encoding shape + array.
  - Non-empty-trace encoding array.
  - MaxSim cosine against a pinned document encoding, plus the semantic
    lift (a radioactivity CoT trace raises similarity to the radium doc).
  - Top-1 retrieval through ``GraphManager.search_nodes`` swaps the top
    node when a CoT trace is supplied (the AgentIR lift).

Encoding goldens compare with an absolute tolerance, not byte-equality:
served-model float output drifts a few percent across vLLM/BLAS/hardware
versions. See ``GOLDEN_ATOL`` for the sizing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# File-level skip: the ColBERT pylate sidecar must be reachable. Without it
# every encode call raises and there's no byte-stable output to lock.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Golden file helpers
# ---------------------------------------------------------------------------

# Goldens are committed (see ../goldens/) so this suite runs on a fresh
# checkout. They bind to model ``lightonai/LateOn`` served by vLLM with the
# ``ColBERTModernBertModel`` hf-override (the pylate_server fixture). Re-record
# after a deliberate model/serving change with ``RECORD_GOLDEN=1 uv run pytest
# tests/core/integration/test_joint_trace_encoding.py``.
GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"

# Served-model embeddings are not byte-stable across serving-stack, BLAS, or
# hardware versions. Measured cross-version drift between the 2026-06-12
# recording and the current vLLM stack was max |Δ|≈0.028 (mean ≈0.006) on
# values in [-0.52, 0.32]. An absolute tolerance of 0.05 (~2x that drift)
# tolerates the numeric variance while still catching the real regressions —
# a wrong model, a wrong architecture (shape change), or NaN/Inf.
GOLDEN_ATOL = 0.05
# The MaxSim cosine scalars drift ~3e-3 across the same versions; 0.02 leaves
# a comfortable margin while still anchoring the value.
COSINE_ATOL = 0.02


def assert_golden_json(actual, name: str) -> None:
    path = GOLDEN_DIR / name
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, f"Golden mismatch for {name}"


def assert_golden_npy(actual_array: np.ndarray, name: str) -> None:
    path = GOLDEN_DIR / name
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, actual_array)
        return
    expected = np.load(path)
    assert actual_array.shape == expected.shape, (
        f"Golden shape mismatch for {name}: {actual_array.shape} != {expected.shape}"
    )
    assert np.allclose(actual_array, expected, rtol=0.0, atol=GOLDEN_ATOL), (
        f"Golden drift for {name} exceeds atol={GOLDEN_ATOL}: "
        f"max|Δ|={np.abs(actual_array - expected).max():.4f}"
    )


# ---------------------------------------------------------------------------
# Encoder fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def colbert_encoder(pylate_server):
    from cogniverse_core.query.encoders import ColBERTQueryEncoder

    return ColBERTQueryEncoder(
        model_name="lightonai/LateOn",
        embedding_dim=128,
        inference_service_url=pylate_server,
    )


# ---------------------------------------------------------------------------
# encode("discover", trace="") returns (N, 128) array byte-equal golden
# ---------------------------------------------------------------------------


def test_encode_no_trace_shape_and_array(colbert_encoder):
    arr = colbert_encoder.encode("discover", trace="")
    assert isinstance(arr, np.ndarray), type(arr)
    assert arr.ndim == 2, arr.shape
    assert arr.shape[1] == 128, arr.shape
    assert_golden_npy(arr, "colbert_encode_discover_no_trace.npy")


# ---------------------------------------------------------------------------
# encode("discover", trace=<cot>) matches golden within tolerance
# ---------------------------------------------------------------------------


def test_encode_with_trace_matches_golden(colbert_encoder):
    arr = colbert_encoder.encode(
        "discover",
        trace="medical history involving radioactivity research",
    )
    assert arr.shape[1] == 128, arr.shape
    assert_golden_npy(arr, "colbert_encode_discover_with_trace.npy")


# ---------------------------------------------------------------------------
# MaxSim cosine between both encodings and the pinned doc encoding
# ---------------------------------------------------------------------------


def _maxsim_cosine(query_arr: np.ndarray, doc_arr: np.ndarray) -> float:
    """MaxSim cosine: per query token, take max cosine over doc tokens,
    then average across query tokens. Matches the ranking
    profile used by the ColBERT Vespa schema."""
    qn = query_arr / (np.linalg.norm(query_arr, axis=1, keepdims=True) + 1e-9)
    dn = doc_arr / (np.linalg.norm(doc_arr, axis=1, keepdims=True) + 1e-9)
    sim = qn @ dn.T
    per_q_max = sim.max(axis=1)
    return float(per_q_max.mean())


def test_cosine_pairs_match_golden_and_trace_lifts_similarity(colbert_encoder):
    arr_no_trace = colbert_encoder.encode("discover", trace="")
    arr_with_trace = colbert_encoder.encode(
        "discover",
        trace="medical history involving radioactivity research",
    )
    doc_path = GOLDEN_DIR / "colbert_encode_doc_marie.npy"
    if RECORD_GOLDEN:
        # Record the doc encoding alongside the cosine pair so both move
        # in lockstep when re-running with RECORD_GOLDEN=1.
        doc_arr = colbert_encoder.encode(
            "Marie Curie discovered radium in 1898", trace=""
        )
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(doc_path, doc_arr)
    else:
        doc_arr = np.load(doc_path)

    no_trace_cosine = _maxsim_cosine(arr_no_trace, doc_arr)
    with_trace_cosine = _maxsim_cosine(arr_with_trace, doc_arr)

    # The semantic contract, portable across serving stacks: a CoT trace about
    # radioactivity research must raise MaxSim similarity to the radium doc.
    assert with_trace_cosine > no_trace_cosine, (no_trace_cosine, with_trace_cosine)

    pairs = {
        "no_trace_cosine": round(no_trace_cosine, 4),
        "with_trace_cosine": round(with_trace_cosine, 4),
    }
    if RECORD_GOLDEN:
        assert_golden_json(pairs, "cosine_pairs_discover.json")
        return
    expected = json.loads((GOLDEN_DIR / "cosine_pairs_discover.json").read_text())
    for key, value in pairs.items():
        assert abs(value - expected[key]) <= COSINE_ATOL, (
            f"{key} drift {abs(value - expected[key]):.4f} > atol={COSINE_ATOL} "
            f"(got {value}, golden {expected[key]})"
        )


# ---------------------------------------------------------------------------
# Top-1 retrieval via GraphManager.search_nodes
# ---------------------------------------------------------------------------


def _graph_manager_available() -> bool:
    """Return True when a GraphManager can be constructed against a real
    Vespa with the KG schema and colbert sidecar reachable."""
    try:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
        config = json.loads(config_path.read_text())
        backend = config.get("backend", {})
        url = backend.get("url") or os.environ.get("VESPA_URL")
        port = backend.get("port") or os.environ.get("VESPA_PORT")
        if not (url and port):
            return False
        resp = httpx.get(f"{url}:{port}/ApplicationStatus", timeout=2.0)
        return resp.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module")
def seeded_graph_manager(pylate_server):
    """Deploy knowledge_graph schema for tenant=g4test in live Vespa and
    upsert the Marie Curie node set the search test exercises.

    Real Vespa, real ColBERT — no mocks. The schema deploy uses the same
    SchemaRegistry pathway production ingestion uses, so this is
    re-runnable: deploy is idempotent at the registry layer, upsert is
    idempotent on deterministic node_id.
    """
    from cogniverse_agents.graph.graph_manager import GraphManager
    from cogniverse_agents.graph.graph_schema import (
        ExtractionResult,
        Mention,
        Node,
    )
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,
        get_backend_registry,
    )
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    backend_cfg = json.loads(
        (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
    ).get("backend", {})
    http_port = int(backend_cfg.get("port") or 8080)
    config_port = int(backend_cfg.get("config_port") or 19071)
    base_url = backend_cfg.get("url") or "http://localhost"
    tenant_id = "g4test"

    store = VespaConfigStore(backend_url=base_url, backend_port=http_port)
    cm = ConfigManager(store=store)
    cm.set_system_config(SystemConfig(backend_url=base_url, backend_port=http_port))
    schema_loader = FilesystemSchemaLoader(
        Path(__file__).resolve().parents[3] / "configs" / "schemas"
    )

    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    registry = get_backend_registry()
    ingest_backend = registry.get_ingestion_backend(
        name="vespa",
        config={
            "backend": {
                "url": base_url,
                "port": http_port,
                "config_port": config_port,
            }
        },
        config_manager=cm,
        schema_loader=schema_loader,
        tenant_id=tenant_id,
    )
    schema_name = ingest_backend.schema_registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name="knowledge_graph",
    )

    search_backend = registry.get_search_backend(
        name="vespa",
        config={
            "backend": {
                "url": base_url,
                "port": http_port,
                "config_port": config_port,
            }
        },
        config_manager=cm,
        schema_loader=schema_loader,
    )
    mgr = GraphManager(
        backend=search_backend,
        tenant_id=tenant_id,
        schema_name=schema_name,
        colbert_endpoint_url=pylate_server,
    )

    seg_anchor = Mention(
        source_doc_id="marie_curie_30s",
        segment_id="seg_3",
        ts_start=12.0,
        ts_end=18.5,
        modality="transcript",
        evidence_span="Marie Curie discovered radium in 1898 at the Sorbonne.",
    )
    nobel_anchor = Mention(
        source_doc_id="marie_curie_30s",
        segment_id="seg_4",
        ts_start=18.5,
        ts_end=25.0,
        modality="transcript",
        evidence_span="She later won the Nobel Prize in Physics.",
    )
    nodes = [
        Node(
            tenant_id=tenant_id,
            name=name,
            description=desc,
            kind="entity",
            mentions=[anchor],
        )
        for name, desc, anchor in [
            (
                "Marie Curie",
                "Physicist who discovered radium and won the Nobel Prize.",
                seg_anchor,
            ),
            (
                "radium",
                (
                    "Highly radioactive chemical element central to Marie Curie's "
                    "medical history of radioactivity research; the discovery "
                    "established a new field of radioactivity science."
                ),
                seg_anchor,
            ),
            ("1898", "Calendar year.", seg_anchor),
            (
                "Sorbonne",
                "Parisian university where Marie Curie worked.",
                seg_anchor,
            ),
            (
                "Nobel Prize",
                "International award for distinction; Marie Curie won twice.",
                nobel_anchor,
            ),
            (
                "Physics",
                "Scientific discipline of matter, energy, and radiation.",
                nobel_anchor,
            ),
        ]
    ]
    mgr.upsert(ExtractionResult(source_doc_id="marie_curie_30s", nodes=nodes, edges=[]))

    time.sleep(2)  # Vespa indexing lag
    return mgr


@pytest.mark.skipif(
    not _graph_manager_available(),
    reason="Vespa backend not reachable — search_nodes test needs the deployed KG",
)
def test_top1_swaps_when_trace_supplied(colbert_encoder, seeded_graph_manager):
    """Without a trace, top-1 for "discover" returns one node; with a
    CoT trace targeting radioactivity research, top-1 swaps. Both pinned."""
    mgr = seeded_graph_manager

    no_trace_hits = mgr.search_nodes(query="discover", top_k=1)
    # ColBERTQueryEncoder.encode now takes ``trace`` but GraphManager's
    # public search API doesn't yet thread it; tests should call the
    # method as the design contract specifies. If the kwarg isn't
    # accepted yet, we surface a clear TypeError so the missing wiring
    # gets fixed rather than silently fall back to a no-trace encoding.
    with_trace_hits = mgr.search_nodes(
        query="discover",
        trace="medical history involving radioactivity research",
        top_k=1,
    )

    result_pair = {
        "no_trace_top1_name": (
            no_trace_hits[0].get("name", "")
            if isinstance(no_trace_hits[0], dict)
            else ""
        )
        if no_trace_hits
        else "",
        "with_trace_top1_name": (
            with_trace_hits[0].get("name", "")
            if isinstance(with_trace_hits[0], dict)
            else ""
        )
        if with_trace_hits
        else "",
    }
    assert result_pair["no_trace_top1_name"] != result_pair["with_trace_top1_name"], (
        result_pair
    )
    assert_golden_json(result_pair, "graph_search_top1_pair.json")
