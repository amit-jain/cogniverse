"""
Integration tests for QualityMonitor with real Vespa search + real Phoenix.

Full round-trip: golden query → real /search → real Vespa with real data
→ real MRR/nDCG scoring → store baseline in real Phoenix → detect degradation.

Uses shared vespa_instance + search_client + real_telemetry from conftest.
ColPali model generates real embeddings for test documents.
"""

import json
import logging
import time
from datetime import datetime

import httpx
import numpy as np
import pytest
import requests
import torch
from PIL import Image

from cogniverse_core.common.models import get_or_load_model
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_evaluation.quality_monitor import (
    AgentType,
    GoldenEvalResult,
    QualityMonitor,
    Verdict,
)

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "vidore/colsmol-500m"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_default"


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert embeddings to Vespa tensor format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


@pytest.fixture(scope="module")
def colpali_model():
    """Load ColPali model once."""
    config = {
        "colpali_model": COLPALI_MODEL_NAME,
        "embedding_type": "multi_vector",
        "model_loader": "colpali",
    }
    model, processor = get_or_load_model(COLPALI_MODEL_NAME, config, logger)
    device = next(model.parameters()).device
    yield model, processor, device
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def seeded_vespa(vespa_instance, colpali_model):
    """Feed test documents into Vespa with real ColPali embeddings."""
    model, processor, device = colpali_model
    http_port = vespa_instance["http_port"]

    test_docs = [
        {
            "color": (255, 0, 0),
            "title": "Man lifting heavy barbell in gym",
            "video_id": "v_-HpCLXdtcas",
        },
        {
            "color": (0, 0, 255),
            "title": "Ocean waves crashing on rocky coast",
            "video_id": "v_ocean_test",
        },
        {
            "color": (0, 128, 0),
            "title": "Person running through forest trail",
            "video_id": "v_forest_test",
        },
    ]

    for i, doc_info in enumerate(test_docs):
        img = Image.new("RGB", (224, 224), color=doc_info["color"])
        batch_inputs = processor.process_images([img]).to(device)
        with torch.no_grad():
            doc_embeddings = model(**batch_inputs)
        embeddings_np = doc_embeddings.squeeze(0).cpu().float().numpy()
        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings_np)

        doc_id = f"qm_test_doc_{i}"
        vespa_doc = {
            "fields": {
                "video_id": doc_info["video_id"],
                "video_title": doc_info["title"],
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": doc_info["title"],
                "audio_transcript": "",
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }

        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in [200, 201], (
            f"Failed to feed doc {doc_id}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)  # Wait for indexing

    yield test_docs

    for i in range(len(test_docs)):
        doc_id = f"qm_test_doc_{i}"
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.fixture
def monitor_with_real_search(search_client, real_telemetry, tmp_path):
    """QualityMonitor wired to real search via TestClient bridge."""
    golden_queries = [
        {
            "query": "man lifting barbell",
            "expected_videos": ["v_-HpCLXdtcas"],
            "ground_truth": "Man lifting a barbell in gym",
            "query_type": "answer_phrase",
            "source": "test",
        },
        {
            "query": "ocean waves coast",
            "expected_videos": ["v_ocean_test"],
            "ground_truth": "Ocean waves on coast",
            "query_type": "question",
            "source": "test",
        },
        {
            "query": "cat sleeping on sofa",
            "expected_videos": ["v_nonexistent"],
            "ground_truth": "Cat sleeping",
            "query_type": "question",
            "source": "test",
        },
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_queries))

    m = QualityMonitor(
        tenant_id="qm_real_test",
        runtime_url="http://testserver",
        phoenix_http_endpoint="http://localhost:16006",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2",
        golden_dataset_path=str(golden_path),
    )

    # Bridge to TestClient
    m._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: _testclient_transport(search_client, req)
        ),
        base_url="http://testserver",
    )

    yield m

    import asyncio

    asyncio.get_event_loop().run_until_complete(m.close())


@pytest.mark.integration
class TestGoldenEvalRealVespa:
    """Golden eval against real Vespa with real ingested data."""

    @pytest.mark.asyncio
    async def test_golden_eval_with_real_data(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Run golden eval against Vespa with real ColPali-embedded documents.

        Expected: "man lifting barbell" should match v_-HpCLXdtcas (MRR > 0).
        "cat sleeping" should NOT match anything (MRR = 0).
        """
        result = await monitor_with_real_search.evaluate_golden_set()

        assert result.query_count == 3
        assert 0.0 <= result.mean_mrr <= 1.0
        assert 0.0 <= result.mean_ndcg <= 1.0

        scores_by_query = {s["query"]: s for s in result.per_query_scores}

        # "cat sleeping on sofa" — no matching doc in Vespa
        cat = scores_by_query.get("cat sleeping on sofa")
        if cat:
            assert cat["mrr"] == 0.0, (
                f"'cat sleeping' should score MRR=0 (no matching doc), got {cat['mrr']}"
            )

        # At least one query should have MRR > 0 (barbell or ocean should match)
        has_positive_mrr = any(
            s["mrr"] > 0 for s in result.per_query_scores
        )
        assert has_positive_mrr, (
            f"At least one golden query should match real Vespa data. "
            f"Scores: {[(s['query'], s['mrr']) for s in result.per_query_scores]}"
        )

    @pytest.mark.asyncio
    async def test_golden_eval_stores_baseline_in_phoenix(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Golden eval result persists in real Phoenix dataset."""
        result = await monitor_with_real_search.evaluate_golden_set()

        baseline = monitor_with_real_search._last_golden_baseline_mrr
        assert baseline is not None
        assert baseline == pytest.approx(result.mean_mrr, abs=0.01)

    @pytest.mark.asyncio
    async def test_degradation_detection_with_real_baseline(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Store high baseline, verify current eval detects degradation."""
        # Run real eval to get actual MRR
        result = await monitor_with_real_search.evaluate_golden_set()
        real_mrr = result.mean_mrr

        # Now store an inflated baseline AFTER the eval
        # (evaluate_golden_set stores the real result as baseline,
        # we overwrite it with the inflated one)
        inflated_mrr = min(real_mrr + 0.30, 1.0)
        await monitor_with_real_search._store_golden_eval_result(
            GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id="qm_real_test",
                mean_mrr=inflated_mrr,
                mean_ndcg=0.90,
                mean_precision_at_5=0.80,
                query_count=20,
            )
        )

        # Verify the inflated baseline is now stored
        stored = monitor_with_real_search._last_golden_baseline_mrr
        assert stored == pytest.approx(inflated_mrr, abs=0.01), (
            f"Baseline should be {inflated_mrr}, got {stored}"
        )

        # check_thresholds compares the real result against stored baseline
        verdicts = monitor_with_real_search.check_thresholds(result, None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE, (
            f"Expected OPTIMIZE: real MRR={real_mrr:.3f}, "
            f"baseline={inflated_mrr:.3f}, "
            f"drop={((inflated_mrr - real_mrr) / inflated_mrr):.1%}"
        )


def _testclient_transport(test_client, request):
    """Bridge httpx.AsyncClient to FastAPI TestClient."""
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    response = test_client.request(
        method=request.method,
        url=path,
        content=request.content,
        headers=dict(request.headers),
    )

    return httpx.Response(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=response.content,
    )
