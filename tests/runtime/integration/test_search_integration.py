"""
Integration tests for search router with real Vespa backend.

Tests verify the full wiring between routers, ConfigManager (VespaConfigStore),
BackendRegistry, SchemaLoader, and real ColPali query encoder with real
documents fed into Vespa.
"""

import json
import logging
import time

import numpy as np
import pytest
import requests
import torch
from PIL import Image

from cogniverse_core.common.models import get_or_load_model
from cogniverse_core.query.encoders import QueryEncoderFactory

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "vidore/colsmol-500m"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_default"


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert (num_patches, 128) float32 embeddings to Vespa tensor format.

    Returns:
        (float_dict, binary_dict) for embedding and embedding_binary fields.
        Float dict: {patch_idx: [128 floats]} for tensor<bfloat16>(patch{}, v[128])
        Binary dict: {patch_idx: [16 int8s]} for tensor<int8>(patch{}, v[16])
    """
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}

    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}

    return float_dict, binary_dict


@pytest.fixture(scope="module")
def colpali_model():
    """Load ColPali model once for the entire module."""
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
def seeded_documents(vespa_instance, colpali_model):
    """Feed real ColPali-embedded documents into Vespa for search tests."""
    model, processor, device = colpali_model

    test_docs = [
        {
            "color": (255, 0, 0),
            "title": "Red sunset landscape",
            "video_id": "sunset_vid",
        },
        {
            "color": (0, 0, 255),
            "title": "Ocean waves coastal scene",
            "video_id": "ocean_vid",
        },
        {
            "color": (0, 128, 0),
            "title": "Forest trail nature walk",
            "video_id": "forest_vid",
        },
    ]

    http_port = vespa_instance["http_port"]

    for i, doc_info in enumerate(test_docs):
        img = Image.new("RGB", (224, 224), color=doc_info["color"])

        batch_inputs = processor.process_images([img]).to(device)
        with torch.no_grad():
            doc_embeddings = model(**batch_inputs)
        embeddings_np = doc_embeddings.squeeze(0).cpu().float().numpy()

        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings_np)

        doc_id = f"search_test_doc_{i}"
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

    time.sleep(5)

    yield test_docs

    for i in range(len(test_docs)):
        doc_id = f"search_test_doc_{i}"
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.requires_vespa
class TestListProfilesIntegration:
    def test_list_profiles_from_vespa_config(self, search_client):
        """GET /search/profiles returns seeded profiles from real VespaConfigStore.

        Profile list includes both system profiles (from configs/config.json)
        and tenant-specific profiles seeded via ConfigManager.add_backend_profile().
        """
        resp = search_client.get("/search/profiles")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "default"
        # Count includes system profiles merged with seeded test profiles
        assert data["count"] >= 2

        profile_names = {p["name"] for p in data["profiles"]}

        assert "test_colpali" in profile_names
        assert "test_videoprism" in profile_names

        profiles_by_name = {p["name"]: p for p in data["profiles"]}
        assert profiles_by_name["test_colpali"]["model"] == COLPALI_MODEL_NAME
        assert profiles_by_name["test_colpali"]["type"] == "video"
        assert profiles_by_name["test_videoprism"]["model"] == "google/videoprism-base"
        assert profiles_by_name["test_videoprism"]["type"] == "video"

    def test_list_profiles_tenant_scoping(self, search_client):
        """GET /search/profiles?tenant_id=tenant_b returns that tenant's profiles.

        tenant_b has its own seeded profile plus system profiles from config.json.
        The key assertion is that default-only profiles (test_colpali, test_videoprism)
        are NOT present for tenant_b.
        """
        resp = search_client.get("/search/profiles?tenant_id=tenant_b")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "tenant_b"
        assert data["count"] >= 1

        profile_names = {p["name"] for p in data["profiles"]}
        assert "tenant_b_profile" in profile_names

        # Profiles seeded exclusively for default tenant should not appear
        assert "test_colpali" not in profile_names
        assert "test_videoprism" not in profile_names


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.requires_vespa
class TestSearchIntegration:
    def test_search_returns_real_results(
        self,
        search_client,
        seeded_documents,
    ):
        """POST /search with real ColPali encoder and real Vespa documents.

        Verifies SearchService works through the full chain:
        ConfigManager -> profile lookup -> real encoder creation -> encode query ->
        VespaSearchBackend -> Vespa query -> ranked results from seeded documents.
        Real Phoenix telemetry captures spans via TelemetryManager singleton.
        """
        resp = search_client.post(
            "/search",
            json={
                "query": "sunset landscape scenery",
                "profile": "test_colpali",
                "strategy": "default",
                "top_k": 5,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "sunset landscape scenery"
        assert data["profile"] == "test_colpali"
        assert isinstance(data["results"], list)
        assert data["results_count"] == len(data["results"])
        assert data["results_count"] > 0, (
            "Real encoder + seeded docs should return results"
        )

    def test_search_stream_returns_real_results(
        self,
        search_client,
        seeded_documents,
    ):
        """POST /search (stream=True) with real ColPali encoder — verifies SSE path.

        Real Phoenix telemetry captures spans via TelemetryManager singleton.
        """
        resp = search_client.post(
            "/search",
            json={
                "query": "ocean waves water",
                "profile": "test_colpali",
                "strategy": "default",
                "top_k": 3,
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) == 2
        assert events[0]["type"] == "status"
        assert events[0]["query"] == "ocean waves water"
        assert events[1]["type"] == "final"
        assert events[1]["data"]["query"] == "ocean waves water"
        assert isinstance(events[1]["data"]["results"], list)
        assert len(events[1]["data"]["results"]) > 0
