"""Real-Vespa ranking for image-to-image search.

Query-by-image is the feature: a user sends a photo and gets back the stored
images that actually look like it. This drives ImageSearchAgent against a live
Vespa with a deployed image_colpali_mv schema and two stored images whose
embeddings point in different directions, then asserts the ranking RESPONDS to
the query — querying with image alpha's vector puts alpha first and beta
second, and querying with beta's vector INVERTS that order.

The inversion is the point. A test that only asserts "alpha came back first"
still passes if ranking is broken and results always arrive in insertion order,
or if the query tensor never reaches the rank profile at all. Asserting the
order flips when the query changes proves MaxSim actually scored the query
against the stored patches.

The encoder is covered separately (test_colpali_encode_image_real.py) with a
real ColPali model: the deployed image encoder is Tomoro ColQwen3, which is
remote-only and needs a GPU vLLM sidecar, so the query vectors here are built
directly rather than encoded. Everything below the encoder — YQL, model
restrict, query-tensor serialization, the rank profile, MaxSim scoring and
result parsing — is the real thing against real Vespa.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from cogniverse_agents.image_search_agent import ImageSearchAgent, ImageSearchDeps
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

pytestmark = pytest.mark.integration

BASE_SCHEMA = "image_colpali_mv"
PATCH_DIM = 320  # image_colpali_mv: tensor<bfloat16>(patch{}, v[320])
ALPHA_AXIS = 0
BETA_AXIS = 100
# The readiness probe lives on its own axis so it scores ~0 against both query
# images and never competes for a rank the assertions below check.
PROBE_AXIS = 200


def _axis_embedding(axis: int, patches: int = 4) -> np.ndarray:
    """Multi-vector where every patch is the unit vector along one axis.

    Two documents built on different axes score ~0 against each other under
    MaxSim, so the expected ranking is unambiguous.
    """
    emb = np.zeros((patches, PATCH_DIM), dtype=np.float32)
    emb[:, axis] = 1.0
    return emb


def _image_doc(image_id: str, title: str, axis: int) -> Document:
    doc = Document(
        id=image_id,
        content_type=ContentType.IMAGE,
        content_id=image_id,
        status=ProcessingStatus.COMPLETED,
    )
    doc.add_embedding(
        "embedding", _axis_embedding(axis), {"type": "float", "raw": True}
    )
    doc.add_metadata("image_id", image_id)
    doc.add_metadata("image_title", title)
    doc.add_metadata("source_url", f"s3://images/{image_id}.jpg")
    doc.add_metadata("image_description", f"{title} description")
    return doc


@pytest.fixture(scope="module")
def image_backend(shared_memory_vespa):
    from pathlib import Path

    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    http_port = shared_memory_vespa["http_port"]
    config_port = shared_memory_vespa["config_port"]

    store = VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=http_port)
    )

    tenant = f"imgrank{uuid.uuid4().hex[:6]}"
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": config_port,
                "port": http_port,
            }
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    backend.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name=BASE_SCHEMA
    )

    # prepareandactivate returns before content nodes activate the schema.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = backend.ingest_documents(
            [_image_doc("__ready__", "probe", PROBE_AXIS)], BASE_SCHEMA
        )
        if result["success_count"] == 1:
            break
        time.sleep(2)
    else:
        pytest.fail(f"{BASE_SCHEMA} not feedable within 180s of deploy")

    fed = backend.ingest_documents(
        [
            _image_doc("img-alpha", "Alpha Photo", ALPHA_AXIS),
            _image_doc("img-beta", "Beta Photo", BETA_AXIS),
        ],
        BASE_SCHEMA,
    )
    assert fed["success_count"] == 2, fed
    assert fed["failed_count"] == 0, fed

    agent = ImageSearchAgent(
        deps=ImageSearchDeps(
            vespa_endpoint=f"http://localhost:{http_port}",
            tenant_id=tenant,
        )
    )

    # Documents are searchable only once indexed; poll the real query path.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        import asyncio

        hits = asyncio.run(
            agent._search_vespa(
                query_embedding=_axis_embedding(ALPHA_AXIS),
                query_text="",
                search_mode="semantic",
                limit=10,
            )
        )
        if {"img-alpha", "img-beta"} <= {h.image_id for h in hits}:
            break
        time.sleep(2)
    else:
        pytest.fail("fed images did not become searchable within 60s")

    return agent


class TestImageToImageRanking:
    @pytest.mark.asyncio
    async def test_query_image_ranks_its_own_match_first(self, image_backend):
        agent = image_backend

        results = await agent._search_vespa(
            query_embedding=_axis_embedding(ALPHA_AXIS),
            query_text="",
            search_mode="semantic",
            limit=10,
        )

        ranked = [
            r.image_id for r in results if r.image_id in {"img-alpha", "img-beta"}
        ]
        assert ranked[:2] == ["img-alpha", "img-beta"], ranked
        top = results[0]
        assert top.image_id == "img-alpha"
        assert top.title == "Alpha Photo"
        assert top.image_url == "s3://images/img-alpha.jpg"
        assert top.description == "Alpha Photo description"
        # The matching image must outscore the unrelated one, not merely appear.
        by_id = {r.image_id: r.relevance_score for r in results}
        assert by_id["img-alpha"] > by_id["img-beta"]

    @pytest.mark.asyncio
    async def test_ranking_inverts_when_the_query_image_changes(self, image_backend):
        """Same corpus, different query image — the order must flip.

        This is what distinguishes real ranking from results arriving in
        insertion order or the query tensor being ignored.
        """
        agent = image_backend

        alpha_first = await agent._search_vespa(
            query_embedding=_axis_embedding(ALPHA_AXIS),
            query_text="",
            search_mode="semantic",
            limit=10,
        )
        beta_first = await agent._search_vespa(
            query_embedding=_axis_embedding(BETA_AXIS),
            query_text="",
            search_mode="semantic",
            limit=10,
        )

        def ordering(results):
            return [
                r.image_id for r in results if r.image_id in {"img-alpha", "img-beta"}
            ][:2]

        assert ordering(alpha_first) == ["img-alpha", "img-beta"]
        assert ordering(beta_first) == ["img-beta", "img-alpha"]

        alpha_scores = {r.image_id: r.relevance_score for r in alpha_first}
        beta_scores = {r.image_id: r.relevance_score for r in beta_first}
        assert alpha_scores["img-alpha"] > alpha_scores["img-beta"]
        assert beta_scores["img-beta"] > beta_scores["img-alpha"]


class TestSearchByImageEndToEnd:
    @pytest.mark.asyncio
    async def test_search_by_image_bytes_returns_the_matching_image(
        self, image_backend, monkeypatch
    ):
        """search_by_image decodes real PNG bytes and returns the image whose
        stored embedding matches, through the real Vespa ranking path.

        The query image's bytes are decoded for real; only the encode step is
        substituted (the deployed 320-d encoder needs a GPU sidecar), and it is
        substituted with the SAME vector the stored alpha document holds, so the
        assertion below is a real end-to-end retrieval result.
        """
        import base64
        import io

        from PIL import Image

        agent = image_backend
        seen = {}

        def encode_from_real_image(image):
            seen["size"] = image.size
            seen["mode"] = image.mode
            return _axis_embedding(ALPHA_AXIS)

        monkeypatch.setattr(agent, "_encode_image", encode_from_real_image)

        buf = io.BytesIO()
        Image.new("RGB", (73, 41), color=(200, 30, 30)).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        results = await agent.search_by_image(base64.b64decode(png_b64), limit=10)

        # The real bytes were decoded to the real image before encoding.
        assert seen["size"] == (73, 41)
        assert seen["mode"] == "RGB"
        # And the retrieval result is the matching stored image, ranked first.
        assert results[0].image_id == "img-alpha"
        assert results[0].title == "Alpha Photo"
        assert [r.image_id for r in results if r.image_id == "img-beta"] == ["img-beta"]
        by_id = {r.image_id: r.relevance_score for r in results}
        assert by_id["img-alpha"] > by_id["img-beta"]
