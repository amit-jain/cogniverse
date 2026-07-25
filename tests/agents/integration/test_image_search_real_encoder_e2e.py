"""Query-by-image end to end through the real deployed encoder and real Vespa.

This is the whole feature with nothing substituted: two real images are embedded
by the real Tomoro ColQwen3 encoder (served by a real vLLM sidecar, the same
``/pooling`` API production uses) and fed to a real Vespa image_colpali_mv
schema; then a real PNG of one of them is handed to
``ImageSearchAgent.search_by_image`` as raw bytes, encoded by that same
encoder, and matched by MaxSim.

The assertions are the retrieval outcome, not the plumbing: querying with the
first image ranks the first image's document top, querying with the second
inverts the order, and each query scores its own image strictly higher than the
other. A run where the query tensor never reached the rank profile, where the
ingest and query sides disagreed on dimensionality, or where results came back
in insertion order would fail here.

test_image_search_vespa_ranking.py covers the same ranking path with directly
constructed vectors and no sidecar, so ordinary runs stay fast; this module is
the one that proves the deployed 320-d encoder and the schema agree.
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cogniverse_agents.image_search_agent import ImageSearchAgent, ImageSearchDeps
from cogniverse_core.query.encoders import ColPaliFamilyQueryEncoder
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_only,
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
]

BASE_SCHEMA = "image_colpali_mv"
TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"
SCHEMA_PATCH_DIM = 320


def _vertical_split_png() -> bytes:
    """Left half black, right half white."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, 64:, :] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _horizontal_stripes_png() -> bytes:
    """Alternating horizontal bands — different structure, same palette."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for row in range(0, 128, 24):
        arr[row : row + 12, :, :] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def real_encoder(tomoro_inference_url):
    """The production encoder, pointed at a live Tomoro vLLM sidecar."""
    return ColPaliFamilyQueryEncoder(
        model_name=TOMORO_MODEL,
        model_loader="colqwen",
        inference_service_url=tomoro_inference_url,
    )


@pytest.fixture(scope="module")
def searchable_images(shared_memory_vespa, real_encoder):
    """Feed two real images embedded by the real encoder, return a live agent."""
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

    tenant = f"imge2e{uuid.uuid4().hex[:6]}"
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

    # Embed the stored images with the SAME encoder the query side uses — that
    # agreement is what makes MaxSim between query and document meaningful.
    payloads = {
        "img-split": _vertical_split_png(),
        "img-stripes": _horizontal_stripes_png(),
    }
    embeddings = {}
    for image_id, png in payloads.items():
        emb = np.asarray(
            real_encoder.encode_image(Image.open(io.BytesIO(png))), dtype=np.float32
        )
        assert emb.ndim == 2, f"{image_id}: expected (patches, dim), got {emb.shape}"
        assert emb.shape[1] == SCHEMA_PATCH_DIM, (
            f"{image_id}: encoder emitted {emb.shape[1]}-d patches but "
            f"{BASE_SCHEMA} declares v[{SCHEMA_PATCH_DIM}]"
        )
        embeddings[image_id] = emb

    docs = []
    for image_id, emb in embeddings.items():
        doc = Document(
            id=image_id,
            content_type=ContentType.IMAGE,
            content_id=image_id,
            status=ProcessingStatus.COMPLETED,
        )
        doc.add_embedding("embedding", emb, {"type": "float", "raw": True})
        doc.add_metadata("image_id", image_id)
        doc.add_metadata("image_title", image_id.replace("img-", "").title())
        doc.add_metadata("source_url", f"s3://images/{image_id}.png")
        docs.append(doc)

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = backend.ingest_documents(docs, BASE_SCHEMA)
        if result["success_count"] == len(docs):
            break
        time.sleep(2)
    else:
        pytest.fail(f"{BASE_SCHEMA} not feedable within 180s of deploy")

    agent = ImageSearchAgent(
        deps=ImageSearchDeps(
            vespa_endpoint=f"http://localhost:{http_port}",
            tenant_id=tenant,
            colpali_model=TOMORO_MODEL,
        )
    )
    agent._query_encoder = real_encoder

    import asyncio

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        hits = asyncio.run(
            agent._search_vespa(
                query_embedding=embeddings["img-split"],
                query_text="",
                search_mode="semantic",
                limit=10,
            )
        )
        if {"img-split", "img-stripes"} <= {h.image_id for h in hits}:
            break
        time.sleep(2)
    else:
        pytest.fail("fed images did not become searchable within 90s")

    return agent, payloads


def _order(results):
    return [r.image_id for r in results if r.image_id in {"img-split", "img-stripes"}]


class TestQueryByImageEndToEnd:
    @pytest.mark.asyncio
    async def test_query_photo_retrieves_that_photo(self, searchable_images):
        agent, payloads = searchable_images

        results = await agent.search_by_image(payloads["img-split"], limit=10)

        assert _order(results)[:2] == ["img-split", "img-stripes"]
        top = results[0]
        assert top.image_id == "img-split"
        assert top.title == "Split"
        assert top.image_url == "s3://images/img-split.png"
        scores = {r.image_id: r.relevance_score for r in results}
        assert scores["img-split"] > scores["img-stripes"]

    @pytest.mark.asyncio
    async def test_a_different_query_photo_inverts_the_ranking(self, searchable_images):
        """Same corpus, different photo — the order must flip.

        Distinguishes real content matching from results arriving in insertion
        order or every query returning the same document.
        """
        agent, payloads = searchable_images

        split_first = await agent.search_by_image(payloads["img-split"], limit=10)
        stripes_first = await agent.search_by_image(payloads["img-stripes"], limit=10)

        assert _order(split_first)[:2] == ["img-split", "img-stripes"]
        assert _order(stripes_first)[:2] == ["img-stripes", "img-split"]

        split_scores = {r.image_id: r.relevance_score for r in split_first}
        stripes_scores = {r.image_id: r.relevance_score for r in stripes_first}
        assert split_scores["img-split"] > split_scores["img-stripes"]
        assert stripes_scores["img-stripes"] > stripes_scores["img-split"]

    @pytest.mark.asyncio
    async def test_base64_payload_from_a_chat_client_retrieves_the_image(
        self, searchable_images
    ):
        """The bytes a chat gateway sends travel base64-encoded; decoding them
        and searching must return the same document as the raw bytes do."""
        agent, payloads = searchable_images

        encoded = base64.b64encode(payloads["img-stripes"]).decode("ascii")
        results = await agent.search_by_image(base64.b64decode(encoded), limit=10)

        assert results[0].image_id == "img-stripes"
        assert results[0].title == "Stripes"
