"""Real round-trip for document_visual ingestion: PDF pages → ColPali → Vespa → search.

Renders a 2-page PDF through the real DocumentVisualSegmentationStrategy dispatch,
embeds each page with real ColPali via the production EmbeddingGeneratorImpl path,
feeds the tenant-scoped document_visual schema through VespaPyClient, then asserts
DocumentAgent._search_visual retrieves the fed pages. Exercises the full
segmentation → embedding → feed → search wiring end to end.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from cogniverse_agents.document_agent import DocumentAgent
from cogniverse_core.query.encoders import ColPaliQueryEncoder
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (
    EmbeddingGeneratorImpl,
)
from cogniverse_runtime.ingestion.strategies import (
    DocumentVisualEmbeddingStrategy,
    DocumentVisualSegmentationStrategy,
    NoDescriptionStrategy,
    NoTranscriptionStrategy,
)
from cogniverse_vespa.ingestion_client import VespaPyClient
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
]

TENANT = "docvis_ingest_rt"
COLPALI_MODEL = "vidore/colsmol-500m"
SCHEMAS_DIR = Path("configs/schemas")


class _BackendAdapter:
    """Adapt VespaPyClient to the ingest_documents() interface the generator calls."""

    def __init__(self, client: VespaPyClient):
        self._client = client

    def ingest_documents(self, documents, schema_name):
        prepared = [self._client.process(doc) for doc in documents]
        success, failed = self._client._feed_prepared_batch(prepared)
        return {
            "success_count": success,
            "failed_count": len(failed),
            "failed_documents": failed,
            "total_documents": len(documents),
        }


@pytest.fixture(scope="module")
def two_page_pdf(tmp_path_factory):
    from PIL import Image

    pdf = tmp_path_factory.mktemp("docvis_pdf") / "manual.pdf"
    page1 = Image.new("RGB", (320, 440), color=(250, 250, 250))
    page2 = Image.new("RGB", (320, 440), color=(170, 170, 170))
    page1.save(pdf, format="PDF", save_all=True, append_images=[page2])
    return pdf


@pytest.fixture(scope="module")
def ingested(shared_vespa, two_page_pdf, tmp_path_factory):
    full = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT, base_schema_name="document_visual"
    )
    http_port = shared_vespa["http_port"]
    out_dir = tmp_path_factory.mktemp("docvis_out")

    seg = DocumentVisualSegmentationStrategy(max_files=10, dpi=72)
    strategy_set = ProcessingStrategySet(
        segmentation=seg,
        transcription=NoTranscriptionStrategy(),
        description=NoDescriptionStrategy(),
        embedding=DocumentVisualEmbeddingStrategy(),
    )

    class Ctx:
        profile_output_dir = out_dir
        logger = type(
            "L",
            (),
            {
                "info": staticmethod(lambda m: None),
                "warning": staticmethod(lambda m: None),
                "error": staticmethod(lambda m: None),
            },
        )()

    seg_result = asyncio.run(
        strategy_set._process_segmentation(seg, two_page_pdf, None, Ctx())
    )
    pages = seg_result["document_pages"]
    assert len(pages) == 2

    # The strategy-aware field mapping reads the gitignored ranking_strategies
    # cache, which only auto-generates when absent — refresh it so document_visual
    # (added after any pre-existing cache) is present.
    from cogniverse_vespa.ranking_strategy_extractor import (
        extract_all_ranking_strategies,
        save_ranking_strategies,
    )

    save_ranking_strategies(
        extract_all_ranking_strategies(SCHEMAS_DIR),
        SCHEMAS_DIR / "ranking_strategies.json",
    )

    client = VespaPyClient(
        config={
            "schema_name": full,
            "base_schema_name": "document_visual",
            "url": "http://localhost",
            "port": http_port,
            "schema_loader": FilesystemSchemaLoader(SCHEMAS_DIR),
        }
    )
    client.connect()

    generator = EmbeddingGeneratorImpl(
        config={
            "embedding_model": COLPALI_MODEL,
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
            "schema_name": full,
        },
        backend_client=_BackendAdapter(client),
    )
    result = generator.generate_embeddings(
        {"video_id": "manual", "document_pages": pages}, output_dir=out_dir
    )

    time.sleep(3)
    return {"full": full, "http_port": http_port, "result": result, "pages": pages}


def test_schema_name_matches_agent_query_target(ingested):
    assert ingested["full"] == schema_full_name("document_visual", TENANT)


def test_all_pages_embedded_and_fed(ingested):
    result = ingested["result"]
    assert result.errors == [], result.errors
    assert result.documents_processed == 2
    assert result.documents_fed == 2


@pytest.mark.asyncio
async def test_search_visual_retrieves_ingested_pages(ingested):
    agent = DocumentAgent.__new__(DocumentAgent)
    agent._vespa_endpoint = f"http://localhost:{ingested['http_port']}"
    agent._tenant_id = TENANT
    agent._query_encoder = ColPaliQueryEncoder(model_name=COLPALI_MODEL)

    results = await agent._search_visual("a page from the product manual", limit=10)

    assert results, "real document_visual search returned no results for fed pages"
    assert {r.document_id for r in results} == {"manual"}
    assert all(r.strategy_used == "visual" for r in results)
    assert all(r.document_url.endswith("manual.pdf") for r in results)
    assert all(r.relevance_score > 0 for r in results)
