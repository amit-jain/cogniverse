"""A tenant's stored profile keeps the shipped result granularity, against real Vespa.

``POST /admin/profiles`` stores the tenant's copy of a shipped profile with the
fields its request model carries, which do not include ``result_granularity``.
Searching that profile through ``SearchService`` must still answer one hit per
source, with every matching window of the source in ``matched_segments``: the
tenant's stored profile overrides the keys it sets and nothing else.
"""

import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from cogniverse_agents.search.service import SearchService
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import get_config
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

PROFILE = "document_text_semantic"
BASE_SCHEMA = "document_text"
WINDOWS = {
    "orchardsource": [
        "orchard tractors harvest the apple orchard at dawn",
        "the orchard crew loads orchard crates onto trailers",
        "every orchard row is pruned before the orchard frost",
    ],
    "harboursource": [
        "harbour cranes unload containers along the quay",
        "a single orchard mural decorates the harbour office",
        "tugboats guide freighters through the harbour mouth",
    ],
}


def _documents(source_id: str, windows: list[str]) -> list[Document]:
    documents = []
    offset = 0
    for index, text in enumerate(windows):
        document = Document(
            id=f"{source_id}_{source_id}_w{index:04d}",
            content_type=ContentType.DOCUMENT,
            content_id=source_id,
            status=ProcessingStatus.COMPLETED,
        )
        document.add_embedding(
            "embedding",
            np.full((3, 128), 0.1 * (index + 1), dtype=np.float32),
            {"type": "float", "raw": True},
        )
        document.add_metadata("document_id", source_id)
        document.add_metadata("document_title", f"{source_id}.md")
        document.add_metadata("document_type", "markdown")
        document.add_metadata("document_path", f"/corpus/{source_id}.md")
        document.add_metadata("full_text", text)
        document.add_metadata("page_count", 1)
        document.add_metadata("chunk_index", index)
        document.add_metadata("chunk_count", len(windows))
        document.add_metadata("chunk_start", offset)
        document.add_metadata("chunk_end", offset + len(text))
        offset += len(text)
        documents.append(document)
    return documents


def _profile_as_the_admin_route_stores_it(
    shipped: dict, **overrides
) -> BackendProfileConfig:
    """The BackendProfileConfig ``create_profile`` builds from its request."""
    return BackendProfileConfig(
        profile_name=PROFILE,
        type=shipped["type"],
        description=shipped["description"],
        schema_name=shipped["schema_name"],
        embedding_model=shipped["embedding_model"],
        pipeline_config=shipped["pipeline_config"],
        strategies=shipped["strategies"],
        embedding_type=shipped["embedding_type"],
        schema_config=shipped["schema_config"],
        model_specific=None,
        **overrides,
    )


@pytest.fixture(scope="module")
def seeded_tenants(vespa_instance):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            inference_service_urls={"colbert_pylate": "http://127.0.0.1:9"},
        )
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    suffix = uuid.uuid4().hex[:8]
    inherited = f"granularity{suffix}"
    overriding = f"segmented{suffix}"

    backends = []
    for tenant_id in (inherited, overriding):
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id=tenant_id,
            config={
                "backend": {
                    "url": "http://localhost",
                    "config_port": vespa_instance["config_port"],
                    "port": vespa_instance["http_port"],
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        for source_id, windows in WINDOWS.items():
            outcome = backend.ingest_documents(
                _documents(source_id, windows), BASE_SCHEMA
            )
            assert outcome == {
                "success_count": 3,
                "failed_count": 0,
                "failed_documents": [],
                "total_documents": 3,
            }
        backends.append(backend)

    shipped = get_config(tenant_id=inherited, config_manager=config_manager).get(
        "backend"
    )["profiles"][PROFILE]
    assert shipped["result_granularity"] == "source"
    config_manager.add_backend_profile(
        _profile_as_the_admin_route_stores_it(shipped), tenant_id=inherited
    )
    config_manager.add_backend_profile(
        _profile_as_the_admin_route_stores_it(
            shipped, extra_config={"result_granularity": "segment"}
        ),
        tenant_id=overriding,
    )
    try:
        yield config_manager, schema_loader, inherited, overriding
    finally:
        for backend, tenant_id in zip(backends, (inherited, overriding)):
            backend.schema_manager.delete_schema(tenant_id, BASE_SCHEMA)


def _search(config_manager, schema_loader, tenant_id):
    service = SearchService(
        config=get_config(tenant_id=tenant_id, config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    return service.search(
        query="orchard",
        profile=PROFILE,
        tenant_id=tenant_id,
        top_k=6,
        ranking_strategy="bm25_only",
    )


def _collapsed(results) -> list[tuple[str, set[str]]]:
    return [
        (
            hit.document.metadata["source_id"],
            {segment["document_id"] for segment in hit.matched_segments},
        )
        for hit in results
    ]


EXPECTED_SOURCES = [
    (
        "orchardsource",
        {
            "orchardsource_orchardsource_w0000",
            "orchardsource_orchardsource_w0001",
            "orchardsource_orchardsource_w0002",
        },
    ),
    ("harboursource", {"harboursource_harboursource_w0001"}),
]
EXPECTED_WINDOWS = {
    "orchardsource_orchardsource_w0000",
    "orchardsource_orchardsource_w0001",
    "orchardsource_orchardsource_w0002",
    "harboursource_harboursource_w0001",
}


@pytest.mark.integration
class TestStoredTenantProfileKeepsSourceGranularity:
    def test_a_stored_tenant_profile_answers_one_hit_per_source(self, seeded_tenants):
        config_manager, schema_loader, inherited, _ = seeded_tenants

        results = _search(config_manager, schema_loader, inherited)

        assert results.result_granularity == "source"
        assert _collapsed(results) == EXPECTED_SOURCES
        assert [hit.segments_in_window for hit in results] == [3, 1]

    def test_a_tenant_that_sets_segment_granularity_keeps_it(self, seeded_tenants):
        config_manager, schema_loader, _, overriding = seeded_tenants

        results = _search(config_manager, schema_loader, overriding)

        assert results.result_granularity == "segment"
        assert {hit.document.id for hit in results} == EXPECTED_WINDOWS

    def test_concurrent_tenants_each_search_with_their_own_profile(
        self, seeded_tenants
    ):
        config_manager, schema_loader, inherited, overriding = seeded_tenants
        rounds = 4
        tenants = [inherited, overriding] * rounds
        barrier = threading.Barrier(len(tenants))
        outcomes: dict[int, object] = {}

        def search(slot: int, tenant_id: str) -> None:
            barrier.wait()
            try:
                results = _search(config_manager, schema_loader, tenant_id)
                if tenant_id == inherited:
                    outcomes[slot] = (results.result_granularity, _collapsed(results))
                else:
                    outcomes[slot] = (
                        results.result_granularity,
                        {hit.document.id for hit in results},
                    )
            except BaseException as exc:
                outcomes[slot] = exc

        threads = [
            threading.Thread(target=search, args=(slot, tenant_id))
            for slot, tenant_id in enumerate(tenants)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

        assert outcomes == {
            slot: (
                ("source", EXPECTED_SOURCES)
                if tenant_id == inherited
                else ("segment", EXPECTED_WINDOWS)
            )
            for slot, tenant_id in enumerate(tenants)
        }

    def test_an_unreadable_profile_store_fails_the_search(
        self, seeded_tenants, vespa_instance
    ):
        """The tenant's stored profile cannot be read: the search raises
        naming the profile and tenant instead of answering from the
        service's snapshot."""
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore

        config_manager, schema_loader, inherited, _ = seeded_tenants
        service = SearchService(
            config=get_config(tenant_id=inherited, config_manager=config_manager),
            config_manager=ConfigManager(
                store=VespaConfigStore(backend_url="http://127.0.0.1", backend_port=9)
            ),
            schema_loader=schema_loader,
        )

        with pytest.raises(RuntimeError) as failed:
            service.search(
                query="orchard",
                profile=PROFILE,
                tenant_id=inherited,
                top_k=6,
                ranking_strategy="bm25_only",
            )

        assert str(failed.value).startswith(
            f"Reading backend profile '{PROFILE}' for tenant '{inherited}' failed: "
        ), str(failed.value)
