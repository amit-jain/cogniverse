"""Admin PUT /signature_variants actually changes which prompts load.

The admin endpoint writes to ``_signature_variant_overrides`` and
``load_for_request`` reads it, closing the loop end-to-end:

  * ``ArtifactManager.load_for_request`` accepts ``variant_id``
    and qualifies all dataset names through it (so two variants get
    distinct datasets and distinct canary state).
  * ``AgentDispatcher.resolve_artefact_for_request`` reads the
    admin override dict via ``_resolve_signature_variant`` and passes
    the variant_id to ``load_for_request``.

Verifies, against a real Phoenix container:

  * default variant: prompts come from the bare-agent dataset;
  * after admin PUT for ``with_jurisdiction``, the same dispatcher
    call returns prompts from the variant-qualified dataset;
  * back-compat: callers who don't pass variant_id keep loading from
    the bare-agent dataset.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def tenant_id() -> str:
    return f"f32_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def artifact_manager(phoenix_container, tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.fixture
def dispatcher(artifact_manager: ArtifactManager) -> AgentDispatcher:
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="f32_dispatcher", config_manager=cm)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=cm,
        schema_loader=None,
        artifact_manager_factory=lambda _t: artifact_manager,
    )


@pytest.fixture
def admin_client(phoenix_container, monkeypatch) -> TestClient:
    # The signature-variant PUT now persists to the durable artifact store, so
    # point the admin router's factory at the test's real Phoenix container.
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": "admin-sigvar",
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    monkeypatch.setattr(
        admin,
        "_build_artifact_manager",
        lambda key: ArtifactManager(telemetry_provider=provider, tenant_id=key),
    )
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
    return TestClient(app)


@pytest.mark.asyncio
class TestVariantSelectionRoundTrip:
    async def test_default_variant_loads_bare_agent_dataset(
        self,
        dispatcher: AgentDispatcher,
        artifact_manager: ArtifactManager,
        tenant_id: str,
        admin_client: TestClient,
    ):
        admin._reset_admin_overrides_for_tests()
        # Save baseline prompts under the bare agent_type (the default
        # variant key).
        await artifact_manager.save_prompts(
            "search_agent", {"system": "BASELINE_DEFAULT_PROMPT"}
        )
        # No admin PUT — dispatcher should resolve to default variant.
        out = await dispatcher.resolve_artefact_for_request(
            "search_agent", tenant_id, request_seed="req_1"
        )
        assert out is not None
        assert out["variant_id"] == "default"
        assert out["served_from"] == "default"
        assert out["prompts"] == {"system": "BASELINE_DEFAULT_PROMPT"}

    async def test_admin_put_routes_to_variant_specific_dataset(
        self,
        dispatcher: AgentDispatcher,
        artifact_manager: ArtifactManager,
        tenant_id: str,
        admin_client: TestClient,
    ):
        admin._reset_admin_overrides_for_tests()
        # Two distinct datasets: bare for default, variant-qualified for v1.
        await artifact_manager.save_prompts(
            "search_agent", {"system": "BASELINE_DEFAULT_PROMPT"}
        )
        variant_key = ArtifactManager.qualified_agent_key(
            "search_agent", "with_jurisdiction"
        )
        await artifact_manager.save_prompts(
            variant_key, {"system": "VARIANT_JURISDICTION_PROMPT"}
        )

        # Admin PUT: this tenant uses the with_jurisdiction variant for search.
        admin_client.put(
            f"/admin/tenants/{tenant_id}/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )

        # Dispatcher must now load the variant-qualified prompts.
        out = await dispatcher.resolve_artefact_for_request(
            "search_agent", tenant_id, request_seed="req_1"
        )
        assert out is not None
        assert out["variant_id"] == "with_jurisdiction", (
            "dispatcher must resolve the admin-selected variant; got "
            f"variant_id={out['variant_id']!r}"
        )
        assert out["prompts"] == {"system": "VARIANT_JURISDICTION_PROMPT"}, (
            "load_for_request must qualify the dataset name with "
            "variant_id so the variant-specific prompts come back, not "
            "the default ones."
        )

    async def test_other_tenants_unaffected_by_one_tenants_selection(
        self,
        dispatcher: AgentDispatcher,
        artifact_manager: ArtifactManager,
        tenant_id: str,
        admin_client: TestClient,
    ):
        admin._reset_admin_overrides_for_tests()
        await artifact_manager.save_prompts("search_agent", {"system": "BASELINE"})
        await artifact_manager.save_prompts(
            ArtifactManager.qualified_agent_key("search_agent", "vA"),
            {"system": "VARIANT_A"},
        )
        admin_client.put(
            f"/admin/tenants/{tenant_id}/signature_variants/search_agent",
            json={"variant_id": "vA"},
        )

        # Tenant whose variant was set sees vA.
        out_set = await dispatcher.resolve_artefact_for_request(
            "search_agent", tenant_id, request_seed="x"
        )
        assert out_set["prompts"] == {"system": "VARIANT_A"}

        # A different tenant (no admin PUT) still sees the default.
        other_tenant = f"other_{uuid.uuid4().hex[:8]}"
        out_other = await dispatcher.resolve_artefact_for_request(
            "search_agent", other_tenant, request_seed="x"
        )
        assert out_other["variant_id"] == "default"

    async def test_admin_can_change_variant_per_agent_independently(
        self,
        dispatcher: AgentDispatcher,
        artifact_manager: ArtifactManager,
        tenant_id: str,
        admin_client: TestClient,
    ):
        admin._reset_admin_overrides_for_tests()
        # search_agent → vA, summarizer_agent → default
        await artifact_manager.save_prompts(
            ArtifactManager.qualified_agent_key("search_agent", "vA"),
            {"system": "SEARCH_VA"},
        )
        await artifact_manager.save_prompts(
            "summarizer_agent", {"system": "SUMMARIZER_DEFAULT"}
        )

        admin_client.put(
            f"/admin/tenants/{tenant_id}/signature_variants/search_agent",
            json={"variant_id": "vA"},
        )

        out_search = await dispatcher.resolve_artefact_for_request(
            "search_agent", tenant_id, request_seed="x"
        )
        out_summarizer = await dispatcher.resolve_artefact_for_request(
            "summarizer_agent", tenant_id, request_seed="x"
        )

        assert out_search["variant_id"] == "vA"
        assert out_summarizer["variant_id"] == "default", (
            "per-agent variant selections must be independent — setting "
            "search_agent's variant must not leak to summarizer_agent"
        )


class TestResolveHelperBehavior:
    def test_no_admin_dict_yields_default_variant(self):
        admin._reset_admin_overrides_for_tests()
        from cogniverse_agents.optimizer.signature_variants import (
            DEFAULT_VARIANT_ID,
        )

        assert (
            AgentDispatcher._resolve_signature_variant("any_tenant", "search_agent")
            == DEFAULT_VARIANT_ID
        )

    def test_admin_dict_entry_returns_chosen_variant(self):
        admin._reset_admin_overrides_for_tests()
        # Overrides are keyed by the canonical tenant id.
        admin._signature_variant_overrides["acme:acme"] = {
            "search_agent": "with_jurisdiction"
        }
        try:
            assert (
                AgentDispatcher._resolve_signature_variant("acme", "search_agent")
                == "with_jurisdiction"
            )
        finally:
            admin._reset_admin_overrides_for_tests()

    def test_put_and_resolve_agree_across_tenant_forms(self, monkeypatch):
        """A variant set via the admin PUT must resolve in dispatch regardless
        of whether the tenant arrives as simple ('acme') or canonical
        ('acme:acme') form. Pre-fix the PUT stored under the raw path key and
        the resolver looked up a differently-normalised key, so a tenant
        arriving in a different form than the PUT used got the default variant.

        This pins the canonicalization logic, not real persistence, so it uses
        an in-memory artifact-store double (the durable round-trip is covered by
        test_signature_variant_persistence.py).
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        blobs: dict = {}

        class _AM:
            def __init__(self, tenant):
                self._t = tenant

            async def save_blob(self, kind, key, raw):
                blobs[(self._t, kind, key)] = raw

            async def load_blob(self, kind, key):
                return blobs.get((self._t, kind, key))

        monkeypatch.setattr(admin, "_build_artifact_manager", lambda key: _AM(key))

        admin._reset_admin_overrides_for_tests()
        app = FastAPI()
        app.include_router(admin.router, prefix="/admin")
        client = TestClient(app)
        try:
            # Admin selects the variant for the SIMPLE-form tenant.
            r = client.put(
                "/admin/tenants/acme/signature_variants/search_agent",
                json={"variant_id": "with_jurisdiction"},
            )
            assert r.status_code == 200
            # Dispatch resolves for the same tenant in COLON form.
            assert (
                AgentDispatcher._resolve_signature_variant("acme:acme", "search_agent")
                == "with_jurisdiction"
            )
            # ...and in simple form.
            assert (
                AgentDispatcher._resolve_signature_variant("acme", "search_agent")
                == "with_jurisdiction"
            )
        finally:
            admin._reset_admin_overrides_for_tests()
