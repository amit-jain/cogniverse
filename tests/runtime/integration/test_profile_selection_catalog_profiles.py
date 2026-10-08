"""Profile-selection derivation resolves every profile the tenant serves.

A tenant that stores only its own profiles still serves the shipped video
profile whose schema registration deployed for it, and search accepts that
profile. The derivation named it as a candidate but looked it up among the
tenant's stored profiles only, so a routing or profile optimization run with a
chosen dataset failed with "profile ... is not configured for tenant". Real
Vespa config store and schema registry.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path

import pytest

from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.optimization_cli import (
    _profile_selection_profile_types,
    _profile_selection_title_fields,
)
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

SHIPPED = json.loads(Path("configs/config.json").read_text())["backend"]["profiles"]
VIDEO = "video_colpali_smol500_mv_frame"
OWN = "cfg20_docs"
TENANT = f"profcat{uuid.uuid4().hex[:8]}:main"


@pytest.fixture(scope="module")
def tenant(shared_vespa):
    BackendRegistry.clear_instances()
    config_manager = make_config_manager(shared_vespa)
    system = config_manager.get_system_config()
    config_manager.set_system_config(
        dataclasses.replace(
            system,
            inference_service_urls={
                SHIPPED[VIDEO]["inference_services"]["embedding"]: "http://colpali",
                SHIPPED["document_text_semantic"]["inference_services"][
                    "embedding"
                ]: "http://colbert",
            },
        )
    )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(OWN, SHIPPED["document_text_semantic"]),
        tenant_id=TENANT,
    )
    for schema in (VIDEO, SHIPPED["document_text_semantic"]["schema_name"]):
        deploy_tenant_schema(
            shared_vespa,
            tenant_id=TENANT,
            base_schema_name=schema,
            config_manager=config_manager,
        )
    yield config_manager
    BackendRegistry.clear_instances()


def test_the_derivation_resolves_the_shipped_profile_beside_the_tenants_own(
    tenant,
):
    candidates = tenant_usable_profile_names(tenant, TENANT)

    title_fields = _profile_selection_title_fields(
        tenant, TENANT, candidates, FilesystemSchemaLoader(Path("configs/schemas"))
    )
    profile_types = _profile_selection_profile_types(tenant, TENANT, candidates)

    assert candidates == [VIDEO, OWN]
    assert tenant.get_backend_profile(VIDEO, tenant_id=TENANT) is None
    assert title_fields == {VIDEO: "video_title", OWN: "document_title"}
    assert profile_types == {VIDEO: "video", OWN: "document"}
