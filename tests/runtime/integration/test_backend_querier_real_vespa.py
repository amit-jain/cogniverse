"""BackendQuerier grounds synthetic data in real Vespa content.

Real Vespa (managed by the shared_vespa fixture): deploy a video schema,
feed a metadata doc, and assert _query_profile returns the fed content.
"""

from __future__ import annotations

import time
import uuid

import pytest

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    FieldMappingConfig,
)
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.utils.vespa_test_helpers import deploy_tenant_schema

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_query_profile_returns_real_vespa_content(
    shared_vespa, config_manager, schema_loader
):
    tenant = f"bq{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name="video_colpali_smol500_mv_frame",
        config_manager=config_manager,
    )

    http_port = shared_vespa["http_port"]
    vespa_app = make_vespa_app(url="http://localhost", port=http_port)
    feed = vespa_app.feed_data_point(
        schema=schema,
        data_id="vidA_seg_0",
        fields={
            "video_id": "vidA",
            "video_title": "Robots playing soccer",
            "source_url": "http://example.test/vidA",
            "segment_id": 0,
            "segment_description": "two robots play soccer on a field",
            "start_time": 0.0,
            "end_time": 5.0,
        },
    )
    assert feed.is_successful(), feed.json

    backend = VespaBackend(
        backend_config=BackendConfig(
            backend_type="vespa",
            url="http://localhost",
            port=http_port,
            tenant_id=tenant,
        ),
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})

    querier = BackendQuerier(
        backend=backend,
        backend_config=BackendConfig(profiles={}, tenant_id=tenant),
        field_mappings=FieldMappingConfig(),
    )

    # Poll briefly for Vespa to index the fed doc.
    samples = []
    for _ in range(20):
        samples = await querier._query_profile(
            {"schema_name": schema}, sample_size=5, strategy="diverse"
        )
        if samples:
            break
        time.sleep(0.5)

    assert samples, "BackendQuerier returned no grounded samples from real Vespa"
    assert any(s.get("topic") == "Robots playing soccer" for s in samples)
