"""export_embeddings filters real Vespa content and escapes filter values.

Real Vespa (shared_vespa fixture): two docs are fed under the production
``content`` namespace; an export filtered on a video_id containing a double
quote must return only that doc — exercising the Document-v1 selection
escaping and proving the filter is applied (not dropped).
"""

from __future__ import annotations

import time
import uuid

import pytest

from cogniverse_foundation.config.utils import get_config
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import deploy_tenant_schema

pytestmark = pytest.mark.integration


def test_export_embeddings_filter_is_applied_and_escaped(
    shared_vespa, config_manager, schema_loader
):
    tenant = f"ex{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name="video_colpali_smol500_mv_frame",
        config_manager=config_manager,
    )

    http_port = shared_vespa["http_port"]
    vespa_app = make_vespa_app(url="http://localhost", port=http_port)
    target_id = 'v"a'  # embedded quote must be escaped in the selection
    for data_id, video_id, title in [
        ("doc_target", target_id, "Target"),
        ("doc_other", "other", "Other"),
    ]:
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=data_id,
            namespace="content",
            fields={
                "video_id": video_id,
                "video_title": title,
                "source_url": f"http://example.test/{data_id}",
                "segment_id": 0,
            },
        )
        assert feed.is_successful(), feed.json

    cfg = get_config(tenant_id=tenant, config_manager=config_manager)
    backend = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": http_port,
            "profiles": cfg.get("backend", {}).get("profiles", {}),
            "default_profiles": cfg.get("backend", {}).get("default_profiles", {}),
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    # Poll until the visit returns the filtered doc.
    docs = []
    for _ in range(20):
        docs = backend.export_embeddings(
            schema=schema, filters={"video_id": target_id}, include_embeddings=False
        )
        if docs:
            break
        time.sleep(0.5)

    ids = [d.get("video_id") for d in docs]
    assert ids == [target_id], f"filter not applied/escaped; got {ids}"
