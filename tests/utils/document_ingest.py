"""Ingest text documents under a profile made from a shipped template."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from tests.utils.profile_payload import profile_create_payload

TEXT_TEMPLATE = "document_text_semantic"


def ingest_texts(
    client,
    config_manager,
    schema_loader,
    work_dir: Path,
    texts: dict[str, str],
    *,
    profile: str = "notes",
    tenant: str | None = None,
) -> str:
    """Create ``profile`` from the shipped text template for a new tenant
    (or ``tenant``) through ``client`` (anything with httpx's ``get`` and
    ``post`` against the admin routes), then run each of ``texts`` (file name
    to content) through the production ingestion pipeline under it. The
    template's ``colbert_pylate`` service must be served. Returns the tenant.
    """
    tenant = tenant or f"docs{uuid.uuid4().hex[:8]}:main"
    templates = client.get("/admin/profile-templates", params={"tenant_id": tenant})
    assert templates.status_code == 200, templates.text
    template = next(
        t["config"]
        for t in templates.json()["templates"]
        if t["profile_name"] == TEXT_TEMPLATE
    )
    created = client.post(
        "/admin/profiles", json=profile_create_payload(profile, template, tenant)
    )
    assert created.status_code == 201, created.text
    pipeline = VideoIngestionPipeline(
        tenant_id=tenant,
        config_manager=config_manager,
        schema_loader=schema_loader,
        schema_name=profile,
    )
    for name, text in texts.items():
        source = work_dir / tenant.replace(":", "_") / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(text)
        result = asyncio.run(pipeline.process_video_async(source))
        assert result["status"] == "completed", result
    return tenant
