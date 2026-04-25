#!/usr/bin/env python3
"""Backfill ``source_url`` on existing Vespa documents.

Phase 3 of the unified-MediaLocator rollout populates ``source_url`` on every
new document at ingest time. Pre-existing documents (ingested before the field
was wired) have an empty value, which forces visual evaluators down the
warning-logged legacy probe path.

This script walks documents for a tenant + schema, computes a canonical URI
from the document's ``video_id`` and the supplied ``--media-root-uri``, and
issues partial updates via Vespa's HTTP API.

Example:
    uv run python scripts/backfill_source_url.py \\
        --tenant-id acme:prod \\
        --schema video_colpali_smol500_mv_frame \\
        --media-root-uri s3://corpus/

Use ``--dry-run`` to inspect the planned updates without writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_core.common.media import MediaConfig, MediaLocator  # noqa: E402
from cogniverse_foundation.config.utils import (  # noqa: E402
    create_default_config_manager,
    get_config,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, type=str)
    parser.add_argument("--schema", required=True, type=str)
    parser.add_argument(
        "--media-root-uri",
        required=True,
        type=str,
        help="Media root URI prefix used to build source_url from video_id.",
    )
    parser.add_argument(
        "--media-extension",
        default=".mp4",
        type=str,
        help="File extension appended to video_id (default: .mp4).",
    )
    parser.add_argument(
        "--page-size", default=100, type=int, help="Documents fetched per page."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def build_locator(tenant_id: str, app_config: dict[str, Any]) -> MediaLocator:
    media_section = app_config.get("media", {})
    media_config = (
        MediaConfig.from_dict(media_section) if media_section else MediaConfig()
    )
    return MediaLocator(tenant_id=tenant_id, config=media_config)


def canonical_uri_for_video(
    locator: MediaLocator, media_root_uri: str, video_id: str, ext: str
) -> str:
    raw = video_id if video_id.lower().endswith(ext.lower()) else video_id + ext
    if "://" in media_root_uri:
        return media_root_uri.rstrip("/") + "/" + raw.lstrip("/")
    return locator.to_canonical_uri(raw)


def iter_documents(vespa_app, schema: str, page_size: int, limit: int | None):
    fetched = 0
    offset = 0
    while True:
        remaining = None if limit is None else max(0, limit - fetched)
        if remaining == 0:
            return
        page_limit = page_size if remaining is None else min(page_size, remaining)
        yql = (
            f"select documentid, video_id, source_url from {schema} "
            f"where true limit {page_limit} offset {offset}"
        )
        response = vespa_app.query(yql=yql, hits=page_limit)
        children = response.json.get("root", {}).get("children", []) or []
        if not children:
            return
        for hit in children:
            fields = hit.get("fields", {})
            yield fields
        fetched += len(children)
        offset += len(children)


def update_source_url(
    vespa_app, schema: str, document_id: str, source_url: str
) -> bool:
    """Issue a partial update setting only the source_url field."""
    try:
        result = vespa_app.update_data(
            schema=schema,
            data_id=document_id,
            fields={"source_url": source_url},
            create=False,
        )
        return getattr(result, "is_successful", lambda: True)()
    except Exception as exc:
        logger.error("Update failed for %s: %s", document_id, exc)
        return False


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    config_manager = create_default_config_manager()
    app_config = get_config(tenant_id=args.tenant_id, config_manager=config_manager)
    locator = build_locator(args.tenant_id, app_config)

    backend_section = app_config.get("backend", {})
    base_url = backend_section.get("vespa_url", "http://localhost")
    port = backend_section.get("vespa_port", 8080)

    from vespa.application import Vespa

    vespa_app = Vespa(url=f"{base_url}:{port}")

    logger.info(
        "Backfilling source_url on %s for tenant %s (root=%s, dry_run=%s)",
        args.schema,
        args.tenant_id,
        args.media_root_uri,
        args.dry_run,
    )

    inspected = updated = skipped = 0
    for fields in iter_documents(vespa_app, args.schema, args.page_size, args.limit):
        inspected += 1
        existing_source_url = fields.get("source_url")
        if existing_source_url:
            skipped += 1
            continue

        video_id = fields.get("video_id")
        if not video_id:
            skipped += 1
            continue

        document_id = fields.get("documentid", "").rsplit("::", 1)[-1]
        new_source_url = canonical_uri_for_video(
            locator, args.media_root_uri, video_id, args.media_extension
        )
        if args.dry_run:
            logger.info("DRY-RUN %s -> source_url=%s", document_id, new_source_url)
            continue

        if update_source_url(vespa_app, args.schema, document_id, new_source_url):
            updated += 1
            if updated % 50 == 0:
                logger.info("Updated %d documents", updated)
        else:
            skipped += 1

    logger.info(
        "Backfill complete: inspected=%d updated=%d skipped=%d",
        inspected,
        updated,
        skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
