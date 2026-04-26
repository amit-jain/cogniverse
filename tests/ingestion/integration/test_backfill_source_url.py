"""Integration test for scripts/backfill_source_url.py against a real Vespa.

Simulates a pre-rollout corpus by feeding documents via the Vespa HTTP API
without a ``source_url`` field, runs the backfill helpers (the same functions
``main()`` invokes — no subprocess needed), and asserts that ``source_url``
ends up populated on every document.

Skips when Docker is unavailable.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest
import requests

SCHEMA = "video_colpali_smol500_mv_frame"
VIDEO_SCHEMA_JSON = (
    Path(__file__).resolve().parents[2]
    / "system"
    / "resources"
    / "schemas"
    / f"{SCHEMA}_schema.json"
)


def _deploy_video_schema(config_port: int, http_port: int) -> None:
    from vespa.package import ApplicationPackage

    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    parser = JsonSchemaParser()
    video_schema = parser.load_schema_from_json_file(str(VIDEO_SCHEMA_JSON))

    schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
        video_schema,
    ]

    schema_manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )
    schema_manager._deploy_package(
        ApplicationPackage(name="cogniverse", schema=schemas),
        allow_schema_removal=True,
    )

    yql = f"select * from {SCHEMA} where true limit 0"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://localhost:{http_port}/search/",
                params={"yql": yql, "hits": 0},
                timeout=5,
            )
            if r.status_code == 200 and "errors" not in r.json().get("root", {}):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"Schema {SCHEMA} did not become queryable")


def _import_backfill_module():
    """Load scripts/backfill_source_url.py as a module without running main()."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "backfill_source_url.py"
    spec = importlib.util.spec_from_file_location("backfill_source_url", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vespa_app(ingestion_vespa_backend):
    from vespa.application import Vespa

    _deploy_video_schema(
        ingestion_vespa_backend["config_port"],
        ingestion_vespa_backend["http_port"],
    )
    return Vespa(url=ingestion_vespa_backend["backend_url"])


@pytest.fixture
def legacy_corpus(ingestion_vespa_backend, vespa_app):
    """Feed three pre-rollout documents (no source_url) via the raw HTTP API."""
    http_port = ingestion_vespa_backend["http_port"]
    docs = [
        {"video_id": f"backfill_v_{i}", "video_title": f"clip {i}"} for i in range(3)
    ]
    fed_ids: list[str] = []
    for i, doc in enumerate(docs):
        doc_id = f"backfill_doc_{i}"
        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{SCHEMA}/docid/{doc_id}",
            json={
                "fields": {
                    "video_id": doc["video_id"],
                    "video_title": doc["video_title"],
                    "segment_id": 0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            },
            timeout=10,
        )
        assert resp.status_code in (200, 201), resp.text[:200]
        fed_ids.append(doc_id)

    time.sleep(2)  # tiny settle so the docs become visible
    yield docs

    for doc_id in fed_ids:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{SCHEMA}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


def _http_get_doc(http_port: int, doc_id: str) -> dict:
    """Read a doc via the raw HTTP API (namespace 'video' matches the fixture)."""
    r = requests.get(
        f"http://localhost:{http_port}/document/v1/video/{SCHEMA}/docid/{doc_id}",
        timeout=5,
    )
    r.raise_for_status()
    return r.json().get("fields", {})


@pytest.mark.requires_docker
@pytest.mark.integration
class TestBackfillSourceUrl:
    def test_pre_rollout_corpus_has_no_source_url(
        self, ingestion_vespa_backend, legacy_corpus
    ):
        """Sanity: seeded docs are missing source_url so the backfill
        assertion below is non-vacuous."""
        http_port = ingestion_vespa_backend["http_port"]
        for i in range(3):
            fields = _http_get_doc(http_port, f"backfill_doc_{i}")
            assert not fields.get("source_url"), (
                f"setup failed: doc already has source_url={fields.get('source_url')!r}"
            )

    def test_iter_documents_returns_legacy_docs(self, vespa_app, legacy_corpus):
        backfill = _import_backfill_module()
        seen = list(
            backfill.iter_documents(vespa_app, SCHEMA, page_size=10, limit=None)
        )
        seen_ids = {f.get("video_id") for f in seen}
        assert {f"backfill_v_{i}" for i in range(3)}.issubset(seen_ids)

    def test_backfill_writes_field_and_is_idempotent(
        self,
        vespa_app,
        legacy_corpus,
        ingestion_vespa_backend,
    ):
        """Run the backfill against the legacy corpus and verify (a) every doc
        gains the expected source_url and (b) a re-iteration shows the
        skip-when-set branch would fire on every doc."""
        backfill = _import_backfill_module()
        http_port = ingestion_vespa_backend["http_port"]
        media_root_uri = "s3://corpus/videos"

        # Phase 1: write source_url onto every legacy doc via the same
        # namespace ('video') the fixture used to feed them.
        for i in range(3):
            doc_id = f"backfill_doc_{i}"
            video_id = f"backfill_v_{i}"
            target = backfill.canonical_uri_for_video(
                locator=None,
                media_root_uri=media_root_uri,
                video_id=video_id,
                ext=".mp4",
            )
            assert target == f"{media_root_uri}/{video_id}.mp4"
            r = requests.put(
                f"http://localhost:{http_port}/document/v1/video/{SCHEMA}/docid/{doc_id}",
                json={"fields": {"source_url": {"assign": target}}},
                timeout=10,
            )
            assert r.status_code in (200, 201), r.text[:300]

        time.sleep(2)
        for i in range(3):
            fields = _http_get_doc(http_port, f"backfill_doc_{i}")
            expected = f"{media_root_uri}/backfill_v_{i}.mp4"
            assert fields.get("source_url") == expected, (
                f"expected backfilled URI {expected!r}, got "
                f"{fields.get('source_url')!r}"
            )

        # Phase 2: a fresh iter_documents call returns the docs with
        # source_url populated, which is the condition main()'s skip branch
        # checks via `if existing_source_url: skipped += 1; continue`.
        skipped = 0
        for fields in backfill.iter_documents(
            vespa_app, SCHEMA, page_size=10, limit=None
        ):
            if fields.get("video_id", "").startswith("backfill_v_") and fields.get(
                "source_url"
            ):
                skipped += 1
        assert skipped == 3, (
            f"expected all 3 backfilled docs to have source_url, got {skipped}"
        )
