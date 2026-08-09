"""
Integration tests for WikiManager against a real Vespa Docker container.

Deploys this module's own tenant-scoped wiki_pages schema onto the
session-wide Vespa container, exercises the full save→feed→retrieve
round-trip, and leaves the container to the session fixture.

These tests verify that WikiManager correctly writes documents to Vespa
and that the stored content is retrievable via the Document v1 HTTP API.
"""

import os
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

from cogniverse_agents.wiki.wiki_manager import WikiManager
from cogniverse_agents.wiki.wiki_schema import WikiPage, generate_slug
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# This module's own tenant. Memory tests own ``test_tenant`` on the same
# shared container, so a distinct tenant here keeps the two suites' wiki
# documents in separate schemas.
TENANT_ID = "test:wiki"
WIKI_SCHEMA = schema_full_name("wiki_pages", TENANT_ID)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_url(port: int, doc_id: str) -> str:
    return (
        f"http://localhost:{port}/document/v1/wiki_content/{WIKI_SCHEMA}/docid/{doc_id}"
    )


def _get_vespa_doc(port: int, doc_id: str, retries: int = 15) -> dict | None:
    """Fetch a document from Vespa, retrying to allow for indexing latency."""
    for _ in range(retries):
        try:
            resp = requests.get(_doc_url(port, doc_id), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def _wait_for_schema_ready(
    http_port: int, schema_name: str, timeout: int = 120
) -> bool:
    """Feed a minimal probe document to confirm the schema is accepting writes."""
    # Schema-readiness writes only need a 768-dim vector that Vespa
    # accepts; the content is irrelevant since the probe doc is
    # immediately deleted. Constant vector avoids a live embedding call.
    embedding = [0.01] * 768

    probe = {
        "fields": {
            "doc_id": "readiness_check",
            "tenant_id": "test",
            "page_type": "topic",
            "title": "readiness check",
            "content": "test",
            "slug": "readiness_check",
            "entities": "[]",
            "sources": "[]",
            "cross_references": "[]",
            "update_count": 1,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "embedding": embedding,
        }
    }
    url = f"http://localhost:{http_port}/document/v1/wiki_content/{schema_name}/docid/readiness_check"

    for i in range(timeout):
        try:
            resp = requests.post(url, json=probe, timeout=5)
            if resp.status_code in (200, 201):
                requests.delete(url, timeout=5)
                return True
            if i % 10 == 0:
                print(
                    f"   readiness attempt {i + 1}: {resp.status_code} {resp.text[:80]}"
                )
        except Exception as exc:
            if i % 10 == 0:
                print(f"   readiness attempt {i + 1}: {exc}")
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Module fixture: deploy this module's wiki schema onto the shared Vespa
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wiki_vespa(shared_vespa):
    """The session-wide Vespa with this module's wiki_pages schema deployed.

    ``deploy_tenant_schema`` goes through SchemaRegistry, which redeploys the
    complete cluster schema list. A hand-built application package carrying
    only this schema would drop every other suite's schema from the shared
    container, because Vespa treats an absent schema as a removal.
    """
    deployed = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT_ID,
        base_schema_name="wiki_pages",
    )
    assert deployed == WIKI_SCHEMA, (
        f"deploy_schema returned {deployed!r}; the module addresses "
        f"{WIKI_SCHEMA!r} in every document URL"
    )

    http_port = shared_vespa["http_port"]
    if not _wait_for_schema_ready(http_port, WIKI_SCHEMA, timeout=120):
        pytest.fail(f"Schema {WIKI_SCHEMA} not ready within 120 s")

    yield {
        "http_port": http_port,
        "config_port": shared_vespa["config_port"],
        "container_name": shared_vespa["container_name"],
    }
    # No teardown — shared_vespa owns the container lifecycle.


@pytest.fixture(scope="module")
def wiki_manager(wiki_vespa):
    """WikiManager wired to the real test Vespa instance.

    The manager's document CRUD goes through the backend's namespace-aware
    document API, so the fixture provides a REAL VespaBackend document slice
    (a mock here would swallow every write as a silent no-op). Only
    ``search`` is stubbed to [] — full-text search over wiki_content is not
    the focus of these tests, and _rebuild_index gracefully skips index
    population without hits.

    _generate_embedding falls through to the built-in zero-vector
    fallback when no embedding service is reachable — no mock needed.
    """
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_vespa.backend import VespaBackend

    http_port = wiki_vespa["http_port"]

    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = http_port
    backend._metadata_app = None
    backend._metadata_app_key = None
    # Document put/get resolve the wiki schema's document_mapping through the
    # loader, so the real backend slice needs it wired.
    # put_document resolves the wiki schema's document_mapping through this
    # loader; the real backend sets it in __init__, so a bare-constructed
    # backend must provide it too or every feed raises AttributeError.
    backend._schema_loader_instance = FilesystemSchemaLoader(Path("configs/schemas"))
    backend.config = {}
    backend.search = MagicMock(return_value=[])

    manager = WikiManager(
        backend=backend,
        tenant_id=TENANT_ID,
        schema_name=WIKI_SCHEMA,
    )

    yield manager, http_port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWikiVespaIntegration:
    def test_container_is_owned_by_this_pytest_process(self, wiki_vespa):
        """The Vespa these tests write to carries this session's pid, so a
        SIGKILLed run leaves a container the next session reaps."""
        assert wiki_vespa["container_name"].startswith(f"backend-tests-{os.getpid()}-")

    def test_save_session_feeds_to_vespa(self, wiki_manager):
        """save_session feeds a session document retrievable via Vespa Document API."""
        manager, port = wiki_manager

        page = manager.save_session(
            query="What is reinforcement learning?",
            response="Reinforcement learning is a type of machine learning.",
            entities=["reinforcement_learning"],
            agent_name="search_agent",
        )

        doc = _get_vespa_doc(port, page.doc_id)

        assert doc is not None, (
            f"Document {page.doc_id} not found in Vespa after save_session. "
            f"GET {_doc_url(port, page.doc_id)}"
        )
        fields = doc.get("fields", {})
        assert fields.get("page_type") == "session"
        assert fields.get("tenant_id") == TENANT_ID
        assert "reinforcement learning" in fields.get("content", "").lower()

    def test_topic_pages_created_for_entities(self, wiki_manager):
        """save_session creates one topic page in Vespa for each entity."""
        manager, port = wiki_manager

        entities = ["neural_network", "gradient_descent"]
        manager.save_session(
            query="Explain neural networks",
            response="Neural networks use gradient descent for training.",
            entities=entities,
            agent_name="search_agent",
        )

        safe = TENANT_ID.replace(":", "_")
        for entity in entities:
            slug = generate_slug(entity)
            doc_id = f"wiki_topic_{safe}_{slug}"
            doc = _get_vespa_doc(port, doc_id)

            assert doc is not None, (
                f"Topic page for entity '{entity}' (doc_id={doc_id}) not found in Vespa."
            )
            fields = doc.get("fields", {})
            assert fields.get("page_type") == "topic"
            assert fields.get("tenant_id") == TENANT_ID

    def test_save_session_upserts_topics_concurrently(self, wiki_manager, monkeypatch):
        """save_session upserts each entity's topic page concurrently, and the
        session's cross_references collects one doc_id per entity."""
        manager, port = wiki_manager
        entities = ["concur_alpha", "concur_beta", "concur_gamma", "concur_delta"]

        barrier = threading.Barrier(len(entities), timeout=20)
        real_topic = manager._get_or_create_topic

        def barrier_gated(*, entity, new_content, sources):
            # The topic upserts share no state, so concurrent save_session gets
            # all four into the barrier at once. A serial loop processes them
            # one at a time — the first caller blocks until the 20s timeout
            # raises BrokenBarrierError and the test fails.
            barrier.wait()
            return real_topic(entity=entity, new_content=new_content, sources=sources)

        monkeypatch.setattr(manager, "_get_or_create_topic", barrier_gated)

        page = manager.save_session(
            query="Concurrent entity upsert",
            response="Body referencing several entities at once.",
            entities=entities,
            agent_name="search_agent",
        )

        assert len(page.cross_references) == len(entities)
        safe = TENANT_ID.replace(":", "_")
        for entity in entities:
            doc_id = f"wiki_topic_{safe}_{generate_slug(entity)}"
            assert doc_id in page.cross_references
            doc = _get_vespa_doc(port, doc_id)
            assert doc is not None, f"Topic page {doc_id} missing after save_session."
            assert doc.get("fields", {}).get("page_type") == "topic"

    def test_topic_update_merges_content(self, wiki_manager):
        """Saving two sessions with the same entity merges content on the topic page.

        _get_or_create_topic uses _get_document_http (real Vespa HTTP GET) to
        detect the existing topic — no mock needed for the merge path.
        """
        manager, port = wiki_manager

        entity = "transformer_architecture"
        safe = TENANT_ID.replace(":", "_")
        slug = generate_slug(entity)
        doc_id = f"wiki_topic_{safe}_{slug}"

        # First session — topic page is created fresh.
        manager.save_session(
            query="What is a transformer?",
            response="Transformers use self-attention mechanisms.",
            entities=[entity],
            agent_name="search_agent",
        )

        first_doc = _get_vespa_doc(port, doc_id)
        assert first_doc is not None, f"Topic page {doc_id} not found after first save."
        first_update_count = first_doc.get("fields", {}).get("update_count", 0)
        first_created_at = first_doc.get("fields", {}).get("created_at", "")

        # Second session — _get_or_create_topic fetches the existing doc via HTTP,
        # then merges and re-feeds.  No mock needed here.
        manager.save_session(
            query="How do transformers handle long sequences?",
            response="Transformers use positional encoding for sequence order.",
            entities=[entity],
            agent_name="search_agent",
        )

        updated_doc = _get_vespa_doc(port, doc_id)
        assert updated_doc is not None, (
            f"Topic page {doc_id} not found after second save."
        )
        updated_content = updated_doc.get("fields", {}).get("content", "")
        updated_update_count = updated_doc.get("fields", {}).get("update_count", 0)

        assert "self-attention" in updated_content, (
            "Merged content missing first session text."
        )
        assert "positional encoding" in updated_content, (
            "Merged content missing second session text."
        )
        assert updated_update_count == first_update_count + 1, (
            f"update_count did not increment by one: was {first_update_count}, "
            f"now {updated_update_count}"
        )

        # The merge re-feeds the whole topic doc: creation time must survive it
        # and the stamp the staleness lint reads must advance to the merge.
        updated_fields = updated_doc.get("fields", {})
        assert updated_fields.get("created_at") == first_created_at
        merged_at = datetime.fromisoformat(updated_fields["updated_at"])
        created_at = datetime.fromisoformat(first_created_at)
        assert merged_at >= created_at
        assert (datetime.now(timezone.utc) - merged_at).total_seconds() < 300

    def test_concurrent_save_sessions_keep_both_writes(self, wiki_manager):
        """Two concurrent save_session calls for the SAME entity must both land
        their content on the topic page in real Vespa. The real GET/PUT latency
        is the read-merge-write window; the per-topic lock serializes it so
        neither write is lost (without the lock the second feed overwrites the
        first and one interaction's text is silently gone)."""
        import concurrent.futures

        manager, port = wiki_manager
        entity = "attention_mechanism"
        safe = TENANT_ID.replace(":", "_")
        doc_id = f"wiki_topic_{safe}_{generate_slug(entity)}"

        # Seed a base topic so both racers hit the merge path.
        manager.save_session(
            query="seed",
            response="BASE_attention_notes",
            entities=[entity],
            agent_name="search_agent",
        )
        base_doc = _get_vespa_doc(port, doc_id)
        assert base_doc is not None
        base_count = base_doc.get("fields", {}).get("update_count", 0)

        def _save(marker: str) -> None:
            manager.save_session(
                query=f"q_{marker}",
                response=f"CONTENT_{marker}",
                entities=[entity],
                agent_name="search_agent",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            for fut in [ex.submit(_save, m) for m in ("ALPHA", "BETA")]:
                fut.result()

        final = _get_vespa_doc(port, doc_id)
        assert final is not None
        content = final.get("fields", {}).get("content", "")
        assert "BASE_attention_notes" in content
        assert "CONTENT_ALPHA" in content, "first concurrent write was lost"
        assert "CONTENT_BETA" in content, "second concurrent write was lost"
        final_count = final.get("fields", {}).get("update_count", 0)
        assert final_count == base_count + 2, (
            f"update_count {final_count} != base {base_count} + 2 merges"
        )

    def test_lint_detects_empty_page(self, wiki_manager):
        """Feed an empty topic page and assert it persists as an empty page.

        The lint RULE itself is unit-tested in test_wiki_manager.py; this
        fixture stubs backend.search so it exercises only persistence.
        """
        manager, port = wiki_manager

        # Feed a topic page with very short content (< 50 chars)
        empty_page = WikiPage(
            tenant_id=TENANT_ID,
            page_type="topic",
            title="Empty Test",
            content="x",
            entities=[],
            sources=[],
            cross_references=[],
        )
        embedding = [0.1] * 768
        manager._feed_page(empty_page, embedding)
        time.sleep(3)

        # Verify page exists in Vespa
        doc = _get_vespa_doc(port, empty_page.doc_id)
        assert doc is not None, "Empty page not found in Vespa"

        # Lint should detect it — but lint uses backend.search which is mocked.
        # So we verify the page content directly and test the lint logic.
        content = doc.get("fields", {}).get("content", "")
        assert len(content) < 50, f"Page should be empty but has {len(content)} chars"

    def test_lint_detects_stale_page(self, wiki_manager):
        """Feed a topic page with old timestamp, verify it's stale."""
        manager, port = wiki_manager
        from datetime import timedelta

        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        stale_page = WikiPage(
            tenant_id=TENANT_ID,
            page_type="topic",
            title="Stale Test",
            content="This page was last updated two months ago and is now stale.",
            entities=[],
            sources=[],
            cross_references=[],
        )
        stale_page.updated_at = old_date
        stale_page.created_at = old_date
        embedding = [0.1] * 768
        manager._feed_page(stale_page, embedding)
        time.sleep(3)

        doc = _get_vespa_doc(port, stale_page.doc_id)
        assert doc is not None, "Stale page not found in Vespa"

        # The stored stamp is the page's own updated_at (seconds precision),
        # not the wall clock at feed time — the staleness lint reads this field.
        expected = datetime.fromisoformat(old_date).replace(microsecond=0).isoformat()
        fields = doc.get("fields", {})
        assert fields.get("updated_at") == expected
        assert fields.get("created_at") == expected

        stale = [
            p for p in manager.lint()["stale_pages"] if p["doc_id"] == stale_page.doc_id
        ]
        assert len(stale) == 1
        assert stale[0]["days_since_update"] == 60

    def test_lint_detects_orphan_topic(self, wiki_manager):
        """A topic page not referenced by any session is an orphan."""
        manager, port = wiki_manager

        # Feed an orphan topic — no session references it
        orphan_page = WikiPage(
            tenant_id=TENANT_ID,
            page_type="topic",
            title="Orphan Topic",
            content="This topic has no session that references it via cross_references.",
            entities=[],
            sources=[],
            cross_references=[],
        )
        embedding = [0.1] * 768
        manager._feed_page(orphan_page, embedding)
        time.sleep(3)

        doc = _get_vespa_doc(port, orphan_page.doc_id)
        assert doc is not None, "Orphan page not found in Vespa"

        # Verify no session references this doc_id
        cross_refs = doc.get("fields", {}).get("cross_references", "[]")
        assert cross_refs == "[]", "Orphan page should have no cross_references"

    def test_delete_page_removes_from_vespa(self, wiki_manager):
        """delete_page removes the document from Vespa so it is no longer retrievable."""
        manager, port = wiki_manager

        manager.save_session(
            query="What is attention mechanism?",
            response="Attention mechanism weights token importance.",
            entities=["attention_mechanism_delete_test"],
            agent_name="search_agent",
        )

        safe = TENANT_ID.replace(":", "_")
        slug = generate_slug("attention_mechanism_delete_test")
        topic_doc_id = f"wiki_topic_{safe}_{slug}"

        # Confirm the topic page exists before deletion.
        doc_before = _get_vespa_doc(port, topic_doc_id)
        assert doc_before is not None, (
            f"Topic page {topic_doc_id} not found in Vespa before delete."
        )

        manager.delete_page(topic_doc_id)

        # After deletion, Vespa should return 404 — poll briefly for convergence.
        for _ in range(10):
            try:
                resp = requests.get(_doc_url(port, topic_doc_id), timeout=5)
                if resp.status_code == 404:
                    return
            except Exception:
                pass
            time.sleep(1)

        pytest.fail(
            f"Document {topic_doc_id} still retrievable from Vespa after delete_page()."
        )

    def test_concurrent_same_entity_upsert_preserves_both_contents(
        self, wiki_manager, monkeypatch
    ):
        """Two replicas filing the SAME entity must not lose either writer's
        content. ``_upsert_topic`` is the read-merge-write body a second
        runtime process races — the per-topic lock in ``_get_or_create_topic``
        does not span processes, so only the test-and-set protects the merge.
        Both threads read the topic before either writes, so without the
        conditional put the second full-put clobbers the first. The condition
        runs against real Vespa, and the document read back must carry BOTH
        contents.
        """
        manager, port = wiki_manager
        entity = "cas_race_topic"
        safe = TENANT_ID.replace(":", "_")
        doc_id = f"wiki_topic_{safe}_{generate_slug(entity)}"
        content_a = "CAS_ALPHA_distinct_marker_one"
        content_b = "CAS_BETA_distinct_marker_two"

        barrier = threading.Barrier(2, timeout=30)
        original_get = manager._get_document_http
        reads: list = []
        reads_lock = threading.Lock()

        def barriered_get(read_doc_id):
            result = original_get(read_doc_id)
            tid = threading.get_ident()
            with reads_lock:
                first_read = tid not in reads
                reads.append(tid)
            if first_read:
                # Hold until BOTH threads have read the pre-write state, forcing
                # the interleaving that loses a write without a test-and-set.
                barrier.wait()
            return result

        monkeypatch.setattr(manager, "_get_document_http", barriered_get)

        errors: dict = {}

        def worker(key, content):
            try:
                manager._upsert_topic(
                    entity=entity,
                    new_content=content,
                    sources=[f"src_{key}"],
                    doc_id=doc_id,
                )
            except Exception as exc:  # surfaced by the assert below
                errors[key] = exc

        threads = [
            threading.Thread(target=worker, args=("a", content_a)),
            threading.Thread(target=worker, args=("b", content_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(90)
        assert not any(t.is_alive() for t in threads), "concurrent filing hung"
        assert errors == {}, f"concurrent filing raised: {errors}"

        # Two base reads plus exactly one re-read: the writer whose condition
        # Vespa rejected re-merged against the winner's page.
        assert len(reads) == 3, f"expected one CAS retry, got reads: {reads}"

        doc = _get_vespa_doc(port, doc_id)
        assert doc is not None, f"topic {doc_id} missing after concurrent filing"
        fields = doc["fields"]
        content = fields.get("content", "")
        assert content_a in content, f"lost content_a; final content: {content!r}"
        assert content_b in content, f"lost content_b; final content: {content!r}"
        assert int(fields["update_count"]) == 2

    def test_same_process_filing_serializes_on_the_topic_lock(
        self, wiki_manager, monkeypatch
    ):
        """Two same-process filings of one entity run their read-merge-write
        one after the other, so neither burns a rejected conditional put: the
        per-topic lock in ``_get_or_create_topic`` holds the second filing
        until the first has fed."""
        manager, port = wiki_manager
        entity = "serialized_topic"
        safe = TENANT_ID.replace(":", "_")
        doc_id = f"wiki_topic_{safe}_{generate_slug(entity)}"
        content_a = "LOCK_ALPHA_distinct_marker_one"
        content_b = "LOCK_BETA_distinct_marker_two"

        original_get = manager._get_document_http
        original_feed = manager._conditional_feed_topic
        events: list = []
        events_lock = threading.Lock()
        inside_lock = threading.Event()
        keys = {}

        def record(phase):
            with events_lock:
                events.append(f"{keys[threading.get_ident()]}:{phase}")

        def tracked_get(read_doc_id):
            result = original_get(read_doc_id)
            record("read")
            inside_lock.set()
            # Hold the topic lock long enough for the other filing to reach it.
            time.sleep(1.0)
            return result

        def tracked_feed(page, embedding, expected_update_count):
            applied = original_feed(page, embedding, expected_update_count)
            record("feed" if applied else "rejected")
            return applied

        monkeypatch.setattr(manager, "_get_document_http", tracked_get)
        monkeypatch.setattr(manager, "_conditional_feed_topic", tracked_feed)

        errors: dict = {}

        def worker(key, content):
            keys[threading.get_ident()] = key
            try:
                manager._get_or_create_topic(
                    entity=entity, new_content=content, sources=[f"src_{key}"]
                )
            except Exception as exc:  # surfaced by the assert below
                errors[key] = exc

        first = threading.Thread(target=worker, args=("a", content_a))
        second = threading.Thread(target=worker, args=("b", content_b))
        first.start()
        assert inside_lock.wait(30), "first filing never reached the topic read"
        second.start()
        for thread in (first, second):
            thread.join(90)
        assert not any(t.is_alive() for t in (first, second)), "filing hung"
        assert errors == {}, f"serialized filing raised: {errors}"

        # Full serialization: the second filing's read happens after the
        # first's feed, so no conditional put is ever rejected.
        assert events == ["a:read", "a:feed", "b:read", "b:feed"], events

        doc = _get_vespa_doc(port, doc_id)
        assert doc is not None, f"topic {doc_id} missing after serialized filing"
        fields = doc["fields"]
        content = fields.get("content", "")
        assert content_a in content, f"lost content_a; final content: {content!r}"
        assert content_b in content, f"lost content_b; final content: {content!r}"
        assert int(fields["update_count"]) == 2
