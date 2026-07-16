"""
Unit tests for wiki page dataclasses, slug generation, and WikiManager.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.wiki.wiki_schema import WikiIndex, WikiPage, generate_slug


@pytest.mark.unit
class TestSlugGeneration:
    def test_simple_title(self):
        assert generate_slug("Machine Learning") == "machine_learning"

    def test_special_chars(self):
        assert generate_slug("What's New?") == "whats_new"

    def test_extra_spaces(self):
        assert generate_slug("  hello   world  ") == "hello_world"

    def test_unicode(self):
        # accented chars collapse to ascii equivalents or drop
        result = generate_slug("Café au lait")
        assert result == "cafe_au_lait"

    def test_already_ascii_lowercase(self):
        assert generate_slug("python") == "python"

    def test_numbers_preserved(self):
        assert generate_slug("GPT-4 Overview") == "gpt_4_overview"

    def test_no_leading_trailing_underscores(self):
        result = generate_slug("!hello!")
        assert not result.startswith("_")
        assert not result.endswith("_")


@pytest.mark.unit
class TestWikiPage:
    def test_topic_creation(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="topic",
            title="Machine Learning",
            content="Content here.",
            entities=["ML", "AI"],
            sources=["doc1"],
            cross_references=[],
        )
        assert page.tenant_id == "acme:production"
        assert page.page_type == "topic"
        assert page.slug == "machine_learning"
        assert page.update_count == 1

    def test_topic_doc_id(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="topic",
            title="Machine Learning",
            content="Content here.",
            entities=[],
            sources=[],
            cross_references=[],
        )
        assert page.doc_id == "wiki_topic_acme_production_machine_learning"

    def test_session_creation(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="session",
            title="Session 2024-01-15",
            content="Session notes.",
            entities=[],
            sources=[],
            cross_references=[],
            query="what is ML?",
            agent_used="video_search",
        )
        assert page.query == "what is ML?"
        assert page.agent_used == "video_search"

    def test_session_doc_id_format(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="session",
            title="Session 2024-01-15",
            content="Session notes.",
            entities=[],
            sources=[],
            cross_references=[],
        )
        # Session doc_id uses timestamp digits
        assert page.doc_id.startswith("wiki_session_acme_production_")
        suffix = page.doc_id[len("wiki_session_acme_production_") :]
        assert suffix.isdigit()

    def test_to_vespa_document_structure(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="topic",
            title="Machine Learning",
            content="Content about ML.",
            entities=["ML", "AI"],
            sources=["doc1", "doc2"],
            cross_references=["deep_learning"],
        )
        doc = page.to_vespa_document()
        assert "fields" in doc
        fields = doc["fields"]
        assert fields["doc_id"] == page.doc_id
        assert fields["tenant_id"] == "acme:production"
        assert fields["page_type"] == "topic"
        assert fields["title"] == "Machine Learning"
        assert fields["content"] == "Content about ML."
        assert fields["slug"] == "machine_learning"
        assert fields["update_count"] == 1

    def test_to_vespa_document_lists_serialized(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="topic",
            title="ML",
            content="Content.",
            entities=["ML", "AI"],
            sources=["doc1"],
            cross_references=["dl"],
        )
        doc = page.to_vespa_document()
        fields = doc["fields"]
        # Lists must be JSON strings
        assert isinstance(fields["entities"], str)
        assert isinstance(fields["sources"], str)
        assert isinstance(fields["cross_references"], str)
        assert json.loads(fields["entities"]) == ["ML", "AI"]
        assert json.loads(fields["sources"]) == ["doc1"]
        assert json.loads(fields["cross_references"]) == ["dl"]

    def test_optional_fields_absent_when_none(self):
        page = WikiPage(
            tenant_id="acme:production",
            page_type="topic",
            title="ML",
            content="Content.",
            entities=[],
            sources=[],
            cross_references=[],
        )
        assert page.query is None
        assert page.agent_used is None
        doc = page.to_vespa_document()
        # Optional None fields should not appear in vespa doc
        assert "query" not in doc["fields"]
        assert "agent_used" not in doc["fields"]

    def test_timestamps_present(self):
        page = WikiPage(
            tenant_id="t",
            page_type="topic",
            title="T",
            content="C",
            entities=[],
            sources=[],
            cross_references=[],
        )
        assert page.created_at
        assert page.updated_at


@pytest.mark.unit
class TestWikiIndex:
    def test_creation(self):
        idx = WikiIndex(tenant_id="acme:production")
        assert idx.tenant_id == "acme:production"
        assert idx.page_count == 0
        assert idx.topic_count == 0
        assert idx.session_count == 0

    def test_doc_id(self):
        idx = WikiIndex(tenant_id="acme:production")
        assert idx.doc_id == "wiki_index_acme_production"

    def test_add_topic(self):
        idx = WikiIndex(tenant_id="acme:production")
        idx.add_page("topic", "Machine Learning", "machine_learning", "A summary.")
        assert idx.page_count == 1
        assert idx.topic_count == 1
        assert idx.session_count == 0

    def test_add_session(self):
        idx = WikiIndex(tenant_id="acme:production")
        idx.add_page("session", "Session 2024-01-15", "session_2024_01_15", "Notes.")
        assert idx.page_count == 1
        assert idx.topic_count == 0
        assert idx.session_count == 1

    def test_add_multiple(self):
        idx = WikiIndex(tenant_id="acme:production")
        idx.add_page("topic", "ML", "ml", "ML summary.")
        idx.add_page("topic", "AI", "ai", "AI summary.")
        idx.add_page("session", "Session 1", "session_1", "Notes.")
        assert idx.page_count == 3
        assert idx.topic_count == 2
        assert idx.session_count == 1

    def test_render_markdown_empty(self):
        idx = WikiIndex(tenant_id="acme:production")
        md = idx.render_markdown()
        assert "## Topics" in md
        assert "## Sessions" in md

    def test_render_markdown_with_content(self):
        idx = WikiIndex(tenant_id="acme:production")
        idx.add_page("topic", "Machine Learning", "machine_learning", "All about ML.")
        idx.add_page("session", "Session 2024-01-15", "session_2024_01_15", "Notes.")
        md = idx.render_markdown()
        assert "Machine Learning" in md
        assert "All about ML." in md
        assert "Session 2024-01-15" in md
        assert "Notes." in md


@pytest.mark.unit
class TestWikiManagerSchemaValidation:
    """Guard against the regression where ``main.py`` constructed
    ``schema_name=f"wiki_pages_{tenant_id}"`` without sanitizing the colon
    in tenant_ids like ``"acme:production"``. Every feed call then hit
    Vespa's /document/v1 URL parser with a colon in the schema segment
    and returned 400 "Illegal key-value pair". The constructor now fails
    fast when handed a colon-containing schema_name.
    """

    def test_colon_in_schema_name_rejected_at_construction(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        with pytest.raises(ValueError, match="contains a colon"):
            WikiManager(
                backend=MagicMock(),
                tenant_id="acme:production",
                schema_name="wiki_pages_acme:production",
            )

    def test_sanitized_schema_name_accepted(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        mgr = WikiManager(
            backend=MagicMock(),
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )
        # tenant_id may keep its colon (it's only used for labels and
        # YQL-quoted queries); schema_name is what lands in URLs.
        assert mgr._tenant_id == "acme:production"
        assert mgr._schema_name == "wiki_pages_acme_production"


@pytest.mark.unit
class TestAutoFileThreshold:
    def _make_manager(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        backend = MagicMock()
        return WikiManager(
            backend=backend,
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )

    def test_fires_for_many_entities(self):
        mgr = self._make_manager()
        assert mgr._should_auto_file(["A", "B", "C"], "search_agent", 1) is True

    def test_fires_for_report_agent(self):
        mgr = self._make_manager()
        assert mgr._should_auto_file(["A"], "detailed_report_agent", 1) is True

    def test_fires_for_deep_research(self):
        mgr = self._make_manager()
        assert mgr._should_auto_file(["A"], "deep_research_agent", 1) is True

    def test_fires_for_long_conversation(self):
        mgr = self._make_manager()
        assert mgr._should_auto_file([], "search_agent", 4) is True

    def test_skips_casual_query(self):
        mgr = self._make_manager()
        assert mgr._should_auto_file(["A"], "search_agent", 1) is False


@pytest.mark.unit
class TestSaveSession:
    def _make_manager_with_mocks(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        backend = MagicMock()
        # get_document returns None (no existing topic pages)
        backend.get_document.return_value = None
        mgr = WikiManager(
            backend=backend,
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )
        return mgr, backend

    def test_save_session_feeds_session_and_topics(self):
        mgr, backend = self._make_manager_with_mocks()

        with (
            patch.object(mgr, "_generate_embedding", return_value=[0.1] * 768),
            patch.object(mgr, "_rebuild_index") as mock_rebuild,
            patch.object(mgr, "_feed_page") as mock_feed,
        ):
            session = mgr.save_session(
                query="What is machine learning?",
                response="Machine learning is a subset of AI.",
                entities=["Machine Learning", "AI"],
                agent_name="search_agent",
                sources=["doc1"],
            )

        # 1 session + 2 topics = 3 feed calls
        assert mock_feed.call_count >= 3
        # rebuild called once
        mock_rebuild.assert_called_once()
        # returns a WikiPage
        assert isinstance(session, WikiPage)
        assert session.page_type == "session"
        assert session.query == "What is machine learning?"

    def test_save_session_cross_references_topic_ids(self):
        mgr, backend = self._make_manager_with_mocks()

        with (
            patch.object(mgr, "_generate_embedding", return_value=[0.1] * 768),
            patch.object(mgr, "_rebuild_index"),
            patch.object(mgr, "_feed_page"),
        ):
            session = mgr.save_session(
                query="Tell me about neural networks and deep learning.",
                response="Neural networks are the backbone of deep learning.",
                entities=["Neural Networks", "Deep Learning"],
                agent_name="search_agent",
            )

        # cross_references should contain the topic doc_ids
        assert len(session.cross_references) == 2


@pytest.mark.unit
@pytest.mark.ci_fast
class TestTopicMergeIdempotency:
    """A client retry after a partial-save 500 re-runs the whole topic merge.
    Re-appending the same response duplicated the topic body and double-counted
    update_count; the merge must be idempotent for already-merged content."""

    def _make_manager(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        return WikiManager(
            backend=MagicMock(),
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )

    def _existing(self, content, update_count):
        return SimpleNamespace(
            text_content=content,
            metadata={
                "content": content,
                "sources": "[]",
                "update_count": update_count,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    def test_reappend_of_already_merged_content_is_a_noop(self):
        mgr = self._make_manager()
        response = "Machine learning is a subset of AI."
        # The topic already contains this response (from the first, partial save).
        existing = self._existing(f"Older content.\n\n{response}", update_count=2)

        with (
            patch.object(mgr, "_get_document_http", return_value=existing),
            patch.object(mgr, "_feed_page") as mock_feed,
        ):
            page = mgr._get_or_create_topic(
                entity="Machine Learning", new_content=response, sources=[]
            )

        # Not duplicated, and update_count NOT re-incremented.
        assert page.content.count(response) == 1
        assert page.update_count == 2
        mock_feed.assert_called_once()

    def test_new_content_still_appends_and_increments(self):
        mgr = self._make_manager()
        existing = self._existing("Older content only.", update_count=2)

        with (
            patch.object(mgr, "_get_document_http", return_value=existing),
            patch.object(mgr, "_should_use_rlm_for_merge", return_value=False),
            patch.object(mgr, "_feed_page"),
        ):
            page = mgr._get_or_create_topic(
                entity="Machine Learning",
                new_content="A genuinely new fact.",
                sources=[],
            )

        assert "Older content only." in page.content
        assert "A genuinely new fact." in page.content
        assert page.update_count == 3


@pytest.mark.unit
class TestEmbeddingPromptPrefix:
    """DenseOn prompting is applied client-side: queries must embed with
    ``is_query=True`` (``query:`` prefix) and stored documents with
    ``is_query=False`` (``document:`` prefix). A query embedded with the
    document prefix lands in a different region of the vector space and
    silently degrades search recall.
    """

    def _make_manager(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        backend = MagicMock()
        backend.search.return_value = []
        return WikiManager(
            backend=backend,
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )

    def test_search_embeds_query_with_is_query_true(self):
        import numpy as np

        mgr = self._make_manager()
        embedder = MagicMock()
        embedder.encode.return_value = np.zeros(768, dtype=np.float32)

        with patch(
            "cogniverse_core.common.models.semantic_embedder.get_semantic_embedder",
            return_value=embedder,
        ):
            mgr.search("what is machine learning?", top_k=3)

        embedder.encode.assert_called_once()
        args, kwargs = embedder.encode.call_args
        assert args[0] == "what is machine learning?"
        assert kwargs.get("is_query") is True

    def test_store_embeds_document_with_is_query_false(self):
        import numpy as np

        mgr = self._make_manager()
        embedder = MagicMock()
        embedder.encode.return_value = np.zeros(768, dtype=np.float32)

        with patch(
            "cogniverse_core.common.models.semantic_embedder.get_semantic_embedder",
            return_value=embedder,
        ):
            vec = mgr._generate_embedding("a stored document body")

        embedder.encode.assert_called_once()
        args, kwargs = embedder.encode.call_args
        assert args[0] == "a stored document body"
        assert kwargs.get("is_query") is False
        assert len(vec) == 768

    def test_embedder_outage_raises_instead_of_zero_vector(self):
        """A dead embedder must raise, not return a zero vector — persisting a
        zero embedding poisons the stored page's hybrid ranking permanently."""
        mgr = self._make_manager()

        with patch(
            "cogniverse_core.common.models.semantic_embedder.get_semantic_embedder",
            side_effect=RuntimeError("denseon sidecar unreachable"),
        ):
            with pytest.raises(RuntimeError, match="embedding generation failed"):
                mgr._generate_embedding("a stored document body")


# ---------------------------------------------------------------------------
# Helpers shared by TestWikiLint
# ---------------------------------------------------------------------------


def _make_mock_result(
    doc_id: str,
    title: str,
    content: str,
    page_type: str = "topic",
    updated_at: str = "2026-04-05T00:00:00+00:00",
    cross_references: str = "[]",
) -> dict:
    """Build a wiki page field dict as the page enumeration returns it."""
    return {
        "doc_id": doc_id,
        "title": title,
        "content": content,
        "page_type": page_type,
        "updated_at": updated_at,
        "cross_references": cross_references,
    }


@pytest.mark.unit
class TestWikiLint:
    def _make_manager(self, pages):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        mgr = WikiManager(
            backend=MagicMock(),
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )
        # Lint's classification logic is under test here; the enumeration
        # transport itself is pinned by TestWikiPageEnumeration.
        mgr._list_pages = lambda top_k=500: pages
        return mgr

    def test_detects_empty_pages(self):
        results = [
            _make_mock_result("doc1", "Short", "Too short"),  # 9 chars < 50
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        assert report["issues_found"] >= 1
        empty_ids = [p["doc_id"] for p in report["empty_pages"]]
        assert "doc1" in empty_ids

    def test_detects_stale_pages(self):
        results = [
            _make_mock_result(
                "doc_stale",
                "Old Topic",
                "This is some content that is long enough to pass the empty check.",
                updated_at="2020-01-01T00:00:00+00:00",
            )
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        stale_ids = [p["doc_id"] for p in report["stale_pages"]]
        assert "doc_stale" in stale_ids
        stale_entry = next(
            p for p in report["stale_pages"] if p["doc_id"] == "doc_stale"
        )
        assert stale_entry["days_since_update"] > 30

    def test_detects_orphan_pages(self):
        # A topic page with no session referencing it.
        results = [
            _make_mock_result(
                "wiki_topic_acme_production_machine_learning",
                "Machine Learning",
                "Content about machine learning for testing purposes.",
            )
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        orphan_ids = [p["doc_id"] for p in report["orphan_pages"]]
        assert "wiki_topic_acme_production_machine_learning" in orphan_ids

    def test_referenced_topic_not_orphan(self):
        topic_id = "wiki_topic_acme_production_ml"
        results = [
            _make_mock_result(
                topic_id,
                "ML",
                "Content about machine learning for testing purposes.",
                page_type="topic",
            ),
            _make_mock_result(
                "session_001",
                "Session",
                "Session content here is long enough to pass the check.",
                page_type="session",
                cross_references=f'["{topic_id}"]',
            ),
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        orphan_ids = [p["doc_id"] for p in report["orphan_pages"]]
        assert topic_id not in orphan_ids

    def test_clean_wiki_returns_zero_issues(self):
        # Fresh page, referenced, long content, recent timestamp.
        # updated_at must be relative-to-now so the test doesn't go stale
        # as the calendar advances past lint()'s 30-day threshold.
        from datetime import datetime, timedelta, timezone

        recent_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        topic_id = "wiki_topic_acme_production_healthy"
        results = [
            _make_mock_result(
                topic_id,
                "Healthy Topic",
                "This page has plenty of content and is nicely up to date.",
                page_type="topic",
                updated_at=recent_iso,
            ),
            _make_mock_result(
                "session_001",
                "Session",
                "Session content here is long enough to pass the check.",
                page_type="session",
                cross_references=f'["{topic_id}"]',
            ),
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        assert report["issues_found"] == 0
        assert report["total_pages"] == 2

    def test_total_pages_counts_all_types(self):
        results = [
            _make_mock_result("t1", "Topic", "x" * 60, page_type="topic"),
            _make_mock_result("s1", "Session", "x" * 60, page_type="session"),
        ]
        mgr = self._make_manager(results)
        report = mgr.lint()

        assert report["total_pages"] == 2


@pytest.mark.unit
class TestWikiDelete:
    def _make_manager(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        backend = MagicMock()
        backend._url = "http://localhost"
        backend._port = 8080
        backend.search.return_value = []
        return WikiManager(
            backend=backend,
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )

    def test_delete_routes_through_backend_document_api(self):
        mgr = self._make_manager()
        with patch.object(mgr, "_rebuild_index"):
            mgr.delete_page("wiki_topic_acme_production_ml")

        mgr._backend.delete_document_fields.assert_called_once_with(
            "wiki_topic_acme_production_ml",
            schema_name="wiki_pages_acme_production",
            namespace="wiki_content",
        )

    def test_delete_rebuilds_index(self):
        mgr = self._make_manager()
        with patch.object(mgr, "_rebuild_index") as mock_rebuild:
            mgr.delete_page("wiki_topic_acme_production_ml")

        mock_rebuild.assert_called_once()

    def test_delete_raises_on_vespa_error(self):
        """The backend primitive raises RuntimeError on a non-2xx (status +
        body in the message) — delete_page must propagate it untouched."""
        import pytest

        mgr = self._make_manager()
        mgr._backend.delete_document_fields.side_effect = RuntimeError(
            "Vespa document delete failed for 'nonexistent_doc' (HTTP 404): Not Found"
        )
        with pytest.raises(RuntimeError, match="404"):
            mgr.delete_page("nonexistent_doc")

    def test_delete_wraps_connection_failures_in_runtime_error(self):
        """A non-RuntimeError failure (e.g. pyvespa's own error type on a
        connection drop) is wrapped into the documented RuntimeError contract
        naming the page."""
        import pytest

        mgr = self._make_manager()
        mgr._backend.delete_document_fields.side_effect = ConnectionError(
            "vespa unreachable"
        )
        with pytest.raises(RuntimeError, match="wiki_topic_gone"):
            mgr.delete_page("wiki_topic_gone")


@pytest.mark.unit
def test_get_document_http_treats_empty_fields_as_absent():
    """An empty fields dict from Vespa must read as a missing doc (None), not
    a phantom empty topic that later merges get corrupted into."""
    from unittest.mock import MagicMock

    from cogniverse_agents.wiki.wiki_manager import WikiManager

    backend = MagicMock()
    backend._url = "http://x"
    backend._port = 8080
    backend.search.return_value = []
    mgr = WikiManager(backend=backend, tenant_id="t:t", schema_name="wiki_pages_t")

    backend.get_document_fields.return_value = {}
    assert mgr._get_document_http("wiki_topic_missing") is None

    backend.get_document_fields.return_value = None
    assert mgr._get_document_http("wiki_topic_gone") is None


def _wired_manager(backend=None):
    from cogniverse_agents.wiki.wiki_manager import WikiManager

    if backend is None:
        backend = MagicMock()
    backend._url = "http://x"
    backend._port = 8080
    return WikiManager(backend=backend, tenant_id="t:t", schema_name="wiki_pages_t")


class _PagesResponse:
    """HTTP 200 /search/ body listing wiki pages as unranked children."""

    status_code = 200
    text = "ok"

    def __init__(self, fields_list):
        self._fields = fields_list

    def json(self):
        return {"root": {"children": [{"fields": f} for f in self._fields]}}


_TOPIC = {
    "doc_id": "wiki_topic_t_t_colpali",
    "page_type": "topic",
    "title": "ColPali",
    "slug": "colpali",
    "content": "ColPali is a late-interaction visual document retrieval model.",
    "cross_references": "[]",
    "updated_at": "2099-01-01T00:00:00+00:00",
}
_SESSION = {
    "doc_id": "wiki_session_t_t_abc",
    "page_type": "session",
    "title": "Session — what is colpali",
    "slug": "session_abc",
    "content": "Discussed ColPali retrieval.",
    "cross_references": "[]",
    "updated_at": "2099-01-01T00:00:00+00:00",
}
_INDEX = {
    "doc_id": "wiki_index_t_t",
    "page_type": "index",
    "title": "Wiki Index — t:t",
    "slug": "wiki_index",
    "content": "# Wiki Index — t:t",
    "cross_references": "[]",
    "updated_at": "2099-01-01T00:00:00+00:00",
}


@pytest.mark.unit
class TestWikiPageEnumeration:
    """Index rebuild and lint enumerate pages with an unranked filtered query —
    the previous ranked semantic_search-without-embeddings always failed encoder
    resolution and was swallowed, so the index and lint reported zero pages."""

    def test_rebuild_index_lists_pages_without_embeddings(self, monkeypatch):
        calls = {}

        def fake_post(endpoint, params, timeout=10.0):
            calls["params"] = params
            return _PagesResponse([_TOPIC, _SESSION, _INDEX])

        monkeypatch.setattr(
            "cogniverse_agents.search.vespa_query.vespa_search_post", fake_post
        )
        backend = MagicMock()
        mgr = _wired_manager(backend)
        mgr._rebuild_index()

        assert calls["params"]["ranking"] == "unranked"
        assert not any(k.startswith("input.query") for k in calls["params"])
        assert 'tenant_id contains "t:t"' in calls["params"]["yql"]

        fed = backend.put_document_fields.call_args
        content = fed.args[1]["content"]
        assert "- **ColPali**" in content
        assert "Session — what is colpali" in content
        assert "_No topics yet._" not in content
        # The index document must not list itself as a page.
        assert "- **Wiki Index" not in content

    def test_lint_counts_pages_via_enumeration(self, monkeypatch):
        monkeypatch.setattr(
            "cogniverse_agents.search.vespa_query.vespa_search_post",
            lambda *a, **k: _PagesResponse([_TOPIC, _SESSION, _INDEX]),
        )
        report = _wired_manager().lint()

        assert report["total_pages"] == 3
        # The topic is unreferenced by the session -> exactly one orphan.
        assert [o["doc_id"] for o in report["orphan_pages"]] == [
            "wiki_topic_t_t_colpali"
        ]
        assert report["stale_pages"] == []
        assert report["empty_pages"] == []
        assert report["issues_found"] == 1

    def test_lint_raises_on_outage(self, monkeypatch):
        import requests

        def refuse(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(
            "cogniverse_agents.search.vespa_query.vespa_search_post", refuse
        )
        with pytest.raises(requests.ConnectionError):
            _wired_manager().lint()


@pytest.mark.unit
class TestWikiFailureContract:
    """Wiki reads and writes surface backend failures — a down Vespa previously
    read as saved/empty/healthy (save returned a WikiPage having persisted
    nothing; search -> []; get_topic -> None; lint -> zero-issue report)."""

    def _patch_embedder(self, monkeypatch):
        import numpy as np

        monkeypatch.setattr(
            "cogniverse_core.common.models.semantic_embedder.get_semantic_embedder",
            lambda *a, **k: MagicMock(
                encode=lambda text, is_query=False: np.zeros(768, np.float32)
            ),
        )

    def test_save_session_raises_when_feed_fails(self, monkeypatch):
        self._patch_embedder(monkeypatch)
        backend = MagicMock()
        backend.put_document_fields.side_effect = ConnectionError("refused")
        mgr = _wired_manager(backend)
        with pytest.raises(ConnectionError):
            mgr.save_session(
                query="q", response="r", entities=[], agent_name="search_agent"
            )

    def test_search_raises_on_outage(self, monkeypatch):
        self._patch_embedder(monkeypatch)
        backend = MagicMock()
        backend.search.side_effect = RuntimeError("vespa down")
        mgr = _wired_manager(backend)
        with pytest.raises(RuntimeError, match="vespa down"):
            mgr.search("colpali")

    def test_search_query_dict_has_no_schema_key(self, monkeypatch):
        self._patch_embedder(monkeypatch)
        backend = MagicMock()
        backend.search.return_value = []
        mgr = _wired_manager(backend)
        mgr.search("colpali")
        query_dict = backend.search.call_args.args[0]
        assert "schema" not in query_dict
        assert query_dict["type"] == "wiki"
        assert query_dict["strategy"] == "hybrid"

    def test_get_topic_outage_raises_not_none(self):
        backend = MagicMock()
        backend.get_document_fields.side_effect = ConnectionError("refused")
        mgr = _wired_manager(backend)
        with pytest.raises(ConnectionError):
            mgr.get_topic("colpali")

    def test_get_index_missing_returns_none(self):
        backend = MagicMock()
        backend.get_document_fields.return_value = None
        assert _wired_manager(backend).get_index() is None
