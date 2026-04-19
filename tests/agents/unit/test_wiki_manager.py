"""
Unit tests for wiki page dataclasses, slug generation, and WikiManager.
"""

import json
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
        suffix = page.doc_id[len("wiki_session_acme_production_"):]
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
            patch.object(
                mgr, "_generate_embedding", return_value=[0.1] * 768
            ),
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
) -> MagicMock:
    """Build a mock backend search result."""
    result = MagicMock()
    doc = MagicMock()
    doc.text_content = content
    doc.metadata = {
        "doc_id": doc_id,
        "title": title,
        "page_type": page_type,
        "updated_at": updated_at,
        "cross_references": cross_references,
    }
    result.document = doc
    return result


@pytest.mark.unit
class TestWikiLint:
    def _make_manager(self, search_results):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        backend = MagicMock()
        backend.search.return_value = search_results
        return WikiManager(
            backend=backend,
            tenant_id="acme:production",
            schema_name="wiki_pages_acme_production",
        )

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
        stale_entry = next(p for p in report["stale_pages"] if p["doc_id"] == "doc_stale")
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
        topic_id = "wiki_topic_acme_production_healthy"
        results = [
            _make_mock_result(
                topic_id,
                "Healthy Topic",
                "This page has plenty of content and is nicely up to date.",
                page_type="topic",
                updated_at="2026-04-04T00:00:00+00:00",
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

    def test_delete_calls_vespa_http(self):
        mgr = self._make_manager()
        with patch("cogniverse_agents.wiki.wiki_manager.requests.delete") as mock_del:
            mock_del.return_value = MagicMock(ok=True, status_code=200)
            with patch.object(mgr, "_rebuild_index"):
                mgr.delete_page("wiki_topic_acme_production_ml")

        mock_del.assert_called_once()
        call_url = mock_del.call_args[0][0]
        assert "wiki_topic_acme_production_ml" in call_url
        assert "wiki_pages_acme_production" in call_url

    def test_delete_rebuilds_index(self):
        mgr = self._make_manager()
        with patch("cogniverse_agents.wiki.wiki_manager.requests.delete") as mock_del:
            mock_del.return_value = MagicMock(ok=True, status_code=200)
            with patch.object(mgr, "_rebuild_index") as mock_rebuild:
                mgr.delete_page("wiki_topic_acme_production_ml")

        mock_rebuild.assert_called_once()

    def test_delete_raises_on_vespa_error(self):
        import pytest

        mgr = self._make_manager()
        with patch("cogniverse_agents.wiki.wiki_manager.requests.delete") as mock_del:
            mock_del.return_value = MagicMock(
                ok=False, status_code=404, text="Not Found"
            )
            with pytest.raises(RuntimeError, match="404"):
                mgr.delete_page("nonexistent_doc")
