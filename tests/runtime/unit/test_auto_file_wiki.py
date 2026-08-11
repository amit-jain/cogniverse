"""The dispatcher's wiki auto-file hook runs after every successful dispatch but
had zero coverage of its body — a signature drift on the private
_should_auto_file / save_session calls (or the factory shape) would silently
kill wiki auto-filing on every dispatch, logging only a warning. These exercise
the real body.
"""

import logging
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _dispatcher():
    return object.__new__(AgentDispatcher)


@pytest.mark.asyncio
async def test_auto_file_saves_session_when_threshold_met(monkeypatch):
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    await d._maybe_auto_file_wiki(
        query="what is ml?",
        response={"answer": "Machine learning is a subset of AI."},
        entities=["Machine Learning"],
        agent_name="search_agent",
        tenant_id="acme:acme",
        turn_count=3,
    )

    wm._should_auto_file.assert_called_once_with(
        ["Machine Learning"], "search_agent", 3
    )
    wm.save_session.assert_called_once()
    kwargs = wm.save_session.call_args.kwargs
    assert kwargs["query"] == "what is ml?"
    assert kwargs["response"] == "Machine learning is a subset of AI."
    assert kwargs["entities"] == ["Machine Learning"]
    assert kwargs["agent_name"] == "search_agent"


@pytest.mark.asyncio
async def test_auto_file_skips_when_threshold_not_met(monkeypatch):
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = False
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    await d._maybe_auto_file_wiki(
        query="q",
        response={"answer": "a"},
        entities=[],
        agent_name="search_agent",
        tenant_id="acme:acme",
        turn_count=1,
    )

    wm.save_session.assert_not_called()


@pytest.mark.asyncio
async def test_auto_file_noop_when_no_factory(monkeypatch):
    from cogniverse_runtime.routers import wiki as wiki_router

    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", None)

    d = _dispatcher()
    # Must not raise even though no wiki factory is configured.
    await d._maybe_auto_file_wiki(
        query="q",
        response={"answer": "a"},
        entities=["E"],
        agent_name="search_agent",
        tenant_id="acme:acme",
        turn_count=5,
    )


@pytest.mark.asyncio
async def test_auto_file_swallows_save_errors(monkeypatch):
    """The hook is fire-and-forget: a save failure must not propagate (it would
    crash the background task), only warn."""
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    wm.save_session.side_effect = RuntimeError("vespa down")
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    # Must not raise.
    await d._maybe_auto_file_wiki(
        query="q",
        response={"answer": "a"},
        entities=["E"],
        agent_name="search_agent",
        tenant_id="acme:acme",
        turn_count=5,
    )


@pytest.mark.asyncio
async def test_auto_file_projects_extraction_records_to_titles(monkeypatch):
    """EntityExtractionAgent emits Entity.model_dump() dicts, so the dispatcher
    hands save_session records, not strings. The wiki stores titles."""
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    records = [
        {"text": "Barack Obama", "type": "PERSON", "confidence": 0.9, "context": ""},
        {"text": "Chicago", "type": "PLACE", "confidence": 0.8, "context": ""},
    ]

    d = _dispatcher()
    await d._maybe_auto_file_wiki(
        query="Show me videos about Barack Obama in Chicago",
        response={"answer": "Here are the videos."},
        entities=records,
        agent_name="search_agent",
        tenant_id="acme:acme",
        turn_count=3,
    )

    wm._should_auto_file.assert_called_once_with(
        ["Barack Obama", "Chicago"], "search_agent", 3
    )
    assert wm.save_session.call_args.kwargs["entities"] == ["Barack Obama", "Chicago"]


@pytest.mark.asyncio
async def test_auto_file_reports_the_entity_shape_when_saving_fails(
    monkeypatch, caplog
):
    """A dead auto-file feature ran unnoticed because the handler logged at
    WARNING without the payload shape. The failure must name the shape."""
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    wm.save_session.side_effect = RuntimeError("vespa down")
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    with caplog.at_level(logging.ERROR, logger="cogniverse_runtime.agent_dispatcher"):
        await d._maybe_auto_file_wiki(
            query="q",
            response={"answer": "a"},
            entities=["Machine Learning", "Deep Learning"],
            agent_name="search_agent",
            tenant_id="acme:acme",
            turn_count=5,
        )

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(records) == 1
    message = records[0].getMessage()
    assert message == (
        "Wiki auto-filing failed for search_agent (tenant=acme:acme): "
        "RuntimeError: vespa down; entities=2 items, item types={'str'}"
    )
    assert records[0].exc_info is not None


@pytest.mark.asyncio
async def test_auto_file_reports_dict_shape_including_keys(monkeypatch, caplog):
    """The shape must name a record's keys — that is what identifies the
    producer whose contract drifted."""
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    wm.save_session.side_effect = RuntimeError("boom")
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    with caplog.at_level(logging.ERROR, logger="cogniverse_runtime.agent_dispatcher"):
        await d._maybe_auto_file_wiki(
            query="q",
            response={"answer": "a"},
            entities=[{"text": "Chicago", "type": "PLACE"}],
            agent_name="search_agent",
            tenant_id="acme:acme",
            turn_count=5,
        )

    message = [r for r in caplog.records if r.levelno == logging.ERROR][0].getMessage()
    assert message == (
        "Wiki auto-filing failed for search_agent (tenant=acme:acme): "
        "RuntimeError: boom; entities=1 items, item types={'dict'}, "
        "dict keys={'text', 'type'}"
    )


@pytest.mark.asyncio
async def test_auto_file_rejects_an_entity_record_without_text(monkeypatch, caplog):
    """An unknown record shape must fail naming the shape, not raise a
    TypeError from inside unicodedata four frames down."""
    from cogniverse_runtime.routers import wiki as wiki_router

    wm = MagicMock()
    wm._should_auto_file.return_value = True
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: wm)

    d = _dispatcher()
    with caplog.at_level(logging.ERROR, logger="cogniverse_runtime.agent_dispatcher"):
        await d._maybe_auto_file_wiki(
            query="q",
            response={"answer": "a"},
            entities=[{"name": "Chicago", "type": "PLACE"}],
            agent_name="search_agent",
            tenant_id="acme:acme",
            turn_count=5,
        )

    wm.save_session.assert_not_called()
    message = [r for r in caplog.records if r.levelno == logging.ERROR][0].getMessage()
    assert message == (
        "Wiki auto-filing failed for search_agent (tenant=acme:acme): "
        "InvalidWikiTitleError: entities[0] must be a wiki title str or an "
        "entity record with a 'text' str, got dict with keys ['name', 'type']"
        "; entities=1 items, item types={'dict'}, dict keys={'name', 'type'}"
    )
