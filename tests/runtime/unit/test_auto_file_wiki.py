"""The dispatcher's wiki auto-file hook runs after every successful dispatch but
had zero coverage of its body — a signature drift on the private
_should_auto_file / save_session calls (or the factory shape) would silently
kill wiki auto-filing on every dispatch, logging only a warning. These exercise
the real body.
"""

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
