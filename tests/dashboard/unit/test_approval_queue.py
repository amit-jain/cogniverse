"""Unit tests for the Approval Queue dashboard tab."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_dashboard.tabs import approval_queue


class _SessionState(dict):
    """Mirror Streamlit's session_state: both attribute and item access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.unit
class TestLoadPendingItems:
    def test_queries_persisted_store_not_just_session_batch(self, monkeypatch):
        """Regression: _load_pending_items read only the in-session
        last_generated_batch, so approvals persisted by another process never
        appeared. It must query the agent's persisted pending items."""
        persisted = ["item-a", "item-b"]
        agent = MagicMock()
        agent.get_pending_items = AsyncMock(return_value=persisted)

        stale_batch = MagicMock()
        stale_batch.pending_review = ["stale-from-session"]

        fake_st = MagicMock()
        fake_st.session_state = _SessionState(
            approval_agent=agent,
            current_tenant="acme",
            last_generated_batch=stale_batch,
        )
        monkeypatch.setattr(approval_queue, "st", fake_st)

        approval_queue._load_pending_items()

        assert fake_st.session_state["pending_items"] == persisted
        agent.get_pending_items.assert_awaited_once_with({"tenant_id": "acme"})

    def test_falls_back_to_session_batch_without_agent(self, monkeypatch):
        """No agent yet (e.g. fresh synthetic batch) -> show the session batch."""
        batch = MagicMock()
        batch.pending_review = ["fresh-1", "fresh-2"]
        batch.auto_approved = ["auto-1"]

        fake_st = MagicMock()
        fake_st.session_state = _SessionState(last_generated_batch=batch)
        monkeypatch.setattr(approval_queue, "st", fake_st)

        approval_queue._load_pending_items()

        assert fake_st.session_state["pending_items"] == ["fresh-1", "fresh-2"]
        assert fake_st.session_state["approved_items"] == ["auto-1"]
