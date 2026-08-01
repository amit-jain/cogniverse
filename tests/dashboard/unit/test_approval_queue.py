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


@pytest.mark.unit
class TestPersistDecision:
    """The approve path must feed the training dataset, not only a span."""

    @staticmethod
    def _fake_st(storage):
        fake_st = MagicMock()
        fake_st.session_state = _SessionState(approval_storage=storage)
        return fake_st

    def _item(self):
        from cogniverse_agents.approval import ReviewItem

        return ReviewItem(item_id="s0", data={"query": "q"}, confidence=0.9)

    def test_approved_decision_appends_to_training_dataset(self, monkeypatch):
        from cogniverse_agents.approval import ReviewDecision

        calls = []
        storage = MagicMock()
        storage.record_decision = AsyncMock(
            side_effect=lambda *a, **k: calls.append("record_decision")
        )
        storage.append_to_training_dataset = AsyncMock(
            side_effect=lambda *a, **k: calls.append("append_to_training_dataset")
        )
        monkeypatch.setattr(approval_queue, "st", self._fake_st(storage))

        item = self._item()
        approval_queue._persist_decision(
            ReviewDecision(item_id="s0", approved=True, reviewer="u"), item
        )

        storage.record_decision.assert_awaited_once()
        storage.append_to_training_dataset.assert_awaited_once()
        args, _ = storage.append_to_training_dataset.call_args
        assert args[0] == "approved_synthetic_data"
        assert args[1] == [item]
        # The dataset append must land before the decision is recorded: the
        # decision annotation is what makes the item stop appearing as
        # pending, so recording it first and having the append fail would
        # orphan the item with no pending-queue entry to retry it from.
        assert calls == ["append_to_training_dataset", "record_decision"]

    def test_failed_append_never_records_decision(self, monkeypatch):
        """A training-dataset append failure must not leave the item marked
        approved: the decision write must not happen, so the item stays
        visible as pending and the approval can be retried."""
        from cogniverse_agents.approval import ReviewDecision

        storage = MagicMock()
        storage.record_decision = AsyncMock()
        storage.append_to_training_dataset = AsyncMock(
            side_effect=RuntimeError("dataset backend unreachable")
        )
        monkeypatch.setattr(approval_queue, "st", self._fake_st(storage))

        item = self._item()
        with pytest.raises(RuntimeError, match="dataset backend unreachable"):
            approval_queue._persist_decision(
                ReviewDecision(item_id="s0", approved=True, reviewer="u"), item
            )

        storage.append_to_training_dataset.assert_awaited_once()
        storage.record_decision.assert_not_awaited()

    def test_rejected_decision_does_not_append(self, monkeypatch):
        from cogniverse_agents.approval import ReviewDecision

        storage = MagicMock()
        storage.record_decision = AsyncMock()
        storage.append_to_training_dataset = AsyncMock()
        monkeypatch.setattr(approval_queue, "st", self._fake_st(storage))

        approval_queue._persist_decision(
            ReviewDecision(item_id="s0", approved=False, reviewer="u"), self._item()
        )

        storage.record_decision.assert_awaited_once()
        storage.append_to_training_dataset.assert_not_awaited()
