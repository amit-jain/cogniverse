"""ConversationManager store→parse round-trip against a partition-faithful
Mem0 double.

The write format (``[chat:{id}] [role] {content}``) and the read parser
must agree — the parser previously matched by substring, so ``[chat:1]``
also matched chat 12's turns and a turn whose content contained
``[user]`` was misclassified regardless of its stored role.
"""

import pytest
from cogniverse_messaging.conversation import ConversationManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _SearchableMemory:
    """Partitioned in-memory Mem0 stand-in with a working search."""

    def __init__(self):
        self.store = {}
        self.memory = object()  # non-None: the manager gates on .memory

    def add_memory(self, content, tenant_id, agent_name, metadata=None, **kwargs):
        self.store.setdefault((tenant_id, agent_name), []).append(
            {"memory": content, "metadata": metadata or {}}
        )

    def search_memory(self, query, tenant_id, agent_name, top_k=10):
        return self.store.get((tenant_id, agent_name), [])[:top_k]


@pytest.fixture
def manager():
    return ConversationManager(_SearchableMemory(), tenant_id="acme:alice")


def test_store_then_get_round_trip(manager):
    manager.store_turn("99", "user", "what is a vespa schema?")
    manager.store_turn("99", "assistant", "A schema defines document fields.")

    turns = manager.get_history("99")

    assert turns == [
        {"role": "user", "content": "what is a vespa schema?"},
        {"role": "assistant", "content": "A schema defines document fields."},
    ]


def test_chat_prefix_isolation(manager):
    """Chat "1" must not read chat "12"'s turns — the old substring match
    on ``[chat:1]`` matched both."""
    manager.store_turn("1", "user", "mine")
    manager.store_turn("12", "user", "not mine")

    turns = manager.get_history("1")

    assert turns == [{"role": "user", "content": "mine"}]


def test_role_tag_inside_content_keeps_real_role(manager):
    """A turn whose CONTENT mentions ``[user]`` keeps its stored role —
    the old substring check classified it as a user turn."""
    manager.store_turn("7", "assistant", "the [user] tag marks user turns")

    turns = manager.get_history("7")

    assert turns == [
        {"role": "assistant", "content": "the [user] tag marks user turns"}
    ]


def test_history_capped_by_max_turns(manager):
    for i in range(15):
        manager.store_turn("5", "user", f"m{i}")

    assert len(manager.get_history("5", max_turns=10)) == 10


def test_no_memory_backend_returns_empty(manager):
    manager.memory_manager = None
    assert manager.get_history("99") == []
    manager.store_turn("99", "user", "dropped silently")  # must not raise
