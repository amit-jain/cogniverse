"""
Integration tests for ConversationalQueryRewriteModule with a real LLM.

Validates that the DSPy ChainOfThought module actually resolves anaphoric
references (pronouns, comparative references) using conversation history.
No mocks — exercises the real LLM via the project's configured endpoint.
"""

import pytest

from cogniverse_agents.search_agent import ConversationalQueryRewriteModule
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from tests.agents.integration.conftest import skip_if_no_llm


@pytest.mark.integration
@skip_if_no_llm
class TestQueryRewriteWithRealLLM:
    """Test ConversationalQueryRewriteModule against a real LLM."""

    def test_rewrite_resolves_pronoun_reference(self, dspy_lm):
        """Query with 'those' + history about cats -> rewritten query contains 'cat'."""
        module = ConversationalQueryRewriteModule()

        result = module(
            query="show me more like those",
            conversation_history="user: search for cat videos\nagent: Found 5 cat video results",
        )

        assert result.rewritten_query, "rewritten_query should not be None or empty"
        rewritten = result.rewritten_query.lower()
        assert any(
            word in rewritten for word in ["cat", "video"]
        ), f"Rewritten query '{result.rewritten_query}' should reference cats/videos from history"

    def test_rewrite_resolves_comparative_reference(self, dspy_lm):
        """Query with 'longer ones' + history about cooking -> resolved query."""
        module = ConversationalQueryRewriteModule()

        result = module(
            query="find longer ones",
            conversation_history="user: search for short cooking tutorials\nagent: Found 3 short cooking tutorial results",
        )

        assert result.rewritten_query, "rewritten_query should not be None or empty"
        rewritten = result.rewritten_query.lower()
        assert any(
            word in rewritten for word in ["cooking", "tutorial", "long"]
        ), f"Rewritten query '{result.rewritten_query}' should reference cooking/long from history"

    def test_standalone_query_preserved(self, dspy_lm):
        """A self-contained query should not be distorted by unrelated history."""
        module = ConversationalQueryRewriteModule()

        result = module(
            query="search for dog videos",
            conversation_history="user: find cat photos\nagent: Found 10 cat photos",
        )

        assert result.rewritten_query, "rewritten_query should not be None or empty"
        rewritten = result.rewritten_query.lower()
        assert "dog" in rewritten, (
            f"Standalone query about dogs should still contain 'dog', got '{result.rewritten_query}'"
        )

    def test_rewrite_with_multi_turn_history(self, dspy_lm):
        """3-turn history + follow-up -> LLM resolves using full context."""
        module = ConversationalQueryRewriteModule()

        history = (
            "user: search for cooking videos\n"
            "agent: Found 8 cooking video results\n"
            "user: filter by Italian cuisine\n"
            "agent: Found 3 Italian cooking videos\n"
            "user: which ones are under 10 minutes\n"
            "agent: Found 2 short Italian cooking videos"
        )

        result = module(
            query="show me the recipes from those",
            conversation_history=history,
        )

        assert result.rewritten_query, "rewritten_query should not be None or empty"
        rewritten = result.rewritten_query.lower()
        assert any(
            word in rewritten for word in ["italian", "cooking", "recipe"]
        ), f"Rewritten query '{result.rewritten_query}' should reference Italian cooking from history"

    @pytest.mark.asyncio
    async def test_dispatcher_rewrite_with_real_llm(self, dspy_lm):
        """AgentDispatcher._rewrite_query_with_history produces a resolved string."""
        from unittest.mock import MagicMock

        dispatcher = AgentDispatcher(
            agent_registry=MagicMock(),
            config_manager=MagicMock(),
            schema_loader=MagicMock(),
        )

        history = [
            {"role": "user", "content": "search for cat videos"},
            {"role": "agent", "content": "Found 5 cat video results"},
        ]

        rewritten = await dispatcher._rewrite_query_with_history(
            "show me more like those", history
        )

        assert isinstance(rewritten, str)
        assert len(rewritten) > 0
        # Should resolve the pronoun — result should mention cats/videos
        rewritten_lower = rewritten.lower()
        assert any(
            word in rewritten_lower for word in ["cat", "video"]
        ), f"Dispatcher rewrite '{rewritten}' should resolve 'those' to cats/videos"
