"""Unit tests for the gateway dispatch arms (audit fix #4).

Verifies that ``MessagingGateway._handle_message`` actually dispatches the
four custom command families (/wiki, /instructions, /memories, /jobs)
through to the runtime_client. Before this fix the parsed flags were
silently dropped and every custom command fell through to the routing
agent — users could see /wiki in the help text but it didn't work.

The tests construct a real ``MessagingGateway`` with a mocked
``runtime_client`` and a mock Telegram ``Update``, parse a command via
``parse_message``, and assert the right runtime_client method was called.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.gateway import MessagingGateway


@pytest.fixture
def gateway_with_mock_client():
    """Build a MessagingGateway whose runtime_client is fully mocked."""
    g = MessagingGateway(
        bot_token="fake-token",
        runtime_url="http://runtime",
    )
    mock_client = MagicMock()
    # Each CRUD method becomes an AsyncMock returning a simple OK shape.
    mock_client.search_wiki = AsyncMock(
        return_value={"results": [], "count": 0}
    )
    mock_client.get_wiki_topic = AsyncMock(
        return_value={"slug": "foo", "content": "topic body"}
    )
    mock_client.get_wiki_index = AsyncMock(return_value={"content": "INDEX"})
    mock_client.lint_wiki = AsyncMock(return_value={"issues": []})
    mock_client.delete_wiki_topic = AsyncMock(
        return_value={"status": "deleted"}
    )
    mock_client.set_instructions = AsyncMock(
        return_value={"text": "x", "updated_at": "now"}
    )
    mock_client.get_instructions = AsyncMock(return_value={"text": "current"})
    mock_client.list_memories = AsyncMock(
        return_value={"memories": [], "count": 5}
    )
    mock_client.clear_memories = AsyncMock(return_value={"status": "cleared"})
    mock_client.list_jobs = AsyncMock(return_value={"jobs": []})
    mock_client.create_job = AsyncMock(
        return_value={
            "job_id": "abc12345",
            "name": "weekly_news",
            "status": "created",
        }
    )
    mock_client.delete_job = AsyncMock(return_value={"status": "deleted"})
    g.runtime_client = mock_client
    return g, mock_client


@pytest.fixture
def mock_update():
    """Build a mock Telegram Update with a recording reply_text."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


@pytest.mark.unit
@pytest.mark.ci_fast
class TestWikiDispatch:
    @pytest.mark.asyncio
    async def test_wiki_search_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki search ColPali")
        assert parsed.is_wiki and parsed.wiki_subcommand == "search"

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.search_wiki.assert_awaited_once_with(
            tenant_id="acme", query="ColPali"
        )
        mock_update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wiki_topic_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki topic foo")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.get_wiki_topic.assert_awaited_once_with(
            tenant_id="acme", slug="foo"
        )

    @pytest.mark.asyncio
    async def test_wiki_index_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki index")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.get_wiki_index.assert_awaited_once_with(tenant_id="acme")

    @pytest.mark.asyncio
    async def test_wiki_lint_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki lint")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.lint_wiki.assert_awaited_once_with(tenant_id="acme")

    @pytest.mark.asyncio
    async def test_wiki_delete_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki delete foo")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.delete_wiki_topic.assert_awaited_once_with(
            tenant_id="acme", slug="foo"
        )

    @pytest.mark.asyncio
    async def test_wiki_search_without_query_shows_usage(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki search")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        # Should not call the API with an empty query.
        client.search_wiki.assert_not_awaited()
        mock_update.message.reply_text.assert_awaited_once()
        msg = mock_update.message.reply_text.call_args[0][0]
        assert "Usage" in msg


@pytest.mark.unit
@pytest.mark.ci_fast
class TestInstructionsDispatch:
    @pytest.mark.asyncio
    async def test_instructions_set_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/instructions set always respond in french")

        await g._handle_instructions_command(mock_update, parsed, "acme")

        client.set_instructions.assert_awaited_once_with(
            tenant_id="acme", text="always respond in french"
        )

    @pytest.mark.asyncio
    async def test_instructions_show_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/instructions show")

        await g._handle_instructions_command(mock_update, parsed, "acme")

        client.get_instructions.assert_awaited_once_with(tenant_id="acme")

    @pytest.mark.asyncio
    async def test_instructions_set_without_text_shows_usage(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/instructions set")

        await g._handle_instructions_command(mock_update, parsed, "acme")

        client.set_instructions.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMemoriesDispatch:
    @pytest.mark.asyncio
    async def test_memories_list_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/memories list")

        await g._handle_memories_command(mock_update, parsed, "acme")

        client.list_memories.assert_awaited_once_with(
            tenant_id="acme", agent_name=None
        )

    @pytest.mark.asyncio
    async def test_memories_list_with_agent_filter(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/memories list agent=search_agent")

        await g._handle_memories_command(mock_update, parsed, "acme")

        client.list_memories.assert_awaited_once_with(
            tenant_id="acme", agent_name="search_agent"
        )

    @pytest.mark.asyncio
    async def test_memories_clear_strategies_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/memories clear strategies")

        await g._handle_memories_command(mock_update, parsed, "acme")

        client.clear_memories.assert_awaited_once_with(
            tenant_id="acme", agent_name="strategies"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestJobsDispatch:
    @pytest.mark.asyncio
    async def test_jobs_list_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/jobs list")

        await g._handle_jobs_command(mock_update, parsed, "acme")

        client.list_jobs.assert_awaited_once_with(tenant_id="acme")

    @pytest.mark.asyncio
    async def test_jobs_create_parses_quoted_schedule(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message('/jobs create "0 9 * * 1" weekly AI news')

        await g._handle_jobs_command(mock_update, parsed, "acme")

        client.create_job.assert_awaited_once()
        call = client.create_job.call_args
        assert call.kwargs["tenant_id"] == "acme"
        assert call.kwargs["schedule"] == "0 9 * * 1"
        assert call.kwargs["query"] == "weekly AI news"
        assert call.kwargs["name"] == "weekly AI news"

    @pytest.mark.asyncio
    async def test_jobs_create_without_quotes_shows_usage(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/jobs create 0 9 * * 1 weekly")

        await g._handle_jobs_command(mock_update, parsed, "acme")

        client.create_job.assert_not_awaited()
        mock_update.message.reply_text.assert_awaited_once()
        assert "Usage" in mock_update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_jobs_delete_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/jobs delete abc12345")

        await g._handle_jobs_command(mock_update, parsed, "acme")

        client.delete_job.assert_awaited_once_with(
            tenant_id="acme", job_id="abc12345"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestParseJobsCreateArgs:
    """Test the static helper for parsing /jobs create syntax."""

    def test_quoted_schedule_and_query(self):
        result = MessagingGateway._parse_jobs_create_args(
            '"0 9 * * 1" latest AI papers'
        )
        assert result == ("0 9 * * 1", "latest AI papers", "latest AI papers")

    def test_no_quotes_returns_none(self):
        assert MessagingGateway._parse_jobs_create_args(
            "0 9 * * 1 latest"
        ) == (None, None, None)

    def test_unclosed_quote_returns_none(self):
        assert MessagingGateway._parse_jobs_create_args(
            '"0 9 * * 1 latest'
        ) == (None, None, None)

    def test_empty_query_returns_none(self):
        assert MessagingGateway._parse_jobs_create_args('"0 9 * * 1"') == (
            None,
            None,
            None,
        )
