"""Unit tests for the gateway dispatch arms.

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
    mock_client.search_wiki = AsyncMock(return_value={"results": [], "count": 0})
    mock_client.get_wiki_topic = AsyncMock(
        return_value={"slug": "foo", "content": "topic body"}
    )
    mock_client.get_wiki_index = AsyncMock(return_value={"content": "INDEX"})
    mock_client.lint_wiki = AsyncMock(return_value={"issues": []})
    mock_client.delete_wiki_topic = AsyncMock(return_value={"status": "deleted"})
    mock_client.set_instructions = AsyncMock(
        return_value={"text": "x", "updated_at": "now"}
    )
    mock_client.get_instructions = AsyncMock(return_value={"text": "current"})
    mock_client.list_memories = AsyncMock(return_value={"memories": [], "count": 5})
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

        client.search_wiki.assert_awaited_once_with(tenant_id="acme", query="ColPali")
        mock_update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wiki_topic_calls_runtime_client(
        self, gateway_with_mock_client, mock_update
    ):
        g, client = gateway_with_mock_client
        parsed = parse_message("/wiki topic foo")

        await g._handle_wiki_command(mock_update, parsed, "acme")

        client.get_wiki_topic.assert_awaited_once_with(tenant_id="acme", slug="foo")

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

        client.delete_wiki_topic.assert_awaited_once_with(tenant_id="acme", slug="foo")

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

        client.list_memories.assert_awaited_once_with(tenant_id="acme", agent_name=None)

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
            tenant_id="acme", category="strategies"
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

        client.delete_job.assert_awaited_once_with(tenant_id="acme", job_id="abc12345")


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
        assert MessagingGateway._parse_jobs_create_args("0 9 * * 1 latest") == (
            None,
            None,
            None,
        )

    def test_unclosed_quote_returns_none(self):
        assert MessagingGateway._parse_jobs_create_args('"0 9 * * 1 latest') == (
            None,
            None,
            None,
        )

    def test_empty_query_returns_none(self):
        assert MessagingGateway._parse_jobs_create_args('"0 9 * * 1"') == (
            None,
            None,
            None,
        )


def _message_update(
    text=None, caption=None, photo=None, video=None, user_id=7, chat_id=99
):
    """Full fake Telegram Update for driving _handle_message itself."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    msg = update.message
    msg.text = text
    msg.caption = caption
    msg.photo = photo or []
    msg.video = video
    msg.reply_text = AsyncMock()
    msg.chat.send_action = AsyncMock()
    return update


@pytest.mark.unit
@pytest.mark.ci_fast
class TestHandleMessage:
    """Drive the central _handle_message handler itself — registration gate,
    parse_message routing, media extraction, dispatch_agent call, and the
    chunked-reply loop. Every non-command Telegram message and the
    /search|/summarize|/report|/research|/code commands flow through here;
    previously only the sub-handlers were tested."""

    def _gateway(self, tenant_id="acme:acme"):
        g = MessagingGateway(bot_token="fake-token", runtime_url="http://runtime")
        g.runtime_client = MagicMock()
        g.runtime_client.dispatch_agent = AsyncMock(
            return_value={"message": "the answer"}
        )
        g._user_mapper = MagicMock()
        g._user_mapper.get_tenant_id.return_value = tenant_id
        return g

    @pytest.mark.asyncio
    async def test_unregistered_user_is_gated_and_never_dispatched(self):
        from cogniverse_messaging.telegram_handler import format_registration_required

        g = self._gateway(tenant_id=None)
        update = _message_update(text="find sunsets")

        await g._handle_message(update, context=None)

        update.message.reply_text.assert_awaited_once_with(
            format_registration_required()
        )
        g.runtime_client.dispatch_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plain_text_dispatches_to_gateway_agent_with_history(self):
        g = self._gateway()
        conv = MagicMock()
        conv.get_history.return_value = [{"role": "user", "content": "earlier"}]
        g._get_conversation_manager = MagicMock(return_value=conv)
        update = _message_update(text="find sunsets over water")

        await g._handle_message(update, context=None)

        g.runtime_client.dispatch_agent.assert_awaited_once_with(
            agent_name="gateway_agent",
            query="find sunsets over water",
            tenant_id="acme:acme",
            context_id="99",
            conversation_history=[{"role": "user", "content": "earlier"}],
            context={},
        )
        update.message.chat.send_action.assert_awaited_once_with("typing")
        update.message.reply_text.assert_awaited_once_with("the answer")
        # Both turns stored: the user's query and the assistant's reply.
        conv.store_turn.assert_any_call("99", "user", "find sunsets over water")
        conv.store_turn.assert_any_call("99", "assistant", "the answer")

    @pytest.mark.asyncio
    async def test_search_command_routes_to_search_agent(self):
        g = self._gateway()
        update = _message_update(text="/search red kite on a beach")

        await g._handle_message(update, context=None)

        kwargs = g.runtime_client.dispatch_agent.await_args.kwargs
        assert kwargs["agent_name"] == "search_agent"
        assert kwargs["query"] == "red kite on a beach"
        assert kwargs["tenant_id"] == "acme:acme"

    @pytest.mark.asyncio
    async def test_help_replies_without_dispatching(self):
        from cogniverse_messaging.telegram_handler import format_help

        g = self._gateway()
        update = _message_update(text="/help")

        await g._handle_message(update, context=None)

        update.message.reply_text.assert_awaited_once_with(format_help())
        g.runtime_client.dispatch_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_photo_with_caption_threads_media_context(self):
        g = self._gateway()
        photo = MagicMock()
        photo.file_id = "photo-file-123"
        update = _message_update(caption="what is in this image", photo=[photo])

        await g._handle_message(update, context=None)

        kwargs = g.runtime_client.dispatch_agent.await_args.kwargs
        assert kwargs["context"] == {
            "media_type": "photo",
            "media_file_id": "photo-file-123",
        }
        assert kwargs["query"] == "what is in this image"

    @pytest.mark.asyncio
    async def test_empty_query_prompts_for_usage(self):
        g = self._gateway()
        update = _message_update(text="/search")

        await g._handle_message(update, context=None)

        g.runtime_client.dispatch_agent.assert_not_awaited()
        reply = update.message.reply_text.await_args.args[0]
        assert "query" in reply.lower() or "help" in reply.lower()

    @pytest.mark.asyncio
    async def test_wiki_command_delegates_to_wiki_handler_with_tenant(self):
        g = self._gateway()
        g._handle_wiki_command = AsyncMock()
        update = _message_update(text="/wiki search ColPali")

        await g._handle_message(update, context=None)

        args = g._handle_wiki_command.await_args.args
        assert args[0] is update
        assert args[1].is_wiki and args[1].wiki_subcommand == "search"
        assert args[2] == "acme:acme"
        g.runtime_client.dispatch_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_response_is_replied_chunk_by_chunk(self):
        """A response beyond Telegram's message limit is split by
        format_agent_response and EVERY chunk is sent, in order."""
        from cogniverse_messaging.telegram_handler import format_agent_response

        long_message = "\n".join(f"line {i}: " + "x" * 80 for i in range(120))
        response = {"message": long_message}
        expected_chunks = format_agent_response(response)
        assert len(expected_chunks) >= 2  # exceeds MAX_MESSAGE_LENGTH -> split

        g = self._gateway()
        g.runtime_client.dispatch_agent = AsyncMock(return_value=response)
        update = _message_update(text="find clips")

        await g._handle_message(update, context=None)

        sent = [c.args[0] for c in update.message.reply_text.await_args_list]
        assert sent == expected_chunks
