"""Pin test for audit fixes #17 and #18 — wiki endpoints no longer orphaned.

The audit found two orphan endpoints:
- ``GET /wiki/lint`` — exposed but no UI/CLI/test caller
- ``DELETE /wiki/topic/{slug}`` — exposed but no UI/CLI/test caller

Wave 3 (audit fixes #4 + #5) wired the full Telegram chain so these
endpoints are now reachable via ``/wiki lint`` and ``/wiki delete <slug>``
slash commands. This module pins the chain end-to-end so a regression
that breaks any link surfaces immediately.

The chain is:
    Telegram message → command_router.parse_message → gateway dispatch arm
        → runtime_client method → wiki router endpoint

If any link breaks, these endpoints become orphaned again.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.gateway import MessagingGateway
from cogniverse_messaging.runtime_client import RuntimeClient


@pytest.mark.unit
@pytest.mark.ci_fast
class TestWikiLintNoLongerOrphaned:
    @pytest.mark.asyncio
    async def test_full_chain_for_wiki_lint(self):
        """End-to-end pin: ``/wiki lint`` from a Telegram message must
        actually result in ``GET /wiki/lint`` being called."""
        # Stage 1: parse the slash command.
        parsed = parse_message("/wiki lint")
        assert parsed.is_wiki
        assert parsed.wiki_subcommand == "lint"

        # Stage 2: gateway must have a dispatch arm and call lint_wiki().
        g = MessagingGateway(bot_token="x", runtime_url="http://x")
        mock_client = MagicMock()
        mock_client.lint_wiki = AsyncMock(return_value={"issues": []})
        g.runtime_client = mock_client

        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()

        await g._handle_wiki_command(update, parsed, "acme")

        mock_client.lint_wiki.assert_awaited_once_with(tenant_id="acme")

    @pytest.mark.asyncio
    async def test_runtime_client_lint_wiki_hits_get_endpoint(self):
        """Stage 3: lint_wiki() must POST GET to the canonical endpoint."""
        rc = RuntimeClient("http://runtime")
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"issues": []}),
                text="",
            )
        )
        rc._client = mock_http

        await rc.lint_wiki(tenant_id="acme")

        mock_http.get.assert_awaited_once()
        assert mock_http.get.call_args[0][0] == "/wiki/lint"
        assert mock_http.get.call_args[1]["params"]["tenant_id"] == "acme"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestWikiDeleteNoLongerOrphaned:
    @pytest.mark.asyncio
    async def test_full_chain_for_wiki_delete(self):
        """End-to-end pin: ``/wiki delete foo`` must result in
        ``DELETE /wiki/topic/foo`` being called."""
        parsed = parse_message("/wiki delete foo")
        assert parsed.is_wiki
        assert parsed.wiki_subcommand == "delete"
        assert parsed.query == "foo"

        g = MessagingGateway(bot_token="x", runtime_url="http://x")
        mock_client = MagicMock()
        mock_client.delete_wiki_topic = AsyncMock(
            return_value={"status": "deleted"}
        )
        g.runtime_client = mock_client

        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()

        await g._handle_wiki_command(update, parsed, "acme")

        mock_client.delete_wiki_topic.assert_awaited_once_with(
            tenant_id="acme", slug="foo"
        )

    @pytest.mark.asyncio
    async def test_runtime_client_delete_wiki_topic_hits_canonical_endpoint(self):
        rc = RuntimeClient("http://runtime")
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_http.delete = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "deleted"}),
                text="",
            )
        )
        rc._client = mock_http

        await rc.delete_wiki_topic(tenant_id="acme", slug="foo")

        mock_http.delete.assert_awaited_once()
        assert mock_http.delete.call_args[0][0] == "/wiki/topic/foo"
        assert mock_http.delete.call_args[1]["params"]["tenant_id"] == "acme"
