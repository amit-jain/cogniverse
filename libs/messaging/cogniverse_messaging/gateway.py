"""Messaging Gateway — main entry point.

Runs as a separate service. Supports webhook (production) and
long-polling (development) modes for Telegram.

Usage:
    python -m cogniverse_messaging.gateway
    GATEWAY_MODE=webhook python -m cogniverse_messaging.gateway
"""

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.runtime_client import RuntimeClient
from cogniverse_messaging.telegram_handler import (
    format_agent_response,
    format_help,
    format_invalid_token,
    format_registration_required,
    format_registration_success,
)

logger = logging.getLogger(__name__)


class MessagingGateway:
    """Telegram messaging gateway for Cogniverse.

    Translates Telegram messages to runtime API calls and sends
    formatted responses back. Registration, tenant resolution, and
    conversation history all live behind the runtime's HTTP API (the
    runtime owns the token store, the user-tenant mappings, and Mem0), so
    the gateway holds no backend connection and works in the deployed
    chart from RUNTIME_URL alone.
    """

    _TENANT_CACHE_TTL_SECONDS = 60.0
    _TENANT_CACHE_MAX_ENTRIES = 10000

    def __init__(
        self,
        bot_token: str,
        runtime_url: str,
        mode: str = "polling",
        webhook_url: str = "",
        webhook_listen: str = "0.0.0.0",
        webhook_port: int = 8443,
        webhook_path: str = "",
        outbound_poll_seconds: float = 5.0,
    ):
        self.bot_token = bot_token
        self.mode = mode
        self.webhook_url = webhook_url
        self.webhook_listen = webhook_listen
        self.webhook_port = webhook_port
        self.webhook_path = webhook_path
        self.runtime_client = RuntimeClient(runtime_url)
        self._outbound_poll_seconds = outbound_poll_seconds

        self._app: Application = None
        # user_id -> (tenant_id, expiry). Only positive resolves are cached
        # so a fresh registration is visible on the user's next message.
        self._tenant_cache: dict = {}

    async def _resolve_tenant(self, user_id: str) -> dict:
        """Resolve a Telegram user's tenant via the runtime, with a short
        positive-only cache so it costs one HTTP hop per user per minute
        instead of one per message."""
        now = time.monotonic()
        cached = self._tenant_cache.get(user_id)
        if cached and cached[1] > now:
            return {"status": "ok", "tenant_id": cached[0]}
        result = await self.runtime_client.resolve_tenant("telegram", user_id)
        if result.get("status") == "ok" and result.get("tenant_id"):
            if len(self._tenant_cache) >= self._TENANT_CACHE_MAX_ENTRIES:
                self._tenant_cache.clear()
            self._tenant_cache[user_id] = (
                result["tenant_id"],
                now + self._TENANT_CACHE_TTL_SECONDS,
            )
        return result

    async def _handle_start(self, update: Update, context) -> None:
        """Handle /start command — user registration via the runtime.

        The runtime validates the token, stores the mapping, and consumes
        the token in that order, so "unavailable" always means the token
        survived for a retry.
        """
        # Parse through the shared router so /start uses one parser with the
        # rest of the commands (its is_registration / registration_token fields).
        parsed = parse_message(update.message.text or "")
        token = parsed.registration_token if parsed.is_registration else None

        if not token:
            await update.message.reply_text(
                "Welcome to Cogniverse!\n\n"
                "To register, send:\n/start <invite_token>\n\n"
                "Get a token from your admin."
            )
            return

        user_id = str(update.effective_user.id)
        result = await self.runtime_client.register_user("telegram", user_id, token)
        status = result.get("status")
        if status == "invalid_token":
            await update.message.reply_text(format_invalid_token())
            return
        if status != "registered" or not result.get("tenant_id"):
            logger.error("Registration unavailable: %s", result.get("message"))
            await update.message.reply_text(
                "Registration is temporarily unavailable — your token was not "
                "consumed, please try again shortly."
            )
            return

        tenant_id = result["tenant_id"]
        self._tenant_cache[user_id] = (
            tenant_id,
            time.monotonic() + self._TENANT_CACHE_TTL_SECONDS,
        )
        await update.message.reply_text(format_registration_success(tenant_id))

    async def _handle_help(self, update: Update, context) -> None:
        """Handle /help command."""
        await update.message.reply_text(format_help())

    async def _handle_message(self, update: Update, context) -> None:
        """Handle all messages — text, commands, and media."""
        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        resolved = await self._resolve_tenant(user_id)
        if resolved.get("status") != "ok":
            logger.error("Tenant lookup unavailable: %s", resolved.get("message"))
            await update.message.reply_text(
                "Service temporarily unavailable — please try again shortly."
            )
            return

        tenant_id = resolved.get("tenant_id")
        if not tenant_id:
            await update.message.reply_text(format_registration_required())
            return

        msg = update.message
        has_photo = bool(msg.photo)
        has_video = bool(msg.video)
        photo_file_id = msg.photo[-1].file_id if msg.photo else None
        video_file_id = msg.video.file_id if msg.video else None

        parsed = parse_message(
            text=msg.text or msg.caption,
            has_photo=has_photo,
            has_video=has_video,
            photo_file_id=photo_file_id,
            video_file_id=video_file_id,
        )

        if parsed.is_help:
            await update.message.reply_text(format_help())
            return

        # Dispatch the four custom command families. Each handler calls the
        # matching /wiki/* or /admin/tenant/* endpoint via runtime_client
        # and replies with a formatted result.
        if parsed.is_wiki:
            await self._handle_wiki_command(update, parsed, tenant_id)
            return
        if parsed.is_instructions:
            await self._handle_instructions_command(update, parsed, tenant_id)
            return
        if parsed.is_memories:
            await self._handle_memories_command(update, parsed, tenant_id)
            return
        if parsed.is_jobs:
            await self._handle_jobs_command(update, parsed, tenant_id)
            return

        if not parsed.query:
            await update.message.reply_text(
                "Please provide a query. Send /help for usage."
            )
            return

        await update.message.chat.send_action("typing")

        agent_context = {}
        if parsed.has_media and parsed.media_file_id:
            agent_context["media_type"] = parsed.media_type
            agent_context["media_file_id"] = parsed.media_file_id

        # Only context_id travels; the runtime loads and stores this chat's
        # conversation history around the agent call (it owns Mem0), so the
        # gateway holds no memory connection and history works in the
        # deployed chart.
        response = await self.runtime_client.dispatch_agent(
            agent_name=parsed.agent_name,
            query=parsed.query,
            tenant_id=tenant_id,
            context_id=chat_id,
            context=agent_context,
        )

        messages = format_agent_response(response)
        for chunk in messages:
            await update.message.reply_text(chunk)

    async def _handle_wiki_command(
        self, update: Update, parsed, tenant_id: str
    ) -> None:
        """Handle ``/wiki <subcommand> [args]`` — search/topic/index/lint/save/delete.

        Dispatches to the matching runtime_client method and replies with a
        short formatted result.
        """
        subcmd = (parsed.wiki_subcommand or "").lower()
        if subcmd == "search":
            if not parsed.query:
                await update.message.reply_text("Usage: /wiki search <query>")
                return
            result = await self.runtime_client.search_wiki(
                tenant_id=tenant_id, query=parsed.query
            )
            count = result.get("count", 0)
            await update.message.reply_text(
                f"Found {count} wiki result(s) for '{parsed.query}'."
            )
        elif subcmd == "topic":
            if not parsed.query:
                await update.message.reply_text("Usage: /wiki topic <slug>")
                return
            result = await self.runtime_client.get_wiki_topic(
                tenant_id=tenant_id, slug=parsed.query
            )
            if result.get("status") == "error":
                await update.message.reply_text(f"Topic '{parsed.query}' not found.")
            else:
                await update.message.reply_text(
                    str(result.get("content", result))[:3500]
                )
        elif subcmd == "index":
            result = await self.runtime_client.get_wiki_index(tenant_id=tenant_id)
            # ``content`` is a string from the runtime; treat both
            # missing-key AND empty-string as "empty wiki" so the
            # Telegram reply is always a non-empty message the
            # operator can see (an empty reply otherwise looks like
            # the command was dropped).
            content = result.get("content") or "(empty wiki)"
            await update.message.reply_text(str(content)[:3500])
        elif subcmd == "lint":
            result = await self.runtime_client.lint_wiki(tenant_id=tenant_id)
            # WikiManager.lint returns issues_found (int) plus orphan/stale/empty
            # page lists — NOT an "issues" key. Reading the missing key made
            # /wiki lint always report "no issues".
            issue_count = result.get("issues_found", 0)
            if issue_count:
                orphan = len(result.get("orphan_pages", []))
                stale = len(result.get("stale_pages", []))
                empty = len(result.get("empty_pages", []))
                await update.message.reply_text(
                    f"Wiki lint: {issue_count} issue(s) found "
                    f"({orphan} orphan, {stale} stale, {empty} empty)."
                )
            else:
                await update.message.reply_text("Wiki lint: no issues.")
        elif subcmd == "delete":
            if not parsed.query:
                await update.message.reply_text("Usage: /wiki delete <slug>")
                return
            result = await self.runtime_client.delete_wiki_topic(
                tenant_id=tenant_id, slug=parsed.query
            )
            await update.message.reply_text(
                f"Deleted wiki topic '{parsed.query}'."
                if result.get("status") == "deleted"
                else f"Delete failed: {result.get('message', 'unknown error')}"
            )
        elif subcmd == "save":
            await update.message.reply_text(
                "Wiki auto-saves agent sessions in the background. "
                "Use /wiki search <query> to find what's been saved."
            )
        else:
            await update.message.reply_text(
                "Unknown /wiki subcommand. Try: search, topic, index, lint, delete."
            )

    async def _handle_instructions_command(
        self, update: Update, parsed, tenant_id: str
    ) -> None:
        """Handle ``/instructions <set|show> [text]``."""
        subcmd = (parsed.instructions_subcommand or "").lower()
        if subcmd == "set":
            if not parsed.query:
                await update.message.reply_text("Usage: /instructions set <text>")
                return
            result = await self.runtime_client.set_instructions(
                tenant_id=tenant_id, text=parsed.query
            )
            if result.get("status") == "error":
                await update.message.reply_text(
                    f"Failed to set instructions: {result.get('message', '')}"
                )
            else:
                await update.message.reply_text("Instructions updated.")
        elif subcmd == "show":
            result = await self.runtime_client.get_instructions(tenant_id=tenant_id)
            if result.get("status") == "error":
                await update.message.reply_text("No instructions set for this tenant.")
            else:
                text = result.get("text", "")
                await update.message.reply_text(
                    f"Current instructions:\n\n{text}" if text else "(empty)"
                )
        else:
            await update.message.reply_text(
                "Unknown /instructions subcommand. Try: set, show."
            )

    async def _handle_memories_command(
        self, update: Update, parsed, tenant_id: str
    ) -> None:
        """Handle ``/memories <list|clear> [filter]``."""
        subcmd = (parsed.memories_subcommand or "").lower()
        if subcmd == "list":
            # Optional "agent=<name>" filter
            agent_name = None
            if parsed.query and parsed.query.startswith("agent="):
                agent_name = parsed.query[len("agent=") :].strip() or None
            result = await self.runtime_client.list_memories(
                tenant_id=tenant_id, agent_name=agent_name
            )
            count = result.get("count", 0)
            await update.message.reply_text(
                f"Found {count} memorie(s) for tenant {tenant_id}"
                + (f" (agent={agent_name})" if agent_name else "")
                + "."
            )
        elif subcmd == "clear":
            category = parsed.query.strip() or None
            result = await self.runtime_client.clear_memories(
                tenant_id=tenant_id, category=category
            )
            if result.get("status") == "error":
                await update.message.reply_text(
                    f"Clear failed: {result.get('message', '')}"
                )
            elif category:
                deleted = result.get("deleted", 0)
                await update.message.reply_text(
                    f"Cleared {deleted} '{category}' memories."
                )
            else:
                await update.message.reply_text("Cleared all user memories.")
        else:
            await update.message.reply_text(
                "Unknown /memories subcommand. Try: list, clear."
            )

    async def _handle_jobs_command(
        self, update: Update, parsed, tenant_id: str
    ) -> None:
        """Handle ``/jobs <list|create|delete> [args]``.

        ``/jobs create`` parses ``"<cron schedule>" <query>`` from
        ``parsed.query`` — the schedule must be quoted because cron strings
        contain spaces.
        """
        subcmd = (parsed.jobs_subcommand or "").lower()
        if subcmd == "list":
            result = await self.runtime_client.list_jobs(tenant_id=tenant_id)
            jobs = result.get("jobs", [])
            if not jobs:
                await update.message.reply_text("No jobs scheduled.")
                return
            lines = [
                f"- {j.get('name', '?')} ({j.get('schedule', '?')}) "
                f"[{j.get('job_id', '?')}]"
                for j in jobs
            ]
            await update.message.reply_text(
                f"Scheduled jobs ({len(jobs)}):\n" + "\n".join(lines)
            )
        elif subcmd == "create":
            schedule, name, query = self._parse_jobs_create_args(parsed.query)
            if not schedule or not query:
                await update.message.reply_text(
                    'Usage: /jobs create "<cron>" <query>\n'
                    'Example: /jobs create "0 9 * * 1" weekly AI news'
                )
                return
            result = await self.runtime_client.create_job(
                tenant_id=tenant_id,
                name=name,
                schedule=schedule,
                query=query,
            )
            if result.get("status") == "error":
                await update.message.reply_text(
                    f"Job create failed: {result.get('message', '')}"
                )
            else:
                await update.message.reply_text(
                    f"Created job '{result.get('name')}' "
                    f"({result.get('job_id')}) on schedule '{schedule}'."
                )
        elif subcmd == "delete":
            job_id = parsed.query.strip()
            if not job_id:
                await update.message.reply_text("Usage: /jobs delete <job_id>")
                return
            result = await self.runtime_client.delete_job(
                tenant_id=tenant_id, job_id=job_id
            )
            if result.get("status") == "error":
                await update.message.reply_text(
                    f"Delete failed: {result.get('message', '')}"
                )
            else:
                await update.message.reply_text(f"Deleted job {job_id}.")
        else:
            await update.message.reply_text(
                "Unknown /jobs subcommand. Try: list, create, delete."
            )

    @staticmethod
    def _parse_jobs_create_args(text: str) -> tuple:
        """Parse ``"<cron>" <query>`` into (schedule, name, query).

        The schedule MUST be wrapped in double quotes because cron strings
        contain spaces. ``name`` is derived from the first 30 chars of the
        query for convenience. Returns (None, None, None) on parse failure.
        """
        if not text:
            return None, None, None
        text = text.strip()
        if not text.startswith('"'):
            return None, None, None
        end_quote = text.find('"', 1)
        if end_quote < 0:
            return None, None, None
        schedule = text[1:end_quote].strip()
        query = text[end_quote + 1 :].strip()
        if not schedule or not query:
            return None, None, None
        name = query[:30].strip()
        return schedule, name, query

    def build_app(self) -> Application:
        """Build the Telegram Application with handlers.

        Every slash-command family needs its own CommandHandler: the text
        MessageHandler filters with ``~filters.COMMAND``, so a command
        without one matches nothing and Telegram silently drops it.
        Updates process concurrently so one slow agent dispatch cannot
        stall every other chat behind it.
        """
        builder = Application.builder().token(self.bot_token).concurrent_updates(32)
        self._app = builder.build()

        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))

        for command in [
            "search",
            "summarize",
            "report",
            "research",
            "code",
            "wiki",
            "instructions",
            "memories",
            "jobs",
        ]:
            self._app.add_handler(CommandHandler(command, self._handle_message))

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO | filters.VIDEO, self._handle_message)
        )
        self._app.add_error_handler(self._handle_error)

        return self._app

    async def _handle_error(self, update: object, context) -> None:
        """Tell the user a handler failed instead of leaving silence.

        Without an error handler, an exception in any handler (runtime
        unreachable, malformed response) is only logged server-side and the
        user sees "typing…" followed by nothing.
        """
        logger.error("Handler error: %s", context.error, exc_info=context.error)
        message = getattr(update, "effective_message", None)
        if message is None:
            return
        try:
            await message.reply_text(
                "Something went wrong handling that — please try again."
            )
        except Exception:  # noqa: BLE001 — the notice itself is best-effort
            logger.exception("Failed to send error notice")

    async def _outbound_drain_loop(self) -> None:
        """Deliver messages the runtime enqueued for this tenant's chats.

        Every ``self._outbound_poll_seconds`` the loop drains the runtime's
        outbound queue and sends each message via the bot. A drain failure
        (runtime blip) is logged and retried next tick; a malformed message
        is logged and skipped; a per-message send failure is logged and
        skipped so one bad chat never stops the others or the loop. Runs
        until cancelled on shutdown.
        """
        while True:
            try:
                messages = await self.runtime_client.drain_outbound()
            except Exception as exc:  # noqa: BLE001 — retry next tick, never die
                logger.error("Outbound drain failed: %s", exc)
                messages = []
            for msg in messages or []:
                if not isinstance(msg, dict):
                    logger.error("Malformed outbound message skipped: %r", msg)
                    continue
                chat_id = msg.get("chat_id")
                text = msg.get("text")
                if not chat_id or not text:
                    continue
                try:
                    await self._app.bot.send_message(chat_id=chat_id, text=text)
                except Exception as exc:  # noqa: BLE001 — isolate one bad chat
                    logger.error("Outbound send to chat %s failed: %s", chat_id, exc)
            await asyncio.sleep(self._outbound_poll_seconds)

    async def run_polling(self) -> None:
        """Run in long-polling mode (development)."""
        app = self.build_app()
        logger.info("Starting Telegram bot in polling mode")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        drain_task = asyncio.create_task(self._outbound_drain_loop())
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            await self.runtime_client.close()

    async def run_webhook(self) -> None:
        """Run in webhook mode (production)."""
        app = self.build_app()
        logger.info(f"Starting Telegram bot in webhook mode at {self.webhook_url}")
        await app.initialize()
        await app.start()
        # start_webhook BOTH registers the webhook with Telegram AND binds the
        # HTTP server that receives updates. set_webhook alone registered the
        # URL but served nothing, so webhook mode received zero messages.
        await app.updater.start_webhook(
            listen=self.webhook_listen,
            port=self.webhook_port,
            url_path=self.webhook_path,
            webhook_url=self.webhook_url,
            drop_pending_updates=True,
        )

        drain_task = asyncio.create_task(self._outbound_drain_loop())
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            await app.updater.stop()
            # delete_webhook talks to Telegram — if it fails during shutdown
            # the local teardown (stop/shutdown/close) must still run, or
            # the app and httpx client leak on every failed shutdown.
            try:
                await app.bot.delete_webhook()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                logger.error("delete_webhook failed during shutdown: %s", exc)
            await app.stop()
            await app.shutdown()
            await self.runtime_client.close()

    async def _log_runtime_reachability(self) -> bool:
        """Probe the runtime once at startup and log the outcome.

        A fail-fast diagnostic, not a hard gate: the gateway must still start
        during a runtime deploy so it is ready when the runtime comes up.
        Returns whether the runtime was reachable.
        """
        reachable = await self.runtime_client.health()
        if reachable:
            logger.info("Runtime reachable at %s", self.runtime_client.runtime_url)
        else:
            logger.warning(
                "Runtime not reachable at %s at startup — messages degrade until "
                "it is available",
                self.runtime_client.runtime_url,
            )
        return reachable

    async def run(self) -> None:
        """Run the gateway in the configured mode.

        Installs a SIGTERM handler that cancels this task: orchestrators
        stop containers with SIGTERM, whose default disposition kills the
        process without running any ``finally`` — the webhook stayed
        registered at a dead endpoint and nothing was closed. Cancellation
        routes shutdown through the same path as Ctrl+C.
        """
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
        await self._log_runtime_reachability()
        try:
            if self.mode == "webhook":
                await self.run_webhook()
            elif self.mode == "polling":
                await self.run_polling()
            else:
                # A typo'd GATEWAY_MODE silently running polling binds no
                # webhook server — the deployment receives zero messages
                # with no error.
                raise ValueError(
                    f"Unknown gateway mode {self.mode!r}; "
                    f"expected 'polling' or 'webhook'"
                )
        finally:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signal.SIGTERM)


def main():
    """CLI entry point for the messaging gateway."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
        sys.exit(1)

    def _env_number(name: str, default: str, cast):
        raw = os.environ.get(name, default)
        try:
            return cast(raw)
        except ValueError:
            logger.error("Invalid %s=%r — must be a number", name, raw)
            sys.exit(1)

    runtime_url = os.environ.get("RUNTIME_URL", "http://localhost:28000")
    mode = os.environ.get("GATEWAY_MODE", "polling")
    webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "")
    webhook_listen = os.environ.get("GATEWAY_WEBHOOK_LISTEN", "0.0.0.0")
    webhook_port = _env_number("GATEWAY_WEBHOOK_PORT", "8443", int)
    webhook_path = os.environ.get("GATEWAY_WEBHOOK_PATH", "")
    outbound_poll_seconds = _env_number("GATEWAY_OUTBOUND_POLL_SECONDS", "5", float)

    if mode == "webhook" and not webhook_url:
        logger.error("TELEGRAM_WEBHOOK_URL required for webhook mode")
        sys.exit(1)

    gateway = MessagingGateway(
        bot_token=bot_token,
        runtime_url=runtime_url,
        mode=mode,
        webhook_url=webhook_url,
        webhook_listen=webhook_listen,
        webhook_port=webhook_port,
        webhook_path=webhook_path,
        outbound_poll_seconds=outbound_poll_seconds,
    )

    asyncio.run(gateway.run())


if __name__ == "__main__":
    main()
