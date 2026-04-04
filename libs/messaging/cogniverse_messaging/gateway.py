"""Messaging Gateway — main entry point.

Runs as a separate service. Supports webhook (production) and
long-polling (development) modes for Telegram.

Usage:
    python -m cogniverse_messaging.gateway
    GATEWAY_MODE=webhook python -m cogniverse_messaging.gateway
"""

import asyncio
import logging
import os
import sys

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
    formatted responses back. Manages user registration via invite
    tokens and conversation history via Mem0.
    """

    def __init__(
        self,
        bot_token: str,
        runtime_url: str,
        mode: str = "polling",
        webhook_url: str = "",
        memory_manager=None,
        config_manager=None,
    ):
        self.bot_token = bot_token
        self.runtime_url = runtime_url
        self.mode = mode
        self.webhook_url = webhook_url
        self.runtime_client = RuntimeClient(runtime_url)

        self._memory_manager = memory_manager
        self._config_manager = config_manager
        self._token_manager = None
        self._user_mapper = None
        self._app: Application = None

    def _init_auth(self):
        """Lazy-initialize auth components."""
        if self._token_manager is None and self._config_manager is not None:
            from cogniverse_messaging.auth import (
                InviteTokenManager,
                UserTenantMapper,
            )

            self._token_manager = InviteTokenManager(self._config_manager)
            if self._memory_manager is not None:
                self._user_mapper = UserTenantMapper(self._memory_manager)

    def _get_conversation_manager(self, tenant_id: str):
        """Create a ConversationManager for a tenant."""
        if self._memory_manager is None:
            return None
        from cogniverse_messaging.conversation import ConversationManager

        return ConversationManager(self._memory_manager, tenant_id)

    async def _handle_start(self, update: Update, context) -> None:
        """Handle /start command — user registration."""
        self._init_auth()
        text = update.message.text or ""
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) > 1 else None

        if not token:
            await update.message.reply_text(
                "Welcome to Cogniverse!\n\n"
                "To register, send:\n/start <invite_token>\n\n"
                "Get a token from your admin."
            )
            return

        if self._token_manager is None:
            await update.message.reply_text(
                "Registration not available — config_manager not initialized."
            )
            return

        tenant_id = self._token_manager.validate_token(token)
        if not tenant_id:
            await update.message.reply_text(format_invalid_token())
            return

        user_id = str(update.effective_user.id)

        if self._user_mapper:
            self._user_mapper.register_user("telegram", user_id, tenant_id)

        self._token_manager.mark_token_used(token, tenant_id)
        await update.message.reply_text(format_registration_success(tenant_id))

    async def _handle_help(self, update: Update, context) -> None:
        """Handle /help command."""
        await update.message.reply_text(format_help())

    async def _handle_message(self, update: Update, context) -> None:
        """Handle all messages — text, commands, and media."""
        self._init_auth()

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        tenant_id = None
        if self._user_mapper:
            tenant_id = self._user_mapper.get_tenant_id("telegram", user_id)

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

        if not parsed.query:
            await update.message.reply_text(
                "Please provide a query. Send /help for usage."
            )
            return

        conv_manager = self._get_conversation_manager(tenant_id)
        history = []
        if conv_manager:
            history = conv_manager.get_history(chat_id)
            conv_manager.store_turn(chat_id, "user", parsed.query)

        await update.message.chat.send_action("typing")

        agent_context = {}
        if parsed.has_media and parsed.media_file_id:
            agent_context["media_type"] = parsed.media_type
            agent_context["media_file_id"] = parsed.media_file_id

        response = await self.runtime_client.dispatch_agent(
            agent_name=parsed.agent_name,
            query=parsed.query,
            tenant_id=tenant_id,
            context_id=chat_id,
            conversation_history=history,
            context=agent_context,
        )

        messages = format_agent_response(response)
        for chunk in messages:
            await update.message.reply_text(chunk)

        if conv_manager:
            assistant_text = response.get("message", "")
            if assistant_text:
                conv_manager.store_turn(chat_id, "assistant", assistant_text)

    def build_app(self) -> Application:
        """Build the Telegram Application with handlers."""
        builder = Application.builder().token(self.bot_token)
        self._app = builder.build()

        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))

        for command in ["search", "summarize", "report", "research", "code"]:
            self._app.add_handler(
                CommandHandler(command, self._handle_message)
            )

        self._app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND, self._handle_message
            )
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO | filters.VIDEO, self._handle_message)
        )

        return self._app

    async def run_polling(self) -> None:
        """Run in long-polling mode (development)."""
        app = self.build_app()
        logger.info("Starting Telegram bot in polling mode")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
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
        await app.bot.set_webhook(url=self.webhook_url)

        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await app.bot.delete_webhook()
            await app.stop()
            await app.shutdown()
            await self.runtime_client.close()

    async def run(self) -> None:
        """Run the gateway in the configured mode."""
        if self.mode == "webhook":
            await self.run_webhook()
        else:
            await self.run_polling()


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

    runtime_url = os.environ.get("RUNTIME_URL", "http://localhost:28000")
    mode = os.environ.get("GATEWAY_MODE", "polling")
    webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "")

    if mode == "webhook" and not webhook_url:
        logger.error("TELEGRAM_WEBHOOK_URL required for webhook mode")
        sys.exit(1)

    gateway = MessagingGateway(
        bot_token=bot_token,
        runtime_url=runtime_url,
        mode=mode,
        webhook_url=webhook_url,
    )

    asyncio.run(gateway.run())


if __name__ == "__main__":
    main()
