"""Telegram message handling — parse updates, format responses, chunk messages."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096
MAX_RESULTS_PER_MESSAGE = 5


def format_agent_response(response: Dict[str, Any]) -> List[str]:
    """Format an agent response into Telegram-friendly message chunks.

    Returns a list of strings, each within MAX_MESSAGE_LENGTH.
    """
    if response.get("status") == "error":
        error_msg = response.get("message", "An error occurred")
        return [f"Error: {error_msg}"]

    parts = []

    message = response.get("message", "")
    if message:
        parts.append(message)

    results = response.get("results", [])
    if results:
        results_text = _format_results(results[:MAX_RESULTS_PER_MESSAGE])
        parts.append(results_text)

        total = response.get("results_count", len(results))
        if total > MAX_RESULTS_PER_MESSAGE:
            parts.append(
                f"Showing {MAX_RESULTS_PER_MESSAGE} of {total} results."
            )

    if not parts:
        return ["No results found."]

    full_text = "\n\n".join(parts)
    return chunk_message(full_text)


def _format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results as a numbered list."""
    lines = []
    for i, result in enumerate(results, 1):
        title = (
            result.get("video_title")
            or result.get("title")
            or result.get("source_id", "Unknown")
        )
        score = result.get("score", result.get("relevance_score"))
        description = (
            result.get("segment_description")
            or result.get("description", "")
        )

        line = f"{i}. {title}"
        if score is not None:
            line += f" ({float(score):.0%})"
        if description:
            desc_short = (
                description[:100] + "..."
                if len(description) > 100
                else description
            )
            line += f"\n   {desc_short}"

        lines.append(line)

    return "\n".join(lines)


def chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split text into chunks respecting Telegram's message length limit.

    Splits at newline boundaries when possible to avoid breaking mid-sentence.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            split_at = max_length

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


def format_help() -> str:
    """Return help text."""
    from cogniverse_messaging.command_router import HELP_TEXT

    return HELP_TEXT


def format_registration_success(tenant_id: str) -> str:
    """Format successful registration message."""
    return (
        f"Registered as {tenant_id}.\n\n"
        f"Send /help to see available commands, or just type a question."
    )


def format_registration_required() -> str:
    """Format message for unregistered users."""
    return (
        "You need to register first.\n\n"
        "Get an invite token from your admin, then send:\n"
        "/start <token>"
    )


def format_invalid_token() -> str:
    """Format message for invalid/expired tokens."""
    return (
        "Invalid or expired invite token.\n\n"
        "Please request a new token from your admin."
    )
