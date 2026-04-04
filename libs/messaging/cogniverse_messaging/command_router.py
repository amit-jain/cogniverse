"""Command routing for messaging gateway.

Maps slash commands and plain text to agent names.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

AGENT_COMMANDS = {
    "/search": "search_agent",
    "/summarize": "summarizer_agent",
    "/report": "detailed_report_agent",
    "/research": "deep_research_agent",
    "/code": "coding_agent",
}

HELP_TEXT = """Available commands:

/search <query> — Search videos, images, documents
/summarize <query> — Get a summary of search results
/report <query> — Generate a detailed analysis report
/research <query> — Deep research across multiple sources
/code <query> — Code search and analysis
/help — Show this message

Or just send a message — it will be automatically routed to the best agent.

Send images or videos to search for similar content."""


@dataclass
class ParsedCommand:
    """Result of parsing a user message."""

    agent_name: str
    query: str
    is_command: bool
    is_registration: bool = False
    registration_token: Optional[str] = None
    is_help: bool = False
    has_media: bool = False
    media_type: Optional[str] = None
    media_file_id: Optional[str] = None


def parse_message(
    text: Optional[str] = None,
    has_photo: bool = False,
    has_video: bool = False,
    photo_file_id: Optional[str] = None,
    video_file_id: Optional[str] = None,
) -> ParsedCommand:
    """Parse a message into a routable command.

    Args:
        text: Message text (may be None for media-only messages)
        has_photo: Whether message contains a photo
        has_video: Whether message contains a video
        photo_file_id: Telegram file ID for photo
        video_file_id: Telegram file ID for video
    """
    if has_photo or has_video:
        media_type = "photo" if has_photo else "video"
        file_id = photo_file_id if has_photo else video_file_id
        query = text or f"Find similar {media_type} content"
        return ParsedCommand(
            agent_name="search_agent",
            query=query,
            is_command=False,
            has_media=True,
            media_type=media_type,
            media_file_id=file_id,
        )

    if not text:
        return ParsedCommand(
            agent_name="routing_agent",
            query="",
            is_command=False,
        )

    text = text.strip()

    if text.startswith("/start"):
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) > 1 else None
        return ParsedCommand(
            agent_name="",
            query="",
            is_command=True,
            is_registration=True,
            registration_token=token,
        )

    if text == "/help":
        return ParsedCommand(
            agent_name="",
            query="",
            is_command=True,
            is_help=True,
        )

    for command, agent in AGENT_COMMANDS.items():
        if text.startswith(command):
            query = text[len(command):].strip()
            if not query:
                return ParsedCommand(
                    agent_name=agent,
                    query="",
                    is_command=True,
                    is_help=True,
                )
            return ParsedCommand(
                agent_name=agent,
                query=query,
                is_command=True,
            )

    return ParsedCommand(
        agent_name="routing_agent",
        query=text,
        is_command=False,
    )
