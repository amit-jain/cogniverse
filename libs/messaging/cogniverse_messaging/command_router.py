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

FAMILY_COMMANDS = {
    "/wiki": "is_wiki",
    "/instructions": "is_instructions",
    "/memories": "is_memories",
    "/jobs": "is_jobs",
}

HELP_TEXT = """Available commands:

/search <query> — Search videos, images, documents
/summarize <query> — Get a summary of search results
/report <query> — Generate a detailed analysis report
/research <query> — Deep research across multiple sources
/code <query> — Code search and analysis
/wiki save — Save the current session to the wiki
/wiki search <query> — Search the wiki knowledge base
/wiki topic <name> — Look up a topic page by name
/wiki index — Show the wiki index
/wiki lint — Check wiki for orphan, stale, or empty pages
/instructions set <text> — Set custom agent instructions for your tenant
/instructions show — Show current tenant instructions
/memories list — List memories (add agent=<name> to filter)
/memories clear strategies — Clear strategy learner memories
/jobs list — List scheduled agent jobs
/jobs create "<cron>" <query> — Create a new scheduled job
/jobs delete <job_id> — Delete a scheduled job
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
    is_wiki: bool = False
    wiki_subcommand: Optional[str] = None
    is_instructions: bool = False
    instructions_subcommand: Optional[str] = None
    is_memories: bool = False
    memories_subcommand: Optional[str] = None
    is_jobs: bool = False
    jobs_subcommand: Optional[str] = None


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
        # A photo is searchable by its CONTENT: the image agent embeds the photo
        # itself and matches it against stored image embeddings. A video file has
        # no single query embedding, so a video keeps caption-text search on the
        # video index rather than pretending to match its frames.
        if has_photo:
            agent_name = "image_search_agent"
            query = text or "Find visually similar images"
        else:
            agent_name = "search_agent"
            query = text or "Find similar video content"
        return ParsedCommand(
            agent_name=agent_name,
            query=query,
            is_command=False,
            has_media=True,
            media_type=media_type,
            media_file_id=file_id,
        )

    if not text:
        return ParsedCommand(
            agent_name="gateway_agent",
            query="",
            is_command=False,
        )

    text = text.strip()

    # Match the command TOKEN exactly, with any "@botname" suffix stripped —
    # Telegram appends it in group chats. A startswith match bled the suffix
    # into the query ("/search@bot cats" → "@bot cats") and misrouted every
    # prefix collision ("/codebase scan" → coding_agent with "base scan").
    head, _, rest = text.partition(" ")
    command = head.split("@", 1)[0].lower() if head.startswith("/") else ""
    rest = rest.strip()

    if command == "/start":
        return ParsedCommand(
            agent_name="",
            query="",
            is_command=True,
            is_registration=True,
            registration_token=rest or None,
        )

    if command == "/help":
        return ParsedCommand(
            agent_name="",
            query="",
            is_command=True,
            is_help=True,
        )

    if command in FAMILY_COMMANDS:
        parts = rest.split(maxsplit=1)
        subcmd = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        return ParsedCommand(
            agent_name="",
            query=arg,
            is_command=True,
            **{
                FAMILY_COMMANDS[command]: True,
                f"{command[1:]}_subcommand": subcmd,
            },
        )

    agent = AGENT_COMMANDS.get(command)
    if agent:
        if not rest:
            return ParsedCommand(
                agent_name=agent,
                query="",
                is_command=True,
                is_help=True,
            )
        return ParsedCommand(
            agent_name=agent,
            query=rest,
            is_command=True,
        )

    return ParsedCommand(
        agent_name="gateway_agent",
        query=text,
        is_command=False,
    )
