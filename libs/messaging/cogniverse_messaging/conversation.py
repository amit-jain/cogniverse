"""Conversation history manager using Mem0.

Stores and retrieves conversation turns per chat_id using the same
Mem0MemoryManager that the dashboard uses. Each turn is stored as a
memory with the chat_id as context.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

GATEWAY_AGENT_NAME = "_messaging_gateway"
MAX_HISTORY_TURNS = 10


class ConversationManager:
    """Manages per-chat conversation history via Mem0."""

    def __init__(self, memory_manager, tenant_id: str):
        self.memory_manager = memory_manager
        self.tenant_id = tenant_id

    def get_history(
        self, chat_id: str, max_turns: int = MAX_HISTORY_TURNS
    ) -> List[Dict[str, str]]:
        """Load recent conversation turns for a chat.

        Returns list of {"role": "user"|"assistant", "content": "..."}.
        """
        if not self.memory_manager or self.memory_manager.memory is None:
            return []

        try:
            results = self.memory_manager.search_memory(
                query=f"conversation in chat {chat_id}",
                tenant_id=self.tenant_id,
                agent_name=GATEWAY_AGENT_NAME,
                top_k=max_turns,
            )

            # Parse by exact prefix, not substring: "[chat:1]" in memory also
            # matched chat 12's turns, and a turn whose CONTENT contains
            # "[user]" was misclassified regardless of its real role.
            chat_prefix = f"[chat:{chat_id}] "
            turns = []
            for r in results:
                memory = r.get("memory", "")
                if not memory.startswith(chat_prefix):
                    continue
                rest = memory[len(chat_prefix) :]
                for role in ("user", "assistant"):
                    role_prefix = f"[{role}] "
                    if rest.startswith(role_prefix):
                        turns.append(
                            {"role": role, "content": rest[len(role_prefix) :]}
                        )
                        break

            return turns

        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            return []

    def store_turn(self, chat_id: str, role: str, content: str) -> None:
        """Store a conversation turn.

        Args:
            chat_id: Telegram chat ID
            role: "user" or "assistant"
            content: Message content
        """
        if not self.memory_manager or self.memory_manager.memory is None:
            return

        memory_content = f"[chat:{chat_id}] [{role}] {content}"

        try:
            self.memory_manager.add_memory(
                content=memory_content,
                tenant_id=self.tenant_id,
                agent_name=GATEWAY_AGENT_NAME,
                metadata={
                    "type": "conversation",
                    "chat_id": str(chat_id),
                    "role": role,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {e}")
