"""Per-conversation turn history in Mem0, keyed by (tenant_id, context_id).

Each turn is one memory under a dedicated conversation partition, stored
verbatim (``infer=False``) so the retrieval filter is reliable — the LLM
extraction pass must not reword a tagged turn. A dispatch reloads a
context's recent turns before running the agent and appends the two new
turns after.

Retrieval enumerates the partition and filters by metadata (the same
reliable path ``UserTenantMapper`` uses) rather than semantic search,
which could miss turns past the ``top_k`` window or surface another
context's turns. Reads and writes RAISE on a backend outage — the caller
(the agent dispatcher) decides that history is enrichment and degrades to
no-history, so the outage is a logged degrade there, not a silent [] here.
"""

import logging
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

CONVERSATION_AGENT_NAME = "_conversation"
MAX_HISTORY_TURNS = 10


class ConversationStore:
    """Loads and appends per-context conversation turns via Mem0."""

    def __init__(self, memory_manager, tenant_id: str):
        self._memory = memory_manager
        self._tenant_id = tenant_id

    def get_history(
        self, context_id: str, max_turns: int = MAX_HISTORY_TURNS
    ) -> List[Dict[str, str]]:
        """Return the most recent turns for ``context_id`` in order.

        ``[{"role": "user"|"assistant", "content": ...}]``, oldest first,
        capped at ``max_turns``. The role comes from metadata and the
        content is the stored text with its ``[ctx:id] [role] `` prefix
        stripped, so neither depends on parsing the free text for the role.
        """
        rows = self._memory.get_all_memories(
            tenant_id=self._tenant_id,
            agent_name=CONVERSATION_AGENT_NAME,
        )
        ctx = str(context_id)
        prefix = f"[ctx:{ctx}] "
        collected: List = []
        for row in rows:
            meta = row.get("metadata") or {}
            if meta.get("type") != "conversation":
                continue
            if str(meta.get("context_id")) != ctx:
                continue
            role = meta.get("role")
            if role not in ("user", "assistant"):
                continue
            text = row.get("memory", "")
            if text.startswith(prefix):
                text = text[len(prefix) :]
            tag = f"[{role}] "
            if text.startswith(tag):
                text = text[len(tag) :]
            collected.append((meta.get("seq", 0.0), {"role": role, "content": text}))

        collected.sort(key=lambda item: item[0])
        return [turn for _seq, turn in collected[-max_turns:]]

    def store_turn(self, context_id: str, role: str, content: str) -> None:
        """Append one turn. ``seq`` (wall-clock) orders turns on read."""
        ctx = str(context_id)
        self._memory.add_memory(
            content=f"[ctx:{ctx}] [{role}] {content}",
            tenant_id=self._tenant_id,
            agent_name=CONVERSATION_AGENT_NAME,
            metadata={
                "type": "conversation",
                "context_id": ctx,
                "role": role,
                "seq": time.time(),
            },
            infer=False,
        )
