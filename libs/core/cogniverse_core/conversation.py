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

A turn whose assistant reply could not be stored keeps its user turn and
carries an ``assistant_missing`` marker row in the reply's place. The
marker is durable, so a half-turn is findable after a restart, and
``get_history`` never renders it: the next turn reads an unanswered user
message rather than prose no agent produced.
"""

import json
import logging
import time
from typing import Dict, List, Tuple

import httpx
import requests

logger = logging.getLogger(__name__)

CONVERSATION_AGENT_NAME = "_conversation"
MAX_HISTORY_TURNS = 10

# Roles a loaded history renders to the agent.
RENDERED_TURN_ROLES = ("user", "assistant")

# The turn whose reply was lost. Never rendered as history; readable through
# get_missing_assistant_markers.
ASSISTANT_MISSING_ROLE = "assistant_missing"
ASSISTANT_MISSING_PREFIX = "assistant turn not persisted: "

KNOWN_TURN_ROLES = RENDERED_TURN_ROLES + (ASSISTANT_MISSING_ROLE,)

# Backend statuses a turn write may be retried on: the request was refused for
# load or a server-side fault, never because the document itself was rejected.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 507})

# Transport and timeout failures a turn write may be retried on, by TYPE — the
# write never reached a verdict. The ingestion client raises the builtin
# ConnectionError for an unreachable backend, the DenseOn embedder raises
# requests' transport errors, and pyvespa feeds over both httpx and requests.
_TRANSIENT_WRITE_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.TransportError,
    requests.ConnectionError,
    requests.Timeout,
)


def is_transient_turn_write_error(exc: BaseException) -> bool:
    """True when a failed turn write can be retried.

    An error carrying a response status is classified on that status alone, so
    a document the backend refused is never retried even when its exception
    type belongs to a transport family.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS_CODES
    return isinstance(exc, _TRANSIENT_WRITE_ERRORS)


class ConversationStore:
    """Loads and appends per-context conversation turns via Mem0."""

    def __init__(self, memory_manager, tenant_id: str):
        self._memory = memory_manager
        self._tenant_id = tenant_id

    def get_history(
        self, context_id: str, max_turns: int = MAX_HISTORY_TURNS
    ) -> List[Dict[str, str]]:
        """Return the most recent renderable turns for ``context_id`` in order.

        ``[{"role": "user"|"assistant", "content": ...}]``, oldest first,
        capped at ``max_turns``. An ``assistant_missing`` marker is not a
        turn an agent may read, so it is left out and the user turn it
        belongs to reads as unanswered.
        """
        turns = [
            turn
            for turn in self._stored_turns(context_id)
            if turn["role"] in RENDERED_TURN_ROLES
        ]
        return turns[-max_turns:]

    def get_missing_assistant_markers(self, context_id: str) -> List[Dict[str, str]]:
        """Return this context's half-turn markers, oldest first."""
        return [
            turn
            for turn in self._stored_turns(context_id)
            if turn["role"] == ASSISTANT_MISSING_ROLE
        ]

    def _stored_turns(self, context_id: str) -> List[Dict[str, str]]:
        """Every stored turn for ``context_id``, ordered by ``seq``.

        The role comes from metadata and the content is the stored text with
        its ``[ctx:id] [role] `` prefix stripped, so neither depends on
        parsing the free text for the role.
        """
        ctx = str(context_id)
        # Narrow to this context server-side on the stamped session key, then
        # walk every matching turn (limit=None). Enumerating the partition and
        # filtering in Python sees only the newest 100 rows, so a busy
        # neighbour buries this context and its history reloads empty.
        rows = self._memory.get_all_memories(
            tenant_id=self._tenant_id,
            agent_name=CONVERSATION_AGENT_NAME,
            filters={"session_id": ctx},
            limit=None,
        )
        prefix = f"[ctx:{ctx}] "
        collected: List[Tuple[float, Dict[str, str]]] = []
        for row in rows:
            meta = row.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (ValueError, TypeError):
                    continue
            # Mem0 returns metadata as a dict (or the JSON string handled above).
            # A row whose metadata is any other shape is not one of ours — skip
            # it rather than let meta.get(...) raise and lose the whole history
            # to one malformed row.
            if not isinstance(meta, dict):
                continue
            if meta.get("type") != "conversation":
                continue
            if str(meta.get("context_id")) != ctx:
                continue
            # "role" is a Mem0-reserved metadata key (hoisted to a top-level
            # field and overwritten with the message role) — the turn's role
            # is stored under turn_role.
            role = meta.get("turn_role")
            if role not in KNOWN_TURN_ROLES:
                continue
            text = row.get("memory", "")
            if text.startswith(prefix):
                text = text[len(prefix) :]
            tag = f"[{role}] "
            if text.startswith(tag):
                text = text[len(tag) :]
            # seq orders turns on read; store_turn always writes a float. A row
            # whose seq is a foreign/corrupt non-number is dropped rather than
            # crash the sort — a mixed str/float key set is unorderable.
            try:
                seq = float(meta.get("seq", 0.0))
            except (ValueError, TypeError):
                continue
            collected.append((seq, {"role": role, "content": text}))

        collected.sort(key=lambda item: item[0])
        return [turn for _seq, turn in collected]

    def store_turn(self, context_id: str, role: str, content: str) -> None:
        """Append one turn. ``seq`` (wall-clock) orders turns on read."""
        if role not in KNOWN_TURN_ROLES:
            raise ValueError(
                f"unknown conversation turn role {role!r}; "
                f"expected one of {KNOWN_TURN_ROLES}"
            )
        ctx = str(context_id)
        self._memory.add_memory(
            content=f"[ctx:{ctx}] [{role}] {content}",
            tenant_id=self._tenant_id,
            agent_name=CONVERSATION_AGENT_NAME,
            metadata={
                "type": "conversation",
                "context_id": ctx,
                # session_id is a promoted Vespa field, so get_history filters
                # to this context server-side instead of scanning the whole
                # partition; context_id carries it.
                "session_id": ctx,
                # not "role": Mem0 reserves it and overwrites with the
                # message role, so the turn's role would be lost on read.
                "turn_role": role,
                "seq": time.time(),
            },
            infer=False,
        )

    def store_missing_assistant_marker(
        self, context_id: str, cause: BaseException
    ) -> None:
        """Mark this context's last user turn as one whose reply was lost.

        The content names the failure TYPE only — never its message, which
        can quote the user's text or the payload.
        """
        self.store_turn(
            context_id,
            ASSISTANT_MISSING_ROLE,
            f"{ASSISTANT_MISSING_PREFIX}{type(cause).__name__}",
        )
