"""
Wiki page dataclasses and index for the Cogniverse wiki knowledge layer.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def generate_slug(title: str) -> str:
    """Convert a title to a URL/doc-id safe slug.

    Normalizes unicode to ASCII, lowercases, replaces non-alphanumeric chars
    with underscores, and strips leading/trailing underscores.

    Examples:
        "Machine Learning" → "machine_learning"
        "What's New?"      → "whats_new"
        "Café au lait"     → "cafe_au_lait"
    """
    # Decompose unicode (e.g. é → e + combining accent) then encode to ASCII,
    # dropping any code-points that cannot be represented.
    normalized = unicodedata.normalize("NFD", title)
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    ascii_str = ascii_bytes.decode("ascii")

    lowered = ascii_str.lower()
    # Drop possessive/contraction apostrophes (and similar) before splitting on
    # word boundaries, so "what's" → "whats" rather than "what_s".
    stripped = re.sub(r"'", "", lowered)
    # Replace any run of non-alphanumeric characters with a single underscore
    slugged = re.sub(r"[^a-z0-9]+", "_", stripped)
    return slugged.strip("_")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_digits() -> str:
    """Return a compact UTC timestamp string containing only digits."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d%H%M%S%f")


def _safe_tenant(tenant_id: str) -> str:
    """Replace colons with underscores so tenant_id is safe for doc_id use."""
    return tenant_id.replace(":", "_")


@dataclass
class WikiPage:
    """A single wiki page — either a topic or a session."""

    tenant_id: str
    page_type: str  # "topic" | "session"
    title: str
    content: str
    entities: list
    sources: list
    cross_references: list
    query: Optional[str] = None
    agent_used: Optional[str] = None
    update_count: int = 1
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    # Internal: store timestamp once at creation so doc_id stays stable
    _session_ts: str = field(default_factory=_timestamp_digits, repr=False)

    @property
    def slug(self) -> str:
        return generate_slug(self.title)

    @property
    def doc_id(self) -> str:
        safe = _safe_tenant(self.tenant_id)
        if self.page_type == "topic":
            return f"wiki_topic_{safe}_{self.slug}"
        return f"wiki_session_{safe}_{self._session_ts}"

    def to_vespa_document(self) -> dict:
        """Return a Vespa-ready document dict with all fields.

        Lists are JSON-serialized to strings (matching the Vespa schema which
        stores entities/sources/cross_references as plain string fields).
        Optional fields that are None are omitted.
        """
        fields: dict = {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "page_type": self.page_type,
            "title": self.title,
            "content": self.content,
            "slug": self.slug,
            "entities": json.dumps(self.entities),
            "sources": json.dumps(self.sources),
            "cross_references": json.dumps(self.cross_references),
            "update_count": self.update_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.query is not None:
            fields["query"] = self.query
        if self.agent_used is not None:
            fields["agent_used"] = self.agent_used
        return {"fields": fields}


@dataclass
class WikiIndex:
    """Lightweight index of all wiki pages for a tenant."""

    tenant_id: str
    page_count: int = 0
    topic_count: int = 0
    session_count: int = 0
    _topics: list = field(default_factory=list)
    _sessions: list = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return f"wiki_index_{_safe_tenant(self.tenant_id)}"

    def add_page(self, page_type: str, title: str, slug: str, summary: str) -> None:
        """Register a page in the index and update counts."""
        entry = {"title": title, "slug": slug, "summary": summary}
        if page_type == "topic":
            self._topics.append(entry)
            self.topic_count += 1
        else:
            self._sessions.append(entry)
            self.session_count += 1
        self.page_count += 1

    def render_markdown(self) -> str:
        """Render the index as a human-readable markdown document."""
        lines = [f"# Wiki Index — {self.tenant_id}", ""]

        lines.append("## Topics")
        if self._topics:
            for entry in self._topics:
                lines.append(f"- **{entry['title']}**: {entry['summary']}")
        else:
            lines.append("_No topics yet._")

        lines.append("")
        lines.append("## Sessions")
        if self._sessions:
            for entry in self._sessions:
                lines.append(f"- **{entry['title']}**: {entry['summary']}")
        else:
            lines.append("_No sessions yet._")

        return "\n".join(lines)
