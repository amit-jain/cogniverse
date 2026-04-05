"""
WikiManager — persists wiki pages to Vespa and maintains a per-tenant index.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from cogniverse_agents.wiki.wiki_schema import WikiIndex, WikiPage, generate_slug

logger = logging.getLogger(__name__)

_AUTO_FILE_AGENTS = {"detailed_report_agent", "deep_research_agent"}
_CONTENT_SEPARATOR = "\n\n---\n\n"


class WikiManager:
    """Manages wiki knowledge pages for a single tenant, backed by Vespa."""

    def __init__(self, backend: Any, tenant_id: str, schema_name: str) -> None:
        """
        Args:
            backend: VespaSearchBackend instance (provides get_document, search).
            tenant_id: Tenant identifier (e.g. "acme:production").
            schema_name: Vespa schema name for this tenant's wiki pages
                         (e.g. "wiki_pages_acme_production").
        """
        self._backend = backend
        self._tenant_id = tenant_id
        self._schema_name = schema_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _should_auto_file(
        self, entities: List[str], agent_name: str, turn_count: int
    ) -> bool:
        """Return True when this interaction is substantial enough to auto-file."""
        if len(entities) >= 3:
            return True
        if agent_name in _AUTO_FILE_AGENTS:
            return True
        if turn_count >= 4:
            return True
        return False

    def save_session(
        self,
        query: str,
        response: str,
        entities: List[str],
        agent_name: str,
        sources: Optional[List[str]] = None,
    ) -> WikiPage:
        """Persist an agent interaction as a session wiki page.

        For each entity, a topic page is upserted and its doc_id collected as a
        cross-reference on the session page.

        Returns the saved session WikiPage.
        """
        sources = sources or []

        # Upsert a topic page for every entity; collect their doc_ids.
        cross_refs: List[str] = []
        for entity in entities:
            topic = self._get_or_create_topic(
                entity=entity,
                new_content=response,
                sources=sources,
            )
            cross_refs.append(topic.doc_id)

        # Build the session page.
        session = WikiPage(
            tenant_id=self._tenant_id,
            page_type="session",
            title=f"Session — {query[:60]}",
            content=response,
            entities=entities,
            sources=sources,
            cross_references=cross_refs,
            query=query,
            agent_used=agent_name,
        )
        embedding = self._generate_embedding(response)
        self._feed_page(session, embedding)

        self._rebuild_index()
        return session

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Full-text search over wiki pages.

        Returns a list of dicts with doc_id, title, content, page_type, score.
        """
        try:
            results = self._backend.search(
                {
                    "query": query,
                    "type": "wiki",
                    "top_k": top_k,
                    "schema": self._schema_name,
                }
            )
        except Exception:
            logger.exception("Wiki search failed")
            return []

        out = []
        for r in results:
            doc = r.document
            out.append(
                {
                    "doc_id": doc.metadata.get("doc_id", doc.id),
                    "title": doc.metadata.get("title", ""),
                    "content": doc.text_content or "",
                    "page_type": doc.metadata.get("page_type", ""),
                    "score": r.score,
                }
            )
        return out

    def get_topic(self, slug: str) -> Optional[Dict[str, Any]]:
        """Direct lookup of a topic page by slug.

        Returns a dict of the page fields, or None if not found.
        """
        safe = self._tenant_id.replace(":", "_")
        doc_id = f"wiki_topic_{safe}_{slug}"
        doc = self._get_document_http(doc_id)
        if doc is None:
            return None
        return {
            "doc_id": doc.metadata.get("doc_id", doc_id),
            "title": doc.metadata.get("title", ""),
            "content": doc.text_content or "",
            "page_type": doc.metadata.get("page_type", "topic"),
            "entities": doc.metadata.get("entities", "[]"),
            "sources": doc.metadata.get("sources", "[]"),
            "update_count": doc.metadata.get("update_count", 1),
        }

    def get_index(self) -> Optional[str]:
        """Return the rendered markdown of the wiki index document, or None."""
        safe = self._tenant_id.replace(":", "_")
        doc_id = f"wiki_index_{safe}"
        doc = self._get_document_http(doc_id)
        if doc is None:
            return None
        return doc.text_content or doc.metadata.get("content", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_topic(
        self, entity: str, new_content: str, sources: List[str]
    ) -> WikiPage:
        """Upsert a topic page for *entity*.

        If the page already exists its content is appended and update_count
        incremented.  The (new or updated) page is fed to Vespa.
        """
        slug = generate_slug(entity)
        safe = self._tenant_id.replace(":", "_")
        doc_id = f"wiki_topic_{safe}_{slug}"

        existing = self._get_document_http(doc_id)

        if existing is not None:
            # Merge: append new content and bump counter.
            old_content = existing.text_content or existing.metadata.get("content", "")
            merged_content = old_content + _CONTENT_SEPARATOR + new_content
            old_sources = existing.metadata.get("sources", "[]")
            try:
                import json

                existing_sources = json.loads(old_sources) if isinstance(old_sources, str) else old_sources
            except Exception:
                existing_sources = []
            merged_sources = list(dict.fromkeys(existing_sources + sources))
            update_count = int(existing.metadata.get("update_count", 1)) + 1

            page = WikiPage(
                tenant_id=self._tenant_id,
                page_type="topic",
                title=entity,
                content=merged_content,
                entities=[entity],
                sources=merged_sources,
                cross_references=[],
                update_count=update_count,
            )
            # Preserve original timestamps from the existing doc.
            page.created_at = existing.metadata.get("created_at", page.created_at)
        else:
            page = WikiPage(
                tenant_id=self._tenant_id,
                page_type="topic",
                title=entity,
                content=new_content,
                entities=[entity],
                sources=sources,
                cross_references=[],
            )

        embedding = self._generate_embedding(page.content)
        self._feed_page(page, embedding)
        return page

    def _rebuild_index(self) -> None:
        """Query all wiki pages for this tenant and rebuild the index document."""
        safe = self._tenant_id.replace(":", "_")

        try:
            results = self._backend.search(
                {
                    "query": f"tenant_id:{self._tenant_id}",
                    "type": "wiki",
                    "top_k": 500,
                    "schema": self._schema_name,
                }
            )
        except Exception:
            logger.warning("Could not fetch pages for index rebuild; skipping.")
            results = []

        index = WikiIndex(tenant_id=self._tenant_id)
        for r in results:
            doc = r.document
            page_type = doc.metadata.get("page_type", "topic")
            title = doc.metadata.get("title", "")
            slug = doc.metadata.get("slug", generate_slug(title))
            summary = (doc.text_content or "")[:120]
            index.add_page(page_type, title, slug, summary)

        index_content = index.render_markdown()
        index_doc_id = f"wiki_index_{safe}"
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/wiki_content/{self._schema_name}/docid/{index_doc_id}"
        )
        payload = {
            "fields": {
                "doc_id": index_doc_id,
                "tenant_id": self._tenant_id,
                "page_type": "index",
                "title": f"Wiki Index — {self._tenant_id}",
                "content": index_content,
                "slug": "wiki_index",
                "entities": "[]",
                "sources": "[]",
                "cross_references": "[]",
                "update_count": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        try:
            resp = requests.post(feed_url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(
                    "Index feed returned %s: %s", resp.status_code, resp.text[:200]
                )
        except Exception:
            logger.exception("Failed to feed wiki index document")

    def _get_document_http(self, doc_id: str) -> Optional[Any]:
        """Fetch a wiki document from Vespa via HTTP GET.

        Returns a simple object with text_content and metadata attributes,
        or None if not found. Uses direct HTTP instead of backend.get_document
        to avoid schema configuration requirements.
        """
        url = f"{self._backend._url}:{self._backend._port}"
        get_url = (
            f"{url}/document/v1/wiki_content/{self._schema_name}/docid/{doc_id}"
        )
        try:
            resp = requests.get(get_url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            fields = data.get("fields", {})

            class _Doc:
                pass

            doc = _Doc()
            doc.text_content = fields.get("content", "")
            doc.metadata = fields
            return doc
        except Exception:
            return None

    def _feed_page(self, page: WikiPage, embedding: List[float]) -> None:
        """Feed *page* to Vespa via the Document v1 HTTP API."""
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/wiki_content/{self._schema_name}/docid/{page.doc_id}"
        )
        payload = page.to_vespa_document()
        payload["fields"]["embedding"] = embedding

        try:
            resp = requests.post(feed_url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(
                    "Feed for %s returned %s: %s",
                    page.doc_id,
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:
            logger.exception("Failed to feed wiki page %s", page.doc_id)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a text embedding via Ollama nomic-embed-text.

        Falls back to a zero vector on any error so that feed operations are
        never blocked by embedding failures.
        """
        try:
            import ollama

            result = ollama.embed(model="nomic-embed-text", input=text)
            embeddings = result.get("embeddings") or result.get("embedding") or []
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return list(embeddings)
        except Exception:
            logger.warning("Embedding generation failed; using zero vector")
            return [0.0] * 768
