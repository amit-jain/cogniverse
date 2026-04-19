"""
WikiManager — persists wiki pages to Vespa and maintains a per-tenant index.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from cogniverse_agents.inference.rlm_inference import RLMInference
from cogniverse_agents.wiki.wiki_schema import WikiIndex, WikiPage, generate_slug
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

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
                         (e.g. "wiki_pages_acme_production"). Must NOT contain
                         colons — Vespa's /document/v1 URL uses ':' as a path
                         delimiter, so colons in the schema segment produce
                         "Illegal key-value pair" 400s on every feed. Use
                         ``backend.get_tenant_schema_name(tenant_id, base)``
                         to get the sanitized form.
        """
        if ":" in schema_name:
            raise ValueError(
                f"schema_name '{schema_name}' contains a colon — Vespa's "
                "/document/v1 URL cannot parse it. Pass a sanitized schema "
                "name (tenant_id's colon already replaced with underscore)."
            )
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
        import numpy as np

        query_vec = np.asarray(self._generate_embedding(query), dtype=np.float32)
        try:
            # ``hybrid`` combines closeness(embedding) with bm25(title/content).
            # Gives usable scores on the current backend whether or not the
            # nearestNeighbor YQL fast path fires for this schema.
            results = self._backend.search(
                {
                    "query": query,
                    "type": "wiki",
                    "top_k": top_k,
                    "schema": self._schema_name,
                    "tenant_id": self._tenant_id,
                    "strategy": "hybrid",
                    "query_embeddings": query_vec,
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

    def lint(self) -> Dict[str, Any]:
        """Analyse all wiki pages and report quality issues.

        Returns a dict with:
            orphan_pages  — topic pages with zero cross-references from any session
            stale_pages   — pages not updated in 30+ days
            empty_pages   — pages with content shorter than 50 chars
            total_pages   — total page count examined
            issues_found  — sum of all issues
        """
        import json as _json
        from datetime import timedelta

        try:
            results = self._backend.search(
                {
                    "query": f"tenant_id:{self._tenant_id}",
                    "type": "wiki",
                    "top_k": 500,
                    "schema": self._schema_name,
                    "tenant_id": self._tenant_id,
                    "strategy": "semantic_search",
                }
            )
        except Exception:
            logger.exception("Wiki lint: backend search failed")
            results = []

        # Collect all doc_ids referenced as cross_references in session pages.
        referenced_ids: set = set()
        all_pages = []
        for r in results:
            doc = r.document
            page_type = doc.metadata.get("page_type", "")
            if page_type == "session":
                raw = doc.metadata.get("cross_references", "[]")
                try:
                    refs = _json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    refs = []
                referenced_ids.update(refs)
            all_pages.append(doc)

        now = datetime.now(timezone.utc)
        stale_threshold = timedelta(days=30)

        orphan_pages = []
        stale_pages = []
        empty_pages = []

        for doc in all_pages:
            page_type = doc.metadata.get("page_type", "")
            doc_id = doc.metadata.get("doc_id", "")
            title = doc.metadata.get("title", "")
            content = doc.text_content or doc.metadata.get("content", "")

            # Skip index and session pages for orphan/empty checks.
            if page_type in ("index", "session"):
                continue

            # Orphan: topic page not referenced by any session.
            if page_type == "topic" and doc_id not in referenced_ids:
                orphan_pages.append({"doc_id": doc_id, "title": title})

            # Empty: content shorter than 50 chars.
            if len(content) < 50:
                empty_pages.append(
                    {"doc_id": doc_id, "title": title, "content_length": len(content)}
                )

            # Stale: updated_at older than 30 days.
            updated_at_raw = doc.metadata.get("updated_at", "")
            if updated_at_raw:
                try:
                    updated_at = datetime.fromisoformat(updated_at_raw)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    delta = now - updated_at
                    if delta > stale_threshold:
                        stale_pages.append(
                            {
                                "doc_id": doc_id,
                                "title": title,
                                "days_since_update": delta.days,
                            }
                        )
                except Exception:
                    pass

        issues_found = len(orphan_pages) + len(stale_pages) + len(empty_pages)
        return {
            "orphan_pages": orphan_pages,
            "stale_pages": stale_pages,
            "empty_pages": empty_pages,
            "total_pages": len(all_pages),
            "issues_found": issues_found,
        }

    def delete_page(self, doc_id: str) -> None:
        """Delete a wiki document from Vespa by doc_id and rebuild the index.

        Raises RuntimeError if the Vespa DELETE request fails.
        """
        url = f"{self._backend._url}:{self._backend._port}"
        delete_url = (
            f"{url}/document/v1/wiki_content/{self._schema_name}/docid/{doc_id}"
        )
        try:
            resp = requests.delete(delete_url, timeout=10)
            if not resp.ok:
                raise RuntimeError(
                    f"Vespa DELETE returned {resp.status_code}: {resp.text[:200]}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to delete wiki page {doc_id}: {exc}") from exc

        self._rebuild_index()

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
            # Merge: use RLM for large content, otherwise simple append.
            old_content = existing.text_content or existing.metadata.get("content", "")
            if self._should_use_rlm_for_merge(old_content, new_content):
                merged_content = self._merge_with_rlm(old_content, new_content, entity)
            else:
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

    def _should_use_rlm_for_merge(self, old_content: str, new_content: str) -> bool:
        """Return True when combined content length exceeds 50,000 characters."""
        return len(old_content) + len(new_content) >= 50_000

    def _merge_with_rlm(self, old_content: str, new_content: str, entity: str) -> str:
        """Synthesize merged topic content using RLM.

        Falls back to simple append on any error so feed operations are never blocked.
        """
        combined_context = (
            f"## Existing knowledge about {entity}\n\n{old_content}"
            f"\n\n---\n\n## New information about {entity}\n\n{new_content}"
        )
        query = f"Synthesize a comprehensive, non-redundant summary about {entity} from the existing and new information."
        try:
            rlm_llm_config = LLMEndpointConfig(model="ollama/qwen3:4b")
            rlm = RLMInference(llm_config=rlm_llm_config)
            result = rlm.process(query=query, context=combined_context)
            return result.answer
        except Exception:
            logger.warning(
                "RLM merge failed for entity '%s'; falling back to append", entity
            )
            return old_content + _CONTENT_SEPARATOR + new_content

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
                    "tenant_id": self._tenant_id,
                    "strategy": "semantic_search",
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
        """Return a 768-dim text embedding via the shared SemanticEmbedder."""
        from cogniverse_core.common.models.semantic_embedder import (
            get_semantic_embedder,
        )

        try:
            embedder = get_semantic_embedder()
            vec = embedder.encode(text)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except Exception as exc:
            logger.warning("Embedding generation failed (%s); using zero vector", exc)
            return [0.0] * 768
