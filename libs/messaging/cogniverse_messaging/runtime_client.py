"""HTTP client for Cogniverse runtime API.

Handles agent dispatch, event streaming, and health checks.
All agent logic stays in the runtime — this is a thin protocol adapter.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class RuntimeClient:
    """Async HTTP client for the Cogniverse runtime API."""

    def __init__(self, runtime_url: str, timeout: float = 300.0):
        self.runtime_url = runtime_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.runtime_url, timeout=self.timeout
            )
        return self._client

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def dispatch_agent(
        self,
        agent_name: str,
        query: str,
        tenant_id: str,
        context_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a task to an agent and return the response.

        Args:
            agent_name: Target agent (routing_agent, search_agent, etc.)
            query: User query text
            tenant_id: Tenant identifier (org:user format)
            context_id: Conversation identifier (Telegram chat_id)
            conversation_history: Prior conversation turns
            top_k: Max results for search agents
            context: Additional context dict (media URLs, etc.)
        """
        client = await self._get_client()

        payload: Dict[str, Any] = {
            "agent_name": agent_name,
            "query": query,
            "context": {"tenant_id": tenant_id, **(context or {})},
            "top_k": top_k,
        }
        if context_id:
            payload["context_id"] = context_id
        if conversation_history:
            payload["conversation_history"] = conversation_history

        resp = await client.post(
            f"/agents/{agent_name}/process",
            json=payload,
        )

        if resp.status_code != 200:
            logger.error(
                f"Agent dispatch failed: {resp.status_code}: {resp.text[:300]}"
            )
            return {
                "status": "error",
                "message": f"Agent returned {resp.status_code}",
            }

        return resp.json()

    async def stream_events(self, task_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to SSE events for a streaming task."""
        client = await self._get_client()

        async with client.stream("GET", f"/events/workflows/{task_id}") as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip():
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

    async def create_invite_token(
        self, tenant_id: str, expires_in_hours: int = 24
    ) -> Optional[str]:
        """Create an invite token for user registration."""
        client = await self._get_client()
        resp = await client.post(
            "/admin/messaging/invite",
            json={
                "tenant_id": tenant_id,
                "expires_in_hours": expires_in_hours,
            },
        )
        if resp.status_code == 200:
            return resp.json().get("token")
        return None

    # ------------------------------------------------------------------
    # wiki / instructions / memories / jobs CRUD methods
    # ------------------------------------------------------------------
    # Each method below maps one-to-one to a /wiki/* or /admin/tenant/*
    # runtime endpoint and is invoked by the gateway's command handlers.

    async def save_wiki_session(
        self,
        tenant_id: str,
        query: str,
        response: Dict[str, Any],
        agent_name: str = "routing_agent",
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Save an agent interaction as a wiki page (POST /wiki/save)."""
        client = await self._get_client()
        resp = await client.post(
            "/wiki/save",
            json={
                "query": query,
                "response": response,
                "entities": entities or [],
                "agent_name": agent_name,
                "tenant_id": tenant_id,
            },
        )
        return self._json_or_error(resp)

    async def search_wiki(
        self, tenant_id: str, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Full-text search over wiki pages (POST /wiki/search)."""
        client = await self._get_client()
        resp = await client.post(
            "/wiki/search",
            json={"query": query, "tenant_id": tenant_id, "top_k": top_k},
        )
        return self._json_or_error(resp)

    async def get_wiki_topic(self, tenant_id: str, slug: str) -> Dict[str, Any]:
        """Retrieve a topic page by slug (GET /wiki/topic/{slug})."""
        client = await self._get_client()
        resp = await client.get(f"/wiki/topic/{slug}", params={"tenant_id": tenant_id})
        return self._json_or_error(resp)

    async def get_wiki_index(self, tenant_id: str) -> Dict[str, Any]:
        """Return the rendered wiki index (GET /wiki/index)."""
        client = await self._get_client()
        resp = await client.get("/wiki/index", params={"tenant_id": tenant_id})
        return self._json_or_error(resp)

    async def lint_wiki(self, tenant_id: str) -> Dict[str, Any]:
        """Run lint checks on the wiki (GET /wiki/lint)."""
        client = await self._get_client()
        resp = await client.get("/wiki/lint", params={"tenant_id": tenant_id})
        return self._json_or_error(resp)

    async def delete_wiki_topic(self, tenant_id: str, slug: str) -> Dict[str, Any]:
        """Delete a topic page by slug (DELETE /wiki/topic/{slug})."""
        client = await self._get_client()
        resp = await client.delete(
            f"/wiki/topic/{slug}", params={"tenant_id": tenant_id}
        )
        return self._json_or_error(resp)

    async def set_instructions(self, tenant_id: str, text: str) -> Dict[str, Any]:
        """Set per-tenant system instructions (PUT /admin/tenant/{tenant}/instructions)."""
        client = await self._get_client()
        resp = await client.put(
            f"/admin/tenant/{tenant_id}/instructions",
            json={"text": text},
        )
        return self._json_or_error(resp)

    async def get_instructions(self, tenant_id: str) -> Dict[str, Any]:
        """Get per-tenant system instructions (GET /admin/tenant/{tenant}/instructions)."""
        client = await self._get_client()
        resp = await client.get(f"/admin/tenant/{tenant_id}/instructions")
        return self._json_or_error(resp)

    async def list_memories(
        self, tenant_id: str, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """List per-tenant memories (GET /admin/tenant/{tenant}/memories)."""
        client = await self._get_client()
        params: Dict[str, Any] = {}
        if agent_name:
            params["agent_name"] = agent_name
        resp = await client.get(f"/admin/tenant/{tenant_id}/memories", params=params)
        return self._json_or_error(resp)

    async def clear_memories(
        self, tenant_id: str, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clear per-tenant memories (DELETE /admin/tenant/{tenant}/memories)."""
        client = await self._get_client()
        params: Dict[str, Any] = {}
        if agent_name:
            params["agent_name"] = agent_name
        resp = await client.delete(f"/admin/tenant/{tenant_id}/memories", params=params)
        return self._json_or_error(resp)

    async def list_jobs(self, tenant_id: str) -> Dict[str, Any]:
        """List per-tenant scheduled jobs (GET /admin/tenant/{tenant}/jobs)."""
        client = await self._get_client()
        resp = await client.get(f"/admin/tenant/{tenant_id}/jobs")
        return self._json_or_error(resp)

    async def create_job(
        self,
        tenant_id: str,
        name: str,
        schedule: str,
        query: str,
        post_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a per-tenant scheduled job (POST /admin/tenant/{tenant}/jobs)."""
        client = await self._get_client()
        resp = await client.post(
            f"/admin/tenant/{tenant_id}/jobs",
            json={
                "name": name,
                "schedule": schedule,
                "query": query,
                "post_actions": post_actions or [],
            },
        )
        return self._json_or_error(resp)

    async def delete_job(self, tenant_id: str, job_id: str) -> Dict[str, Any]:
        """Delete a per-tenant scheduled job (DELETE /admin/tenant/{tenant}/jobs/{job_id})."""
        client = await self._get_client()
        resp = await client.delete(f"/admin/tenant/{tenant_id}/jobs/{job_id}")
        return self._json_or_error(resp)

    @staticmethod
    def _json_or_error(resp: httpx.Response) -> Dict[str, Any]:
        """Return parsed JSON on 2xx, else a structured error dict.

        Centralised so all CRUD wrappers above handle non-2xx responses
        the same way and the gateway can format errors uniformly.
        """
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return {"status": "ok"}
        return {
            "status": "error",
            "status_code": resp.status_code,
            "message": resp.text[:500] if resp.text else "no response body",
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
