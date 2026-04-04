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

    async def stream_events(
        self, task_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to SSE events for a streaming task."""
        client = await self._get_client()

        async with client.stream(
            "GET", f"/events/workflows/{task_id}"
        ) as response:
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

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
