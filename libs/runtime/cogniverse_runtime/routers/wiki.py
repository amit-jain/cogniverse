"""Wiki endpoints — save, search, and retrieve wiki knowledge pages.

Wiki state is per-tenant. The runtime injects a factory via
``set_wiki_manager_factory``; each request resolves a
``WikiManager`` bound to the request's ``tenant_id`` through
``get_wiki_manager_for_tenant``. Same pattern as ``routers/graph.py``.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# Factory installed by main.py at startup. Takes a tenant_id and returns a
# WikiManager bound to that tenant. The factory is responsible for caching
# (one manager per tenant) so this router never owns lifecycle.
_wiki_manager_factory: Optional[Callable[[str], Any]] = None


def set_wiki_manager_factory(factory: Callable[[str], Any]) -> None:
    """Inject a per-tenant WikiManager factory.

    main.py builds this factory once at startup. Each call to the factory
    returns a WikiManager bound to a specific tenant. Per-tenant caching
    lives in the factory itself.
    """
    global _wiki_manager_factory
    _wiki_manager_factory = factory
    logger.info("WikiManager factory injected into wiki router")


def get_wiki_manager_for_tenant(tenant_id: str):
    """Return the WikiManager for ``tenant_id`` or raise HTTPException 503.

    Used by every endpoint to resolve the per-request manager. Centralised
    so a missing factory always produces the same error response.
    """
    if _wiki_manager_factory is None:
        raise HTTPException(
            status_code=503,
            detail="WikiManager factory not configured. Check runtime startup logs.",
        )
    return _wiki_manager_factory(tenant_id)


class WikiSaveRequest(BaseModel):
    query: str
    response: Dict[str, Any]
    entities: List[str] = []
    agent_name: str = "routing_agent"
    tenant_id: str = Field(..., description="Tenant identity; required")


class WikiSearchRequest(BaseModel):
    query: str
    tenant_id: str = Field(..., description="Tenant identity; required")
    top_k: int = 5


@router.post("/save")
async def save_wiki(request: WikiSaveRequest) -> Dict[str, Any]:
    """Persist an agent interaction as a wiki page for the request's tenant."""
    wm = get_wiki_manager_for_tenant(request.tenant_id)
    response_text = str(request.response.get("answer", request.response))
    page = wm.save_session(
        query=request.query,
        response=response_text,
        entities=request.entities,
        agent_name=request.agent_name,
    )
    return {
        "status": "saved",
        "doc_id": page.doc_id,
        "title": page.title,
        "slug": page.slug,
    }


@router.post("/search")
async def search_wiki(request: WikiSearchRequest) -> Dict[str, Any]:
    """Full-text search over wiki pages for the request's tenant."""
    wm = get_wiki_manager_for_tenant(request.tenant_id)
    results = wm.search(query=request.query, top_k=request.top_k)
    return {"results": results, "count": len(results)}


@router.get("/topic/{slug}")
async def get_wiki_topic(
    slug: str, tenant_id: str = Query(..., description="Tenant identity")
) -> Dict[str, Any]:
    """Retrieve a topic page by slug for the given tenant."""
    wm = get_wiki_manager_for_tenant(tenant_id)
    topic = wm.get_topic(slug)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic '{slug}' not found")
    return topic


@router.get("/index")
async def get_wiki_index(
    tenant_id: str = Query(..., description="Tenant identity"),
) -> Dict[str, Any]:
    """Return the rendered wiki index for the given tenant."""
    wm = get_wiki_manager_for_tenant(tenant_id)
    index_content = wm.get_index()
    return {"content": index_content or ""}


@router.get("/lint")
async def lint_wiki(
    tenant_id: str = Query(..., description="Tenant identity"),
) -> Dict[str, Any]:
    """Run lint checks for the given tenant and return a quality report."""
    wm = get_wiki_manager_for_tenant(tenant_id)
    return wm.lint()


@router.delete("/topic/{slug}")
async def delete_wiki_topic(
    slug: str, tenant_id: str = Query(..., description="Tenant identity")
) -> Dict[str, Any]:
    """Delete a topic page by slug for the given tenant."""
    wm = get_wiki_manager_for_tenant(tenant_id)
    safe = wm._tenant_id.replace(":", "_")
    doc_id = f"wiki_topic_{safe}_{slug}"
    try:
        wm.delete_page(doc_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "deleted", "doc_id": doc_id, "slug": slug}
