"""Wiki endpoints — save, search, and retrieve wiki knowledge pages."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

_wiki_manager = None


def set_wiki_manager(wm) -> None:
    """Inject WikiManager dependency."""
    global _wiki_manager
    _wiki_manager = wm
    logger.info("WikiManager injected into wiki router")


def get_wiki_manager():
    """Return the injected WikiManager or raise if not configured."""
    if _wiki_manager is None:
        raise RuntimeError(
            "WikiManager not configured. Call set_wiki_manager() first."
        )
    return _wiki_manager


class WikiSaveRequest(BaseModel):
    query: str
    response: Dict[str, Any]
    entities: List[str] = []
    agent_name: str = "routing_agent"
    tenant_id: str = "default"


class WikiSearchRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    top_k: int = 5


@router.post("/save")
async def save_wiki(request: WikiSaveRequest) -> Dict[str, Any]:
    """Persist an agent interaction as a wiki page."""
    wm = get_wiki_manager()
    response_text = str(request.response.get("answer", request.response))
    page = wm.save_session(
        query=request.query,
        response=response_text,
        entities=request.entities,
        agent_name=request.agent_name,
    )
    return {"status": "saved", "doc_id": page.doc_id, "title": page.title, "slug": page.slug}


@router.post("/search")
async def search_wiki(request: WikiSearchRequest) -> Dict[str, Any]:
    """Full-text search over wiki pages."""
    wm = get_wiki_manager()
    results = wm.search(query=request.query, top_k=request.top_k)
    return {"results": results, "count": len(results)}


@router.get("/topic/{slug}")
async def get_wiki_topic(slug: str) -> Dict[str, Any]:
    """Retrieve a topic page by slug."""
    wm = get_wiki_manager()
    topic = wm.get_topic(slug)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic '{slug}' not found")
    return topic


@router.get("/index")
async def get_wiki_index() -> Dict[str, Any]:
    """Return the rendered wiki index."""
    wm = get_wiki_manager()
    index_content = wm.get_index()
    return {"content": index_content or ""}


@router.get("/lint")
async def lint_wiki() -> Dict[str, Any]:
    """Run lint checks and return a report of quality issues."""
    wm = get_wiki_manager()
    return wm.lint()


@router.delete("/topic/{slug}")
async def delete_wiki_topic(slug: str) -> Dict[str, Any]:
    """Delete a topic page by slug."""
    wm = get_wiki_manager()
    safe = wm._tenant_id.replace(":", "_")
    doc_id = f"wiki_topic_{safe}_{slug}"
    try:
        wm.delete_page(doc_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "deleted", "doc_id": doc_id, "slug": slug}
