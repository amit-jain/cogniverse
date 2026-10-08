"""A tenant's optimization report, streamed from the detailed report agent.

The route asks ``detailed_report_agent`` for the tenant's optimization
performance report and forwards the agent's events (status, partial, final,
error) as server-sent events, one JSON event per ``data`` frame.
"""

import json
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.agent_registry import AgentRegistryUnavailableError
from cogniverse_runtime.http_errors import failure_response, record_failure
from cogniverse_runtime.routers import agents

router = APIRouter()

REPORT_AGENT = "detailed_report_agent"
REPORT_QUERY = (
    "Generate optimization performance report with findings and recommendations"
)


def _frame(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


async def _report_frames(dispatcher: Any, tenant_id: str) -> AsyncIterator[str]:
    context = {"tenant_id": tenant_id, "request_id": uuid.uuid4().hex}
    try:
        async with aclosing(
            dispatcher.dispatch_stream(REPORT_AGENT, REPORT_QUERY, context)
        ) as events:
            async for event in events:
                yield _frame(event)
    except Exception as exc:
        record_failure(exc, "optimization_report_failed")
        yield _frame(
            {
                "type": "error",
                "message": (
                    f"The report agent failed ({type(exc).__name__}) before "
                    f"finishing the report of tenant {tenant_id}."
                ),
                "error_type": type(exc).__name__,
            }
        )


@router.post("/{tenant_id}/optimize/report")
async def stream_optimization_report(tenant_id: str):
    """The tenant's optimization report as the agent produces it."""
    tenant_id = canonical_tenant_id(tenant_id)
    dispatcher = agents.get_dispatcher()
    try:
        await dispatcher.refresh_agent_registry()
    except AgentRegistryUnavailableError as exc:
        raise failure_response(
            503,
            "agent_registry_unavailable",
            "The agent registry could not be read; retry.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    if not dispatcher.is_registered(REPORT_AGENT):
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{REPORT_AGENT}' is not registered, so no report can "
            "be generated.",
        )
    return StreamingResponse(
        _report_frames(dispatcher, tenant_id), media_type="text/event-stream"
    )
