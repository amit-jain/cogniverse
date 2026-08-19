"""A tracer that routes each span to the current tenant's project.

``DSPyInstrumentor`` (and any other OpenInference instrumentor) is process-
global and takes a single ``TracerProvider``. Giving it a
``TenantRoutingTracerProvider`` makes every instrumented span resolve the
tenant in scope (see :mod:`tenant_context`) and delegate to that tenant's
real provider tracer -- the same provider that owns the request's root span,
so the whole trace is filed in one project with correct parentage.

There is no shared cross-tenant project: a span emitted with no tenant in
context yields a non-recording span and is dropped, with a single warning.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Optional

from opentelemetry.context import Context
from opentelemetry.trace import (
    INVALID_SPAN,
    NonRecordingSpan,
    Span,
    Tracer,
    TracerProvider,
    use_span,
)
from opentelemetry.util.types import Attributes

from cogniverse_foundation.telemetry.tenant_context import current_tenant_id

logger = logging.getLogger(__name__)

# A resolver maps a tenant id to that tenant's real OTel tracer. Production
# passes ``lambda t: manager._get_tracer_for_project(t, None)``; it may return
# ``None`` when telemetry is disabled or the tenant's tracer cannot be built.
ResolveTracer = Callable[[str], Optional[Tracer]]

_ORPHAN_SPAN = NonRecordingSpan(INVALID_SPAN.get_span_context())


class TenantRoutingTracer(Tracer):
    """Delegate each span to the current tenant's tracer, or drop it."""

    def __init__(self, resolve_tracer: ResolveTracer) -> None:
        self._resolve_tracer = resolve_tracer
        self._warned_orphan = False

    def _tenant_tracer(self, span_name: str) -> Optional[Tracer]:
        tenant_id = current_tenant_id()
        if tenant_id is None:
            if not self._warned_orphan:
                # Warn once per tracer: an instrumented span fired with no
                # tenant in context. It is dropped rather than filed under a
                # shared project -- a span with no tenant has nowhere to go.
                logger.warning(
                    "Instrumented span %r emitted with no tenant in context; "
                    "span dropped (not filed under any shared project)",
                    span_name,
                )
                self._warned_orphan = True
            return None
        return self._resolve_tracer(tenant_id)

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Span:
        tracer = self._tenant_tracer(name)
        if tracer is None:
            return _ORPHAN_SPAN
        return tracer.start_span(name, context, *args, **kwargs)

    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Span]:
        tracer = self._tenant_tracer(name)
        if tracer is None:
            # Mirror the SDK's contextmanager contract while recording nothing.
            return use_span(
                _ORPHAN_SPAN,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
        return tracer.start_as_current_span(name, context, *args, **kwargs)


class TenantRoutingTracerProvider(TracerProvider):
    """A provider whose tracers route by the tenant in context."""

    def __init__(self, resolve_tracer: ResolveTracer) -> None:
        self._resolve_tracer = resolve_tracer

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[Attributes] = None,
    ) -> Tracer:
        return TenantRoutingTracer(self._resolve_tracer)
