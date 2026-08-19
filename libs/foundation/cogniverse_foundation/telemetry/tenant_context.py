"""The current tenant, as a context variable, for span routing.

A request handler enters a tenant span (``TelemetryManager.span``) which sets
this variable for the span's lifetime. Anything that emits a span while that
scope is active -- including instrumented libraries such as DSPy running on a
worker thread that copies the context -- resolves its target project from
here, so every span in a trace lands in one tenant's project. An offline job
sets it explicitly from its ``--tenant-id``. A span emitted with no tenant in
context has nowhere to go and is dropped, never filed under a shared project.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

_current_tenant_id: ContextVar[Optional[str]] = ContextVar(
    "cogniverse_current_tenant_id", default=None
)


def current_tenant_id() -> Optional[str]:
    """Return the tenant id in scope, or ``None`` when outside any tenant span."""
    return _current_tenant_id.get()


@contextlib.contextmanager
def tenant_span_context(tenant_id: str) -> Iterator[None]:
    """Bind ``tenant_id`` as the current tenant for the duration of the block.

    Nesting restores the enclosing tenant on exit, so concurrent requests on
    different tasks never observe each other's tenant.
    """
    token = _current_tenant_id.set(tenant_id)
    try:
        yield
    finally:
        _current_tenant_id.reset(token)
