"""The deadline an LM call inherits from the caller waiting on it.

A caller that bounds an LM call binds an ``LMCallDeadline`` for it. The LM
reads the binding: each HTTP request gets the time that is left when it is
sent as its timeout, and no request is sent once the deadline has passed or
the caller has given up. A caller that stops waiting marks the deadline
abandoned, so an attempt that has not started yet never starts.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import httpcore
import httpx

if TYPE_CHECKING:
    import openai


class LMCallDeadlineExceeded(TimeoutError):
    """An LM call stopped because its deadline passed or its caller gave up."""

    def __init__(
        self,
        *,
        endpoint: Optional[str],
        model: str,
        budget_s: float,
        abandoned: bool,
    ) -> None:
        reason = "its caller gave up" if abandoned else "its deadline passed"
        call = (
            f"LM call to {endpoint} for {model}" if endpoint else f"LM call for {model}"
        )
        super().__init__(f"{call} stopped, {reason}: deadline {budget_s:.2f}s")
        self.endpoint = endpoint
        self.model = model
        self.budget_s = budget_s
        self.abandoned = abandoned


@dataclass(frozen=True)
class LMCallDeadline:
    """The monotonic time an LM call must finish by, and whether the caller
    waiting on it has already given up."""

    budget_s: float
    at: float
    _abandoned: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    @classmethod
    def after(cls, budget_s: float) -> "LMCallDeadline":
        return cls(budget_s=budget_s, at=time.monotonic() + budget_s)

    def remaining_s(self) -> float:
        return max(self.at - time.monotonic(), 0.0)

    def abandon(self) -> None:
        self._abandoned.set()

    @property
    def abandoned(self) -> bool:
        return self._abandoned.is_set()

    @property
    def expired(self) -> bool:
        return self.abandoned or time.monotonic() >= self.at

    def exceeded(
        self, *, endpoint: Optional[str], model: str
    ) -> LMCallDeadlineExceeded:
        return LMCallDeadlineExceeded(
            endpoint=endpoint,
            model=model,
            budget_s=self.budget_s,
            abandoned=self.abandoned,
        )


_CURRENT: ContextVar[Optional[LMCallDeadline]] = ContextVar(
    "lm_call_deadline", default=None
)


def current_lm_call_deadline() -> Optional[LMCallDeadline]:
    """The deadline bound for LM calls in this context, if any."""
    return _CURRENT.get()


def _within_bound_deadline(timeout: Optional[float], expired: type[Exception]):
    """``timeout`` clamped to the time the bound deadline leaves right now;
    raise ``expired`` once it is spent or abandoned."""
    deadline = current_lm_call_deadline()
    if deadline is None:
        return timeout
    remaining = deadline.remaining_s()
    if deadline.abandoned or remaining <= 0:
        raise expired(f"LM call deadline {deadline.budget_s:.2f}s spent")
    return remaining if timeout is None else min(timeout, remaining)


class _DeadlineStream(httpcore.NetworkStream):
    """A connection whose every read, write and TLS handshake waits no longer
    than the bound deadline leaves at that moment."""

    def __init__(self, inner: httpcore.NetworkStream) -> None:
        self._inner = inner

    def read(self, max_bytes: int, timeout: Optional[float] = None) -> bytes:
        return self._inner.read(
            max_bytes, _within_bound_deadline(timeout, httpcore.ReadTimeout)
        )

    def write(self, buffer: bytes, timeout: Optional[float] = None) -> None:
        self._inner.write(
            buffer, _within_bound_deadline(timeout, httpcore.WriteTimeout)
        )

    def close(self) -> None:
        self._inner.close()

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return _DeadlineStream(
            self._inner.start_tls(
                ssl_context,
                server_hostname,
                _within_bound_deadline(timeout, httpcore.ConnectTimeout),
            )
        )

    def get_extra_info(self, info: str):
        return self._inner.get_extra_info(info)


class _DeadlineBackend(httpcore.NetworkBackend):
    """Opens connections that honour the deadline bound for the call using
    them, whichever call that is."""

    def __init__(self) -> None:
        self._inner = httpcore.SyncBackend()

    def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        return _DeadlineStream(
            self._inner.connect_tcp(
                host,
                port,
                _within_bound_deadline(timeout, httpcore.ConnectTimeout),
                local_address,
                socket_options,
            )
        )

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return _DeadlineStream(
            self._inner.connect_unix_socket(
                path,
                _within_bound_deadline(timeout, httpcore.ConnectTimeout),
                socket_options,
            )
        )

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


class _DeadlineTransport(httpx.HTTPTransport):
    """httpx's transport over connections that honour the bound deadline."""

    def __init__(self) -> None:
        super().__init__()
        limits = httpx.Limits()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=_DeadlineBackend(),
        )


# One client per configured endpoint and key, shared across calls so bounded
# calls reuse its connection pool.
_DEADLINE_CLIENTS: dict[tuple[str, Optional[str]], "openai.OpenAI"] = {}
_DEADLINE_CLIENTS_LOCK = threading.Lock()


def deadline_bound_openai_client(
    api_base: str, api_key: Optional[str]
) -> "openai.OpenAI":
    """The OpenAI-compatible client whose requests honour the bound deadline."""
    import openai

    key = (api_base, api_key)
    with _DEADLINE_CLIENTS_LOCK:
        client = _DEADLINE_CLIENTS.get(key)
        if client is None:
            client = openai.OpenAI(
                base_url=api_base,
                api_key=api_key or "unset",
                max_retries=0,
                http_client=httpx.Client(transport=_DeadlineTransport()),
            )
            _DEADLINE_CLIENTS[key] = client
        return client


@contextmanager
def bound_lm_call_deadline(deadline: Optional[LMCallDeadline]) -> Iterator[None]:
    """Bind ``deadline`` for LM calls made in this context; ``None`` binds nothing."""
    if deadline is None:
        yield
        return
    token = _CURRENT.set(deadline)
    try:
        yield
    finally:
        _CURRENT.reset(token)
