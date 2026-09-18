"""Served model-id discovery for OpenAI-compatible inference endpoints.

Remote inference runs on scale-to-zero Modal apps, so the first request after a
scaledown waits for the container to boot. Discovery therefore retries until
the service's ``boot_deadline_seconds`` — the same cold-start budget the
runtime's own readiness probes use — instead of failing an ingest job on one
short read timeout. The answer is cached per process and reused across jobs;
an endpoint that never answers raises, naming the endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Mapping

import requests

from cogniverse_foundation.inference_specs import get_inference_service_spec

# One attempt's read budget. Modal holds a request while the container boots,
# so this is long enough to be answered by that boot rather than cut short.
DISCOVERY_ATTEMPT_TIMEOUT_SECONDS = 60.0
DISCOVERY_RETRY_INTERVAL_SECONDS = 5.0

# Statuses a booting engine emits and a ready one does not. Anything else — a
# 404 from a URL that is not an OpenAI-compatible root, a 401 from a missing
# bearer — is a configuration fault that waiting cannot repair, so it fails on
# the first attempt instead of holding the job for the whole cold-start budget.
_RETRYABLE_DISCOVERY_STATUSES = frozenset({408, 425, 429})

_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_BASE_LOCKS: dict[str, threading.Lock] = {}
# Completed-discovery counter and the last failure per endpoint. Jobs that were
# already queued behind a discovery adopt its failure instead of each spending
# the whole cold-start budget on the same dead endpoint; a job arriving after
# that wave discovers afresh, so a failure is never sticky.
_ATTEMPTS: dict[str, int] = {}
_FAILURES: dict[str, str] = {}


class ServedModelUnavailable(RuntimeError):
    """An inference endpoint did not report a served model id in time."""


def reset_served_model_cache() -> None:
    """Forget every discovered model id."""

    with _CACHE_LOCK:
        _CACHE.clear()
        _BASE_LOCKS.clear()
        _ATTEMPTS.clear()
        _FAILURES.clear()


def _is_retryable(status: int) -> bool:
    """Whether a ``/models`` status can still become a served model id."""

    return status >= 500 or status in _RETRYABLE_DISCOVERY_STATUSES


def _base_lock(base_url: str) -> threading.Lock:
    with _CACHE_LOCK:
        lock = _BASE_LOCKS.get(base_url)
        if lock is None:
            lock = threading.Lock()
            _BASE_LOCKS[base_url] = lock
        return lock


def resolve_served_model_id(
    base_url: str,
    *,
    service_name: str,
    headers: Mapping[str, str],
    logger: logging.Logger,
    deadline_seconds: float | None = None,
    retry_interval_seconds: float = DISCOVERY_RETRY_INTERVAL_SECONDS,
    attempt_timeout_seconds: float = DISCOVERY_ATTEMPT_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Return the model id served at ``base_url`` (an OpenAI-compatible ``/v1``
    root), discovering it at most once per process per endpoint.

    ``service_name`` names the inference service spec whose cold-start budget
    bounds discovery. Raises :class:`ServedModelUnavailable` when the endpoint
    reports no model id before that budget runs out.
    """

    cached = _CACHE.get(base_url)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        queued_behind = _ATTEMPTS.get(base_url, 0)

    with _base_lock(base_url):
        cached = _CACHE.get(base_url)
        if cached is not None:
            return cached
        with _CACHE_LOCK:
            if _ATTEMPTS.get(base_url, 0) != queued_behind:
                shared_failure = _FAILURES.get(base_url)
                if shared_failure is not None:
                    raise ServedModelUnavailable(shared_failure)
        if deadline_seconds is None:
            deadline_seconds = get_inference_service_spec(
                service_name
            ).boot_deadline_seconds
        try:
            model_id = _discover(
                base_url,
                service_name=service_name,
                headers=headers,
                logger=logger,
                deadline_seconds=deadline_seconds,
                retry_interval_seconds=retry_interval_seconds,
                attempt_timeout_seconds=attempt_timeout_seconds,
                now=now,
                sleep=sleep,
            )
        except ServedModelUnavailable as exc:
            with _CACHE_LOCK:
                _FAILURES[base_url] = str(exc)
                _ATTEMPTS[base_url] = _ATTEMPTS.get(base_url, 0) + 1
            raise
        with _CACHE_LOCK:
            _CACHE[base_url] = model_id
            _FAILURES.pop(base_url, None)
            _ATTEMPTS[base_url] = _ATTEMPTS.get(base_url, 0) + 1
        return model_id


def _discover(
    base_url: str,
    *,
    service_name: str,
    headers: Mapping[str, str],
    logger: logging.Logger,
    deadline_seconds: float,
    retry_interval_seconds: float,
    attempt_timeout_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> str:
    models_url = f"{base_url}/models"
    deadline = now() + deadline_seconds
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - now()
        try:
            resp = requests.get(
                models_url,
                headers=headers,
                timeout=max(1.0, min(attempt_timeout_seconds, remaining)),
            )
            status = resp.status_code
            if status >= 400 and not _is_retryable(status):
                raise ServedModelUnavailable(
                    f"Inference endpoint {base_url} ({service_name}) answered "
                    f"{status} for {models_url}; waiting cannot repair that, so "
                    f"discovery is not retried"
                )
            resp.raise_for_status()
            served = resp.json().get("data") or []
            if served:
                model_id = served[0]["id"]
                logger.info(
                    "Inference endpoint %s serves %s (attempt %d)",
                    base_url,
                    model_id,
                    attempt,
                )
                return model_id
            last_error = "endpoint listed no models"
        except ServedModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - reported on deadline expiry
            last_error = f"{type(exc).__name__}: {exc}"

        remaining = deadline - now()
        if remaining <= 0:
            raise ServedModelUnavailable(
                f"Inference endpoint {base_url} ({service_name}) did not report a "
                f"served model id within {deadline_seconds:g}s over {attempt} "
                f"attempts; last error: {last_error}"
            )
        wait = min(retry_interval_seconds, remaining)
        logger.info(
            "Inference endpoint %s not ready yet (attempt %d, last error: %s); "
            "retrying in %.1fs",
            base_url,
            attempt,
            last_error,
            wait,
        )
        sleep(wait)
