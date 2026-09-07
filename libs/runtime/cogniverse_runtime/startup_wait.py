"""Wait for service dependencies at process startup.

``main`` is the exec wrapper deployments gate a command on; the wrapper itself
is stdlib-only. ``wait_for_startup_dependency`` is the in-process form the
long-running entrypoints (quality monitor, ingestion worker) use to build
their first config-store-backed object through a store outage.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

_POLL_INTERVAL_SECONDS = 0.1
_PROBE_TIMEOUT_SECONDS = 1.0

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DependencyTimeout(RuntimeError):
    """A required service did not become ready before the deadline."""


class DependencyWaitAborted(RuntimeError):
    """Shutdown was requested while a dependency was still not ready."""


def wait_for_startup_dependency(
    build: Callable[[], T],
    *,
    dependency: str,
    process: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    retry_forever: bool,
    abort: Callable[[], bool] | None = None,
    log: logging.Logger | None = None,
) -> T:
    """Call ``build`` until it returns, retrying transport and config-store
    outages.

    ``timeout_seconds`` is a grace window. Inside it every failure logs a
    WARNING. Once it elapses a long-running ``process`` (``retry_forever``)
    logs one ERROR and keeps retrying with a WARNING per attempt, because a
    restart loop is worse than waiting; a one-shot caller raises
    ``RuntimeError`` instead so the job reports failure rather than holding
    its reservation. ``abort`` is polled after each failure so a shutdown
    request ends the wait with ``DependencyWaitAborted``.
    """
    import httpr
    import requests

    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    log = log or logger
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    attempts = 0
    last_error: Exception | None = None
    while True:
        attempts += 1
        try:
            return build()
        except (
            httpr.TransportError,
            requests.RequestException,
            ConfigStoreUnavailableError,
        ) as error:
            last_error = error

        attempt_word = "attempt" if attempts == 1 else "attempts"
        if abort is not None and abort():
            raise DependencyWaitAborted(
                f"{dependency} wait aborted by shutdown request after "
                f"{attempts} {attempt_word}: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if not retry_forever:
                raise RuntimeError(
                    f"{dependency} was not ready after "
                    f"{attempts} {attempt_word} within {timeout_seconds:.1f}s: "
                    f"{type(last_error).__name__}: {last_error}"
                ) from last_error
            if not timed_out:
                log.error(
                    f"{dependency} was not ready after "
                    f"{attempts} {attempt_word} within {timeout_seconds:.1f}s: "
                    f"{type(last_error).__name__}: {last_error}; "
                    f"keeping the {process} alive and retrying"
                )
                timed_out = True
            else:
                log.warning(
                    "%s is still not ready (attempt %d, retrying in %.1fs): %s: %s",
                    dependency,
                    attempts,
                    poll_interval_seconds,
                    type(last_error).__name__,
                    last_error,
                )
            sleep_for = poll_interval_seconds
        else:
            sleep_for = min(poll_interval_seconds, remaining)
            log.warning(
                "%s is not ready (attempt %d, retrying in %.1fs): %s: %s",
                dependency,
                attempts,
                sleep_for,
                type(last_error).__name__,
                last_error,
            )
        time.sleep(max(sleep_for, 0))


@dataclass(frozen=True)
class TcpDependency:
    value: str
    host: str
    port: int


@dataclass(frozen=True)
class HttpDependency:
    url: str
    statuses: frozenset[int]

    @property
    def value(self) -> str:
        if self.statuses == {200}:
            return self.url
        rendered = ",".join(str(status) for status in sorted(self.statuses))
        return f"{self.url} (statuses: {rendered})"


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return timeout


def _http_dependency(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            f"invalid HTTP dependency {value!r}; expected an absolute HTTP URL"
        )
    return value


def _http_statuses(value: str) -> frozenset[int]:
    raw_statuses = value.split(",")
    try:
        statuses = [int(status) for status in raw_statuses]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid HTTP status list {value!r}; expected comma-separated integers"
        ) from exc
    if (
        not value
        or any(not status for status in raw_statuses)
        or any(status < 100 or status > 599 for status in statuses)
        or len(statuses) != len(set(statuses))
    ):
        raise argparse.ArgumentTypeError(
            f"invalid HTTP status list {value!r}; expected unique values from 100 to 599"
        )
    return frozenset(statuses)


def _tcp_dependency(value: str) -> TcpDependency:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"invalid TCP dependency {value!r}; expected HOST:PORT"
        )
    host, raw_port = parts
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid TCP dependency {value!r}; expected HOST:PORT"
        ) from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"invalid TCP dependency {value!r}; port must be between 1 and 65535"
        )
    return TcpDependency(value=value, host=host, port=port)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", required=True, type=_positive_timeout)
    parser.add_argument("--http", action="append", default=[], type=_http_dependency)
    parser.add_argument(
        "--http-status",
        action="append",
        default=[],
        nargs=2,
        metavar=("URL", "STATUSES"),
    )
    parser.add_argument("--tcp", action="append", default=[], type=_tcp_dependency)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        args.http_dependencies = [
            HttpDependency(url=url, statuses=frozenset({200})) for url in args.http
        ] + [
            HttpDependency(url=_http_dependency(url), statuses=_http_statuses(statuses))
            for url, statuses in args.http_status
        ]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if not args.http_dependencies and not args.tcp:
        parser.error(
            "at least one --http, --http-status, or --tcp dependency is required"
        )
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    return args


def _http_ready(dependency: HttpDependency, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(dependency.url, timeout=timeout) as response:
            return response.status in dependency.statuses
    except urllib.error.HTTPError as exc:
        return exc.code in dependency.statuses
    except (OSError, urllib.error.URLError):
        return False


def _tcp_ready(dependency: TcpDependency, timeout: float) -> bool:
    try:
        with socket.create_connection(
            (dependency.host, dependency.port), timeout=timeout
        ):
            return True
    except OSError:
        return False


def _wait_for_dependency(
    *,
    kind: str,
    value: str,
    ready: Callable[[float], bool],
    deadline: float,
    timeout_seconds: float,
) -> None:
    print(f"waiting for {kind} dependency {value}", flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DependencyTimeout(
                f"timed out waiting for {kind} dependency {value} "
                f"after {timeout_seconds:.2f} seconds"
            )
        if ready(min(_PROBE_TIMEOUT_SECONDS, remaining)):
            print(f"{kind} dependency ready: {value}", flush=True)
            return
        time.sleep(min(_POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        for dependency in args.http_dependencies:
            _wait_for_dependency(
                kind="http",
                value=dependency.value,
                ready=lambda timeout, dependency=dependency: _http_ready(
                    dependency, timeout
                ),
                deadline=deadline,
                timeout_seconds=args.timeout_seconds,
            )
        for dependency in args.tcp:
            _wait_for_dependency(
                kind="tcp",
                value=dependency.value,
                ready=lambda timeout, dependency=dependency: _tcp_ready(
                    dependency, timeout
                ),
                deadline=deadline,
                timeout_seconds=args.timeout_seconds,
            )
    except DependencyTimeout as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    os.execvp(args.command[0], args.command)


if __name__ == "__main__":
    raise SystemExit(main())
