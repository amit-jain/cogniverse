"""
Vespa-based configuration storage with multi-tenant support.
Stores configurations directly in Vespa backend for unified storage.
"""

import base64
import binascii
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, NoReturn, Optional

import requests
from requests.exceptions import HTTPError, RequestException
from vespa.application import Vespa
from vespa.exceptions import VespaError

from cogniverse_sdk.interfaces.config_store import (
    ConfigEntry,
    ConfigScope,
    ConfigStoreUnavailableError,
    ImmutableConfigStore,
)
from cogniverse_vespa._vespa_factory import (
    VespaQueryDegraded,
    canonical_endpoint,
    hit_fields,
    make_persistent_vespa_ops,
    raise_if_degraded,
)
from cogniverse_vespa._yql import yql_quote

logger = logging.getLogger(__name__)

_MAX_VERSION_ALLOCATION_ATTEMPTS = 64
# Each config's version counter lives in one document under this namespace.
# Pruning deletes version documents, and Vespa applies a conditional put with
# ``create`` to a missing document whatever its condition says, so a version
# document's absence cannot tell "never written" from "written and pruned".
# The counter document is never pruned, so a conditional update of it hands
# out each version exactly once. Document v1 visits are scoped to the
# config_metadata namespace and the counter carries no config_id, so no
# config read or prune query ever sees it.
_VERSION_COUNTER_NAMESPACE = "config_version_counter"
# A reservation whose version document has not appeared after this long is
# treated as abandoned (its writer died between the two writes), so
# ``compare_and_set_config`` may reserve past it instead of reporting
# contention forever. A writer that was only slow still cannot report a
# version it lost: its own read-back sees the newer version.
_ABANDONED_RESERVATION_SECONDS = 30.0
_COUNTER_REREAD_EVERY_MISSES = 8

_CONFIG_STORE_READ_MAX_ATTEMPTS = 5
_CONFIG_STORE_READ_INITIAL_BACKOFF_SECONDS = 0.25
_CONFIG_STORE_READ_BACKOFF_MULTIPLIER = 2.0
_CONFIG_STORE_READ_MAX_BACKOFF_SECONDS = 2.0
_CONFIG_STORE_READ_RETRYABLE_HTTP_STATUS_MIN = 500
_CONFIG_STORE_READ_MISSING_HTTP_STATUS = 404
# Per-attempt budget for a single-document read; the same one
# list_immutable_configs uses for its bounded page.
_CONFIG_STORE_DOCUMENT_READ_TIMEOUT_SECONDS = 5


def _raise_if_degraded(response: Any, config_id: str) -> None:
    """Raise on a degraded query response — shared guard, config context.

    A degraded answer is the store not answering, so it raises
    ConfigStoreUnavailableError, the error callers wait out a store outage on.
    """
    try:
        raise_if_degraded(response, f"config {config_id}")
    except VespaQueryDegraded as exc:
        raise ConfigStoreUnavailableError(str(exc)) from exc


def _config_hit_fields(hit: Dict[str, Any], required: tuple, config_id: str):
    """``hit_fields`` in config context: a hit missing them is the store not
    answering, so it raises ConfigStoreUnavailableError."""
    try:
        return hit_fields(hit, required, f"config {config_id}")
    except VespaQueryDegraded as exc:
        raise ConfigStoreUnavailableError(str(exc)) from exc


def _visited_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    """A visited document's fields; one without them is malformed."""
    fields = document.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(
            f"config_metadata document {document.get('id')} came back without fields"
        )
    return fields


def _is_condition_miss(error: Exception) -> bool:
    current: Optional[BaseException] = error
    while current is not None:
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 412:
            return True
        if isinstance(current, HTTPError) and str(current).startswith("HTTP 412:"):
            return True
        current = current.__cause__
    return False


def _config_store_http_status(error: Exception) -> Optional[int]:
    current: Optional[BaseException] = error
    while current is not None:
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        current = current.__cause__
    return None


def _config_store_is_retryable(error: Exception) -> bool:
    current: Optional[BaseException] = error
    while current is not None:
        if isinstance(current, (requests.ConnectionError, requests.Timeout)):
            return True
        status_code = _config_store_http_status(current)
        if (
            status_code is not None
            and _CONFIG_STORE_READ_RETRYABLE_HTTP_STATUS_MIN <= status_code < 600
        ):
            return True
        current = current.__cause__
    return False


def _config_store_visit_backoff_seconds(attempt: int) -> float:
    delay = _CONFIG_STORE_READ_INITIAL_BACKOFF_SECONDS * (
        _CONFIG_STORE_READ_BACKOFF_MULTIPLIER ** (attempt - 1)
    )
    return min(delay, _CONFIG_STORE_READ_MAX_BACKOFF_SECONDS)


def _literal_glob_suffix(suffix: str) -> str:
    """``suffix`` as a document-selection glob literal.

    Selection globs have no escape, so a suffix carrying ``*`` or ``?`` would
    widen the match instead of narrowing it and could carry another tenant's
    rows. Such a suffix is refused rather than silently matched.
    """
    if not suffix:
        raise ValueError("config_key_suffix must be non-empty")
    if "*" in suffix or "?" in suffix:
        raise ValueError(
            f"config_key_suffix {suffix!r} carries a glob wildcard; selection "
            "globs cannot escape one"
        )
    return suffix


_IMMUTABLE_CURSOR_PREFIX = "v1."


def _encode_immutable_cursor(
    visit_continuation: Optional[str], pending_keys: List[str]
) -> Optional[str]:
    """The page cursor: where the visit resumes, plus the config keys a visit
    already returned beyond the last page.

    A visit's ``wantedDocumentCount`` is a hint; one visit can return more
    documents than asked. Those extras are carried here by key and served
    first on the next page, so a page never exceeds its size and no record is
    skipped or repeated.
    """
    if visit_continuation is None and not pending_keys:
        return None
    body = json.dumps(
        {"visit": visit_continuation, "pending": pending_keys},
        separators=(",", ":"),
    ).encode()
    return _IMMUTABLE_CURSOR_PREFIX + base64.urlsafe_b64encode(body).decode()


def _decode_immutable_cursor(cursor: str) -> tuple[Optional[str], List[str]]:
    if not cursor.startswith(_IMMUTABLE_CURSOR_PREFIX):
        raise ValueError("continuation is not a config-store page cursor")
    try:
        body = json.loads(
            base64.urlsafe_b64decode(cursor[len(_IMMUTABLE_CURSOR_PREFIX) :])
        )
        visit = body["visit"]
        pending = body["pending"]
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        raise ValueError("continuation is not a config-store page cursor") from exc
    if not (visit is None or isinstance(visit, str)) or not (
        isinstance(pending, list) and all(isinstance(key, str) for key in pending)
    ):
        raise ValueError("continuation is not a config-store page cursor")
    return visit, pending


def _config_store_read_json(
    path: str, *, params: Dict[str, Any], timeout: int, operation: str = "visit"
) -> Optional[Dict[str, Any]]:
    """One bounded document/v1 read: JSON, or None when the id is absent.

    Worst case is ``timeout`` per attempt across
    ``_CONFIG_STORE_READ_MAX_ATTEMPTS`` attempts plus the capped backoff
    between them, so a hung backend ends in ConfigStoreUnavailableError
    rather than holding the caller open.
    """
    start = time.monotonic()
    for attempt in range(1, _CONFIG_STORE_READ_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(path, params=params, timeout=timeout)
            response.raise_for_status()
        except Exception as exc:
            status_code = _config_store_http_status(exc)
            if status_code == _CONFIG_STORE_READ_MISSING_HTTP_STATUS:
                return None
            if not _config_store_is_retryable(exc):
                raise
            if attempt == _CONFIG_STORE_READ_MAX_ATTEMPTS:
                elapsed = time.monotonic() - start
                message = (
                    f"Failed to read Vespa config {operation} after "
                    f"{attempt} attempts over {elapsed:.3f}s: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(message)
                raise ConfigStoreUnavailableError(message) from exc

            delay = _config_store_visit_backoff_seconds(attempt)
            logger.warning(
                "Vespa config %s failed on attempt %s/%s with %s: %s; "
                "retrying in %.3fs",
                operation,
                attempt,
                _CONFIG_STORE_READ_MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
            continue

        return response.json()
    return None


class VespaConfigStore(ImmutableConfigStore):
    """
    Vespa-based configuration store with multi-tenant support.

    Stores configurations as Vespa documents in a dedicated schema.
    Implements ConfigStore interface using Vespa backend.

    Schema: config_metadata
    Document structure:
    {
        "fields": {
            "config_id": "tenant_id:scope:service:config_key",
            "tenant_id": "acme:production",
            "scope": "system",
            "service": "system",
            "config_key": "system_config",
            "config_value": {...},
            "version": 1,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00"
        }
    }
    """

    def __init__(
        self,
        vespa_app: Optional[Vespa] = None,
        backend_url: str = "http://localhost",
        backend_port: int = 8080,
        schema_name: str = "config_metadata",
        keep_versions: int = 10,
    ):
        """
        Initialize Vespa configuration store.

        Args:
            vespa_app: Existing Vespa application instance (optional)
            backend_url: Backend server URL
            backend_port: Backend server port
            schema_name: Vespa schema name for config storage
            keep_versions: Per-config_id, how many recent versions to retain
                after every ``set_config`` write. The default 10 is enough
                to roll back a few accidental changes while keeping each
                config's reads, which visit every retained version, small.
                Set to a higher number on environments that depend on long
                version history; set to 1 to keep only the latest at the
                cost of losing rollback.
        """
        if vespa_app is not None:
            self.vespa_app = vespa_app
        else:
            # Persistent session: config reads/writes are frequent and the
            # store lives for the process — per-op VespaSync handshakes
            # dominated cache-miss latency.
            self.vespa_app = make_persistent_vespa_ops(
                url=backend_url, port=backend_port
            )

        self.schema_name = schema_name
        self.keep_versions = max(1, keep_versions)
        logger.info(
            f"VespaConfigStore initialized with schema: {schema_name} "
            f"at {backend_url}:{backend_port} (keep_versions={self.keep_versions})"
        )

    @property
    def source(self) -> tuple[str, str, str]:
        """The canonical Vespa endpoint and the config schema it stores under."""
        return ("vespa", canonical_endpoint(self.vespa_app.url), self.schema_name)

    def close(self) -> None:
        """Release the persistent HTTP session (no-op for injected apps)."""
        close = getattr(self.vespa_app, "close", None)
        if callable(close):
            close()

    def initialize(self) -> None:
        """
        Initialize the configuration store.

        For Vespa, this assumes the schema already exists.
        Schema must be deployed separately via vespa-cli or application package.
        """
        # Check if schema exists by attempting a simple query
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            logger.info(f"Vespa schema '{self.schema_name}' is accessible")
        except Exception as e:
            logger.warning(
                f"Could not verify Vespa schema '{self.schema_name}': {e}. "
                "Ensure schema is deployed before using VespaConfigStore."
            )

    def _create_document_id(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> str:
        """Create Vespa document ID from config coordinates"""
        # Vespa doc ID: config_metadata::<config_id>::<version>
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"
        return config_id

    @staticmethod
    def _entry_from_fields(fields: Dict[str, Any]) -> ConfigEntry:
        return ConfigEntry(
            tenant_id=fields["tenant_id"],
            scope=ConfigScope(fields["scope"]),
            service=fields["service"],
            config_key=fields["config_key"],
            config_value=json.loads(fields["config_value"]),
            version=fields["version"],
            created_at=datetime.fromisoformat(fields["created_at"]),
            updated_at=datetime.fromisoformat(fields["updated_at"]),
        )

    def _visit_config_entries(
        self,
        *,
        tenant_id: Optional[str] = None,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
        config_key: Optional[str] = None,
        config_key_suffix: Optional[str] = None,
        latest_only: bool = False,
        skip_malformed: bool = False,
    ) -> List[tuple[str, ConfigEntry]]:
        """Visit the rows the selection names, newest-first per config_id.

        ``latest_only`` returns one entry per config_id — the highest version —
        and decodes only that one. A config's superseded versions are kept
        (``keep_versions``) and are large for documents like the system backend
        config, so a caller that wants the current value must not pay to decode
        the history it is about to discard.
        """
        path = (
            f"{self.vespa_app.url}/document/v1/"
            f"{self.schema_name}/{self.schema_name}/docid/"
        )
        selection_parts = []
        if tenant_id is not None:
            selection_parts.append(
                f"{self.schema_name}.tenant_id == {yql_quote(tenant_id)}"
            )
        if scope is not None:
            selection_parts.append(
                f"{self.schema_name}.scope == {yql_quote(scope.value)}"
            )
        if service is not None:
            selection_parts.append(
                f"{self.schema_name}.service == {yql_quote(service)}"
            )
        if config_key is not None:
            selection_parts.append(
                f"{self.schema_name}.config_key == {yql_quote(config_key)}"
            )
        if config_key_suffix is not None:
            # Document selection's glob match; the visit then carries only the
            # matching documents instead of every service row.
            selection_parts.append(
                f"{self.schema_name}.config_key = "
                f"{yql_quote(f'*{_literal_glob_suffix(config_key_suffix)}')}"
            )
        params: Dict[str, Any] = {"wantedDocumentCount": 1000}
        if selection_parts:
            params["selection"] = " and ".join(selection_parts)

        entries: List[tuple[str, ConfigEntry]] = []
        newest: Dict[str, Dict[str, Any]] = {}
        continuation: Optional[str] = None
        pages = 0
        while True:
            if continuation:
                params["continuation"] = continuation
            payload = _config_store_read_json(path, params=params, timeout=30)
            if payload is None:
                if continuation is None:
                    # No config_metadata document type yet: a genuinely empty
                    # store, not a traversal that stopped halfway.
                    return entries
                raise ConfigStoreUnavailableError(
                    f"Vespa config visit answered HTTP "
                    f"{_CONFIG_STORE_READ_MISSING_HTTP_STATUS} continuing past page "
                    f"{pages} ({len(entries)} documents read); refusing to return a "
                    f"partial traversal"
                )
            pages += 1
            documents = payload["documents"]
            if not isinstance(documents, list):
                raise ValueError("Vespa config visit documents must be a list")
            for document in documents:
                try:
                    fields = _visited_fields(document)
                    config_id = fields["config_id"]
                    if latest_only:
                        known = newest.get(config_id)
                        if known is None or int(fields["version"]) > int(
                            known["version"]
                        ):
                            newest[config_id] = fields
                        continue
                    entry = self._entry_from_fields(fields)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    if not skip_malformed:
                        raise
                    logger.warning(
                        "Skipping malformed config_metadata doc %s: %s",
                        document.get("id"),
                        exc,
                    )
                    continue
                entries.append((config_id, entry))
            continuation = payload.get("continuation")
            if not continuation:
                if latest_only:
                    for config_id, fields in newest.items():
                        try:
                            entries.append((config_id, self._entry_from_fields(fields)))
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ) as exc:
                            if not skip_malformed:
                                raise
                            logger.warning(
                                "Skipping malformed config_metadata doc %s: %s",
                                config_id,
                                exc,
                            )
                return entries

    def get_immutable_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> Optional[ConfigEntry]:
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        # Same bounded reader as list_immutable_configs: this read sits on the
        # request path (harness key resolution) and a backend that accepts and
        # never answers must end as an outage, not as an open request.
        path = self.vespa_app.url + self.vespa_app.get_document_v1_path(
            id=f"{self.schema_name}::{config_id}::1",
            schema=self.schema_name,
            namespace=self.schema_name,
        )
        payload = _config_store_read_json(
            path,
            params={},
            timeout=_CONFIG_STORE_DOCUMENT_READ_TIMEOUT_SECONDS,
            operation="document",
        )
        if payload is None:
            return None
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            raise ConfigStoreUnavailableError(
                f"Immutable config {config_id} came back without fields: {payload}"
            )
        return self._entry_from_fields(fields)

    def put_immutable_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        now = datetime.now(timezone.utc)
        entry = ConfigEntry(
            tenant_id, scope, service, config_key, config_value, 1, now, now
        )
        config_id = entry.get_config_id()
        fields = entry.to_dict()
        fields["config_id"] = config_id
        fields["config_value"] = json.dumps(config_value)
        try:
            response = self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=f"{self.schema_name}::{config_id}::1",
                fields=fields,
                condition=f"{self.schema_name}.version < 1",
                create=True,
            )
            if response.status_code not in (200, 201, 412):
                raise RuntimeError(f"HTTP {response.status_code}: {response.json}")
        except Exception as exc:
            if not _is_condition_miss(exc):
                raise ConfigStoreUnavailableError(
                    f"Failed to write immutable config at {self.vespa_app.url}: {exc}"
                ) from exc
        confirmed = self.get_immutable_config(tenant_id, scope, service, config_key)
        if confirmed is None:
            raise ConfigStoreUnavailableError(
                f"Immutable config {config_id} was not confirmed"
            )
        if confirmed.config_value != config_value:
            raise ValueError(f"immutable config {config_id} has a different value")
        return confirmed

    def list_immutable_configs(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        *,
        page_size: int = 100,
        continuation: Optional[str] = None,
    ) -> tuple[List[ConfigEntry], Optional[str]]:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        visit_continuation: Optional[str] = None
        pending: List[str] = []
        if continuation is not None:
            visit_continuation, pending = _decode_immutable_cursor(continuation)
        if pending:
            # Records an earlier visit returned past its page. One that was
            # deleted since is simply gone; the rest are served in order.
            page = [
                entry
                for key in pending[:page_size]
                if (entry := self.get_immutable_config(tenant_id, scope, service, key))
                is not None
            ]
            return page, _encode_immutable_cursor(
                visit_continuation, pending[page_size:]
            )
        if continuation is not None and visit_continuation is None:
            return [], None
        path = f"{self.vespa_app.url}/document/v1/{self.schema_name}/{self.schema_name}/docid/"
        params = {
            "wantedDocumentCount": page_size,
            "concurrency": 1,
            "selection": " and ".join(
                [
                    f"{self.schema_name}.tenant_id == {yql_quote(tenant_id)}",
                    f"{self.schema_name}.scope == {yql_quote(scope.value)}",
                    f"{self.schema_name}.service == {yql_quote(service)}",
                    f"{self.schema_name}.version == 1",
                ]
            ),
        }
        if visit_continuation is not None:
            params["continuation"] = visit_continuation
        payload = _config_store_read_json(path, params=params, timeout=5)
        if payload is None:
            return [], None
        entries = [
            self._entry_from_fields(_visited_fields(doc))
            for doc in payload["documents"]
        ]
        return entries[:page_size], _encode_immutable_cursor(
            payload.get("continuation"),
            [entry.config_key for entry in entries[page_size:]],
        )

    def _get_latest_version(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> int:
        """The config's latest stored version, 0 when it has none.

        Read by visiting the config's stored versions, as ``get_config`` does,
        so the answer never depends on the coverage a search query reports;
        Vespa answers a query with partial coverage while a concurrent write
        lands. A visit the store does not answer raises
        ``ConfigStoreUnavailableError`` rather than reading as no versions,
        which would make ``set_config`` write version 1 below the real latest.
        """
        latest = self.get_config(tenant_id, scope, service, config_key)
        return 0 if latest is None else latest.version

    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """
        Store or update a configuration entry.

        Creates a new version of the config. All updates are versioned.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            config_value: Configuration value (dict)

        Returns:
            ConfigEntry with new version number
        """
        entry = self._append_version(
            tenant_id, scope, service, config_key, config_value
        )
        self._prune_old_versions(
            self._create_document_id(tenant_id, scope, service, config_key),
            keep=self.keep_versions,
        )
        return entry

    def _read_counter(self, config_id: str) -> Optional[tuple[int, datetime]]:
        """The config's highest reserved version and when it was reserved.

        Read over the store's persistent session: a contended key re-reads
        its counter on every lost reservation, and a fresh connection per
        read exhausts the backend's connection handling under that load.
        """
        for attempt in range(1, _CONFIG_STORE_READ_MAX_ATTEMPTS + 1):
            try:
                response = self.vespa_app.get_data(
                    schema=self.schema_name,
                    namespace=_VERSION_COUNTER_NAMESPACE,
                    data_id=config_id,
                    raise_on_not_found=False,
                )
            except Exception as exc:
                if (
                    not _config_store_is_retryable(exc)
                    or attempt == _CONFIG_STORE_READ_MAX_ATTEMPTS
                ):
                    raise ConfigStoreUnavailableError(
                        f"Failed to read the version counter of config "
                        f"{config_id}: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(_config_store_visit_backoff_seconds(attempt))
                continue
            status = response.status_code
            if status == _CONFIG_STORE_READ_MISSING_HTTP_STATUS:
                return None
            if status == 200:
                fields = response.get_json().get("fields")
                if not isinstance(fields, dict):
                    raise ConfigStoreUnavailableError(
                        f"Version counter of config {config_id} came back "
                        f"without fields: {response.get_json()}"
                    )
                return (
                    int(fields["version"]),
                    datetime.fromisoformat(fields["updated_at"]),
                )
            # An overloaded backend answers a document GET with 429 or 5xx
            # and an error body; that read is retried like any other.
            retryable = (
                status == 429 or _CONFIG_STORE_READ_RETRYABLE_HTTP_STATUS_MIN <= status
            )
            if not retryable or attempt == _CONFIG_STORE_READ_MAX_ATTEMPTS:
                raise ConfigStoreUnavailableError(
                    f"Failed to read the version counter of config {config_id} "
                    f"after {attempt} attempts: HTTP {status}: "
                    f"{response.get_json()}"
                )
            time.sleep(_config_store_visit_backoff_seconds(attempt))
        raise AssertionError("unreachable")

    def _counter(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_id: str,
    ) -> tuple[int, datetime]:
        """Read the version counter, creating it at the stored latest version.

        Creation is conditional on the counter not existing, so concurrent
        first writers create it once and every one of them then reads it.
        """
        counter = self._read_counter(config_id)
        if counter is not None:
            return counter
        latest = self._get_latest_version(tenant_id, scope, service, config_key)
        try:
            self.vespa_app.feed_data_point(
                schema=self.schema_name,
                namespace=_VERSION_COUNTER_NAMESPACE,
                data_id=config_id,
                fields={
                    "version": latest,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                condition=f"{self.schema_name}.version < 0",
                create=True,
            )
        except Exception as error:
            if not _is_condition_miss(error):
                raise
        counter = self._read_counter(config_id)
        if counter is None:
            raise ConfigStoreUnavailableError(
                f"Version counter for config {config_id} was not readable after "
                "it was created"
            )
        return counter

    def _reserve(self, config_id: str, current: int, version: int) -> bool:
        """Move the counter from ``current`` to ``version``; False if it moved."""
        try:
            self.vespa_app.update_data(
                schema=self.schema_name,
                namespace=_VERSION_COUNTER_NAMESPACE,
                data_id=config_id,
                fields={
                    "version": version,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                condition=f"{self.schema_name}.version == {current}",
            )
        except Exception as error:
            if _is_condition_miss(error):
                return False
            raise
        return True

    def _write_reserved_version(self, entry: ConfigEntry) -> bool:
        """Write a reserved version's document; False if it already exists.

        The reservation makes this writer the only one assigned the version,
        so the condition only misses when the document was written by a
        writer that reserved the version before the counter was reset by
        ``delete_config``; the caller then reserves another version.
        """
        config_id = entry.get_config_id()
        fields = entry.to_dict()
        fields["config_id"] = config_id
        fields["config_value"] = json.dumps(entry.config_value)
        try:
            self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=f"{self.schema_name}::{config_id}::{entry.version}",
                fields=fields,
                condition=f"{self.schema_name}.version < {entry.version}",
                create=True,
            )
        except Exception as error:
            if _is_condition_miss(error):
                return False
            logger.error(f"Failed to store config in Vespa: {error}")
            raise
        return True

    def _append_version(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """Write the next version of a configuration without pruning.

        The version is reserved on the config's counter document first, then
        written; the counter is never pruned, so no two writers are handed
        the same version however long either stalls between its steps.
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        floor = 0
        current, _ = self._counter(tenant_id, scope, service, config_key, config_id)
        for attempt in range(1, _MAX_VERSION_ALLOCATION_ATTEMPTS + 1):
            version = max(current, floor) + 1
            if not self._reserve(config_id, current, version):
                # The counter only moves forward, so the next value to try is
                # one past the one that missed; a re-read every few misses
                # catches up with a counter far ahead.
                if attempt % _COUNTER_REREAD_EVERY_MISSES == 0:
                    current, _ = self._counter(
                        tenant_id, scope, service, config_key, config_id
                    )
                else:
                    current += 1
                continue
            now = datetime.now(timezone.utc)
            entry = ConfigEntry(
                tenant_id=tenant_id,
                scope=scope,
                service=service,
                config_key=config_key,
                config_value=config_value,
                version=version,
                created_at=now,
                updated_at=now,
            )
            if self._write_reserved_version(entry):
                logger.info(f"Set config {config_id} v{version} in Vespa")
                return entry
            floor = self._get_latest_version(tenant_id, scope, service, config_key)
            current = version

        raise RuntimeError(
            f"Could not allocate a version for config {config_id} after "
            f"{_MAX_VERSION_ALLOCATION_ATTEMPTS} conditional writes"
        )

    def compare_and_set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
        *,
        expected_version: int,
    ) -> Optional[ConfigEntry]:
        """Append exactly the next version, or return None on contention.

        Version zero requires an absent key. The version is reserved on the
        config's counter only while no other writer holds a reservation past
        ``expected_version``; a reservation older than
        ``_ABANDONED_RESERVATION_SECONDS`` whose version never appeared is
        reserved past. The write is reported only when a strong read then
        shows it as the latest version.
        """
        if expected_version < 0:
            raise ValueError("expected_version must be nonnegative")
        current = self.get_config(tenant_id, scope, service, config_key)
        actual_version = 0 if current is None else current.version
        if actual_version != expected_version:
            return None
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        reserved, reserved_at = self._counter(
            tenant_id, scope, service, config_key, config_id
        )
        if (
            reserved > expected_version
            and (datetime.now(timezone.utc) - reserved_at).total_seconds()
            < _ABANDONED_RESERVATION_SECONDS
        ):
            return None
        version = max(reserved, expected_version) + 1
        if not self._reserve(config_id, reserved, version):
            return None
        now = datetime.now(timezone.utc)
        entry = ConfigEntry(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
            version=version,
            created_at=now,
            updated_at=now,
        )
        if not self._write_reserved_version(entry):
            return None
        latest = self.get_config(tenant_id, scope, service, config_key)
        self._prune_old_versions(config_id, keep=self.keep_versions)
        if latest is None or latest.version != entry.version:
            return None
        return entry

    def _prune_old_versions(self, config_id: str, *, keep: int) -> int:
        """Delete every version of ``config_id`` older than the latest ``keep``.

        Vespa's only delete primitive is per-document; iterate the
        sorted version list and drop everything beyond the head ``keep``
        entries. Best-effort — a listing or delete failure is logged but does
        not propagate, since the leading set_config write already succeeded
        and a stale row only costs query latency, not correctness. A degraded
        listing (``root.errors``, or a hit without its version: a row another
        writer deleted between match and summary fill) prunes nothing; the
        next write prunes again.
        """
        if keep < 1:
            return 0
        yql = (
            f"select version from {self.schema_name} "
            f"where config_id contains {yql_quote(config_id)} "
            f"order by version desc limit {keep + 100}"
        )
        try:
            response = self.vespa_app.query(yql=yql)
            _raise_if_degraded(response, config_id)
            versions = [
                _config_hit_fields(hit, ("version",), config_id)["version"]
                for hit in response.hits or []
            ]
        except (RequestException, VespaError, RuntimeError) as exc:
            logger.warning(f"Could not list versions to prune {config_id!r}: {exc}")
            return 0
        if len(versions) <= keep:
            return 0
        dropped = 0
        for version in versions[keep:]:
            doc_id = f"{self.schema_name}::{config_id}::{version}"
            try:
                self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
                dropped += 1
            except (RequestException, VespaError) as exc:
                logger.warning(f"Failed to prune {config_id!r} v{version}: {exc}")
        if dropped:
            logger.info(
                f"Pruned {dropped} old versions of {config_id!r} (kept latest {keep})"
            )
        return dropped

    def count_version_rows(self) -> Dict[str, int]:
        """Count EVERY stored version row per config_id via a Document v1 visit.

        ``list_all_configs`` returns only latest versions; the prune dry-run
        needs the full per-id row counts to report what pruning would drop.
        """
        url = f"{self.vespa_app.url}/document/v1/"
        path = f"{url}{self.schema_name}/{self.schema_name}/docid/"
        params: Dict[str, Any] = {"wantedDocumentCount": 1000}
        counts: Dict[str, int] = {}
        continuation: Optional[str] = None
        while True:
            if continuation:
                params["continuation"] = continuation
            payload = _config_store_read_json(path, params=params, timeout=60)
            if payload is None:
                return counts
            for doc in payload.get("documents") or []:
                cid = (doc.get("fields") or {}).get("config_id")
                if cid:
                    counts[cid] = counts.get(cid, 0) + 1
            continuation = payload.get("continuation")
            if not continuation:
                break
        return counts

    def prune_all_configs(self, *, keep: Optional[int] = None) -> int:
        """One-shot prune across every config_id in the schema.

        Walks ``config_metadata`` via the Document v1 visit API,
        collects every distinct ``config_id``, and applies
        ``_prune_old_versions`` to each with the configured retention
        window. Use to drain pre-existing bloat that accumulated before
        per-write pruning was added to ``set_config``. Returns the total
        number of stale version rows deleted.
        """
        keep = self.keep_versions if keep is None else max(1, keep)
        url = f"{self.vespa_app.url}/document/v1/"
        path = f"{url}{self.schema_name}/{self.schema_name}/docid/"
        params: Dict[str, Any] = {"wantedDocumentCount": 1000}

        seen: set[str] = set()
        continuation: Optional[str] = None
        while True:
            if continuation:
                params["continuation"] = continuation
            payload = _config_store_read_json(path, params=params, timeout=60)
            if payload is None:
                break
            for doc in payload.get("documents") or []:
                fields = doc.get("fields") or {}
                config_id = fields.get("config_id")
                if config_id:
                    seen.add(config_id)
            continuation = payload.get("continuation")
            if not continuation:
                break

        total_dropped = 0
        for config_id in sorted(seen):
            total_dropped += self._prune_old_versions(config_id, keep=keep)
        logger.info(
            f"prune_all_configs: drained {total_dropped} stale versions "
            f"across {len(seen)} config_ids (kept latest {keep} per id)"
        )
        return total_dropped

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """
        Retrieve a configuration entry.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            version: Specific version (None = latest)

        Returns:
            ConfigEntry if found, None otherwise
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        try:
            matches = [
                entry
                for visited_id, entry in self._visit_config_entries(
                    tenant_id=tenant_id,
                    scope=scope,
                    service=service,
                    config_key=config_key,
                    latest_only=version is None,
                )
                if visited_id == config_id
                and (version is None or entry.version == int(version))
            ]
            if not matches:
                return None
            return max(matches, key=lambda entry: entry.version)
        except Exception as e:
            logger.error(f"Failed to retrieve config from Vespa: {e!r}")
            raise

    def get_config_history(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        limit: int = 10,
    ) -> List[ConfigEntry]:
        """
        Get configuration history (all versions).

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            limit: Maximum number of versions to return

        Returns:
            List of ConfigEntry sorted by version (newest first)
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        try:
            entries = [
                entry
                for visited_id, entry in self._visit_config_entries(
                    tenant_id=tenant_id,
                    scope=scope,
                    service=service,
                    config_key=config_key,
                )
                if visited_id == config_id
            ]
            return sorted(
                entries,
                key=lambda entry: entry.version,
                reverse=True,
            )[: max(0, int(limit))]
        except Exception as e:
            logger.error(f"Failed to retrieve config history from Vespa: {e!r}")
            raise

    def list_configs(
        self,
        tenant_id: str,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations matching criteria.

        Returns only latest versions.

        Args:
            tenant_id: Tenant identifier
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)

        Returns:
            List of latest version ConfigEntry objects
        """
        try:
            latest_configs: Dict[str, ConfigEntry] = {}
            for config_id, entry in self._visit_config_entries(
                tenant_id=tenant_id,
                scope=scope,
                service=service,
            ):
                if (
                    config_id not in latest_configs
                    or entry.version > latest_configs[config_id].version
                ):
                    latest_configs[config_id] = entry
            return list(latest_configs.values())
        except Exception as e:
            logger.error(f"Failed to list configs from Vespa: {e!r}")
            raise

    def list_all_configs(
        self,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
        config_key_suffix: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations across all tenants.

        Returns only latest versions.

        Args:
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)
            config_key_suffix: Keep only rows whose config_key ends with it
                (None = every key). Narrows the visit itself, so a caller that
                wants one owner's rows out of a service does not carry every
                other owner's back.

        Returns:
            List of latest version ConfigEntry objects from all tenants

        Uses the Document v1 visit API for read-after-write consistency.
        Vespa's search endpoint is eventually consistent and races
        cross-process schema_registry writes.
        """
        latest_configs: Dict[str, ConfigEntry] = {}
        try:
            for config_id, entry in self._visit_config_entries(
                scope=scope,
                service=service,
                config_key_suffix=config_key_suffix,
                skip_malformed=True,
            ):
                if (
                    config_id not in latest_configs
                    or entry.version > latest_configs[config_id].version
                ):
                    latest_configs[config_id] = entry
            return list(latest_configs.values())
        except Exception as e:
            logger.error(f"Failed to list all configs from Vespa: {e!r}")
            raise

    def delete_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> bool:
        """
        Delete all versions of a configuration entry.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key

        Returns:
            True if deleted, False if not found
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        # Get all versions
        history = self.get_config_history(
            tenant_id, scope, service, config_key, limit=1000
        )

        if not history:
            self._delete_counter(config_id)
            return False

        # Delete each version
        deleted_count = 0
        failures: List[tuple[int, Exception]] = []
        for entry in history:
            doc_id = f"{self.schema_name}::{config_id}::{entry.version}"
            try:
                self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete version {entry.version}: {e}")
                failures.append((entry.version, e))

        logger.info(
            f"Deleted {deleted_count} versions of config "
            f"{tenant_id}:{scope.value}:{service}:{config_key}"
        )

        if failures:
            details = "; ".join(
                f"version {version}: {error}" for version, error in failures
            )
            raise RuntimeError(
                f"Failed to delete {len(failures)} of {len(history)} versions "
                f"for {tenant_id}:{scope.value}:{service}:{config_key}: {details}"
            ) from failures[0][1]

        self._delete_counter(config_id)
        return True

    def _delete_counter(self, config_id: str) -> None:
        """Remove the version counter once no version of the config remains,
        so a recreated config starts again at version 1."""
        self.vespa_app.delete_data(
            schema=self.schema_name,
            namespace=_VERSION_COUNTER_NAMESPACE,
            data_id=config_id,
        )

    def export_configs(
        self,
        tenant_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Export all configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            include_history: Include all versions (True) or just latest (False)

        Schema-scope rows are left out: they record the schema registry's
        deployments, lease and deployment journal for this tenant's Vespa
        schemas, not configuration another tenant can take.

        Returns:
            Dictionary with all configurations, ordered by config id then
            ascending version. ``import_configs`` replays the rows in file
            order through ``set_config``, so the last row of each key decides
            the restored active value; ascending version makes that the
            exported latest.
        """
        try:
            if include_history:
                # Every retained version, over the same complete Document v1
                # traversal list_configs uses. A bounded query would report a
                # truncated backup as a successful one.
                configs = [
                    entry
                    for _, entry in self._visit_config_entries(tenant_id=tenant_id)
                ]
            else:
                configs = self.list_configs(tenant_id)
            configs = sorted(
                (c for c in configs if c.scope != ConfigScope.SCHEMA),
                key=lambda c: (c.get_config_id(), c.version),
            )

            return {
                "tenant_id": tenant_id,
                "include_history": include_history,
                "configs": [
                    {
                        "tenant_id": c.tenant_id,
                        "scope": c.scope.value,
                        "service": c.service,
                        "config_key": c.config_key,
                        "config_value": c.config_value,
                        "version": c.version,
                        "created_at": c.created_at.isoformat(),
                        "updated_at": c.updated_at.isoformat(),
                    }
                    for c in configs
                ],
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # A degraded or failed read must raise, not return an empty or
            # partial export a caller would persist as authoritative —
            # matching get_config / list_configs / get_config_history. Only a
            # genuinely empty tenant returns an empty configs list.
            logger.error(f"Failed to export configs from Vespa: {e}")
            raise

    def import_configs(
        self,
        tenant_id: str,
        configs: Dict[str, Any],
    ) -> int:
        """
        Import configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            configs: Dictionary of configurations to import

        Returns:
            Number of configurations imported

        Raises:
            ValueError: The payload carries schema-scope rows. Those are
                written only by the schema registry once Vespa holds the
                schema, so the whole payload is refused before any write.
            RuntimeError: A row could not be written. The import is all or
                nothing: every version it had already written is removed
                before this is raised, and versions older than the ones it
                wrote are pruned only once every row is written.
        """
        config_entries = configs.get("configs", [])
        schema_rows = [
            f"{entry.get('service')}/{entry.get('config_key')}"
            for entry in config_entries
            if isinstance(entry, dict)
            and entry.get("scope") == ConfigScope.SCHEMA.value
        ]
        if schema_rows:
            raise ValueError(
                f"Configuration import for tenant {tenant_id} refused: schema rows "
                "record deployments made by the schema registry and are not "
                f"importable: {', '.join(schema_rows)}"
            )

        written: List[ConfigEntry] = []
        for index, config_data in enumerate(config_entries):
            try:
                written.append(
                    self._append_version(
                        tenant_id=tenant_id,
                        scope=ConfigScope(config_data["scope"]),
                        service=config_data["service"],
                        config_key=config_data["config_key"],
                        config_value=config_data["config_value"],
                    )
                )
            except Exception as error:
                self._abort_import(tenant_id, config_entries, index, written, error)

        for config_id in dict.fromkeys(
            self._create_document_id(
                entry.tenant_id, entry.scope, entry.service, entry.config_key
            )
            for entry in written
        ):
            self._prune_old_versions(config_id, keep=self.keep_versions)

        logger.info(f"Imported {len(written)} configs for tenant {tenant_id}")
        return len(written)

    def _abort_import(
        self,
        tenant_id: str,
        config_entries: List[Any],
        failed_index: int,
        written: List[ConfigEntry],
        error: Exception,
    ) -> NoReturn:
        """Remove every version an import wrote, then raise its failure."""
        config_data = config_entries[failed_index]
        label = (
            f"{config_data.get('service')}/{config_data.get('config_key')}"
            if isinstance(config_data, dict)
            else f"index {failed_index}"
        )
        left: List[str] = []
        for entry in reversed(written):
            config_id = self._create_document_id(
                entry.tenant_id, entry.scope, entry.service, entry.config_key
            )
            try:
                self.vespa_app.delete_data(
                    schema=self.schema_name,
                    data_id=f"{self.schema_name}::{config_id}::{entry.version}",
                )
            except Exception as delete_error:
                left.append(
                    f"{entry.service}/{entry.config_key} v{entry.version}: "
                    f"{delete_error}"
                )
        message = (
            f"Configuration import for tenant {tenant_id} failed at row "
            f"{failed_index + 1} of {len(config_entries)} ({label}): {error}; "
            f"removed {len(written) - len(left)} of the {len(written)} versions "
            "it had written"
        )
        if left:
            message += f"; still stored: {'; '.join(left)}"
        logger.error(message)
        raise RuntimeError(message) from error

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            entries = self._visit_config_entries(skip_malformed=True)
            scope_counts: Dict[str, int] = {}
            for _, entry in entries:
                scope_counts[entry.scope.value] = (
                    scope_counts.get(entry.scope.value, 0) + 1
                )

            return {
                "total_configs": len({config_id for config_id, _ in entries}),
                "total_versions": len(entries),
                "total_tenants": len({entry.tenant_id for _, entry in entries}),
                "configs_per_scope": scope_counts,
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
            }

        except Exception as e:
            # An outage or degraded read must raise, not report zero configs /
            # zero tenants — a dashboard keyed off these counts would show an
            # empty store during a Vespa blip. Matches the raising sibling reads.
            logger.error(f"Failed to get stats from Vespa: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            return True
        except Exception as e:
            logger.error(f"Vespa health check failed: {e}")
            return False
