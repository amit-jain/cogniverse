"""
Vespa-based configuration storage with multi-tenant support.
Stores configurations directly in Vespa backend for unified storage.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vespa.application import Vespa

from cogniverse_sdk.interfaces.config_store import (
    ConfigEntry,
    ConfigScope,
    ConfigStore,
)
from cogniverse_vespa._vespa_factory import make_persistent_vespa_ops
from cogniverse_vespa._yql import yql_quote

logger = logging.getLogger(__name__)


class VespaConfigStore(ConfigStore):
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
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
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
                to roll back a few accidental changes while keeping
                ``_get_latest_version`` queries fast. Set to a higher
                number on environments that depend on long version
                history; set to 1 to keep only the latest at the cost of
                losing rollback.
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

    def _get_latest_version(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> int:
        """Get latest version number for a config"""
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        # Query for all versions of this config
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        yql = (
            f"select version from {self.schema_name} "
            f"where config_id contains {yql_quote(config_id)} "
            f"order by version desc limit 1"
        )

        try:
            response = self.vespa_app.query(yql=yql)
        except Exception as e:
            # A backend read failure must not be flattened to 0 — set_config
            # would treat a live config as brand-new (v1) and overwrite its
            # real v1 row. Raise so the write aborts.
            logger.error(f"Failed to query latest config version: {e!r}")
            raise
        if response.hits and len(response.hits) > 0:
            return response.hits[0]["fields"]["version"]
        return 0

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
        # Get next version
        current_version = self._get_latest_version(
            tenant_id, scope, service, config_key
        )
        new_version = current_version + 1

        # Create timestamps
        now = datetime.now()
        created_at = now if new_version == 1 else None  # Only set on first version
        updated_at = now

        # Create config entry
        entry = ConfigEntry(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
            version=new_version,
            created_at=created_at or now,
            updated_at=updated_at,
        )

        # Create Vespa document ID
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        doc_id = f"{self.schema_name}::{config_id}::{new_version}"

        # Prepare document fields
        fields = {
            "config_id": config_id,
            "tenant_id": tenant_id,
            "scope": scope.value,
            "service": service,
            "config_key": config_key,
            "config_value": json.dumps(config_value),
            "version": new_version,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

        # Feed document to Vespa
        try:
            self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=doc_id,
                fields=fields,
            )

            logger.info(f"Set config {entry.get_config_id()} v{new_version} in Vespa")

            # Prune old versions to keep config_metadata size bounded.
            # Without this every set_config call appends a new doc and old
            # versions stick around forever — observed 5800+ backend_config
            # rows after ~4 days of dev work, with each
            # _get_latest_version query slowing as the table grew.
            self._prune_old_versions(config_id, keep=self.keep_versions)

            return entry

        except Exception as e:
            logger.error(f"Failed to store config in Vespa: {e}")
            raise

    def _prune_old_versions(self, config_id: str, *, keep: int) -> int:
        """Delete every version of ``config_id`` older than the latest ``keep``.

        Vespa's only delete primitive is per-document; iterate the
        sorted version list and drop everything beyond the head ``keep``
        entries. Best-effort — a delete failure is logged but does not
        propagate, since the leading set_config write already succeeded
        and a stale row only costs query latency, not correctness.
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
        except Exception as exc:  # noqa: BLE001 — pruning is best-effort
            logger.warning(f"Could not list versions to prune {config_id!r}: {exc}")
            return 0
        hits = list(response.hits or [])
        if len(hits) <= keep:
            return 0
        stale = hits[keep:]
        dropped = 0
        for hit in stale:
            version = hit["fields"]["version"]
            doc_id = f"{self.schema_name}::{config_id}::{version}"
            try:
                self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
                dropped += 1
            except Exception as exc:  # noqa: BLE001 — best-effort prune
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
        import requests

        url = f"{self.vespa_app.url}/document/v1/"
        path = f"{url}{self.schema_name}/{self.schema_name}/docid/"
        params: Dict[str, Any] = {"wantedDocumentCount": 1000}
        counts: Dict[str, int] = {}
        continuation: Optional[str] = None
        while True:
            if continuation:
                params["continuation"] = continuation
            resp = requests.get(path, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
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
        import requests

        keep = self.keep_versions if keep is None else max(1, keep)
        url = f"{self.vespa_app.url}/document/v1/"
        path = f"{url}{self.schema_name}/{self.schema_name}/docid/"
        params: Dict[str, Any] = {"wantedDocumentCount": 1000}

        seen: set[str] = set()
        try:
            continuation: Optional[str] = None
            while True:
                if continuation:
                    params["continuation"] = continuation
                resp = requests.get(path, params=params, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
                for doc in payload.get("documents") or []:
                    fields = doc.get("fields") or {}
                    config_id = fields.get("config_id")
                    if config_id:
                        seen.add(config_id)
                continuation = payload.get("continuation")
                if not continuation:
                    break
        except Exception as exc:  # noqa: BLE001 — best-effort drain
            logger.error(f"prune_all_configs: visit failed: {exc}")
            return 0

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

        # Build YQL query
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        if version is None:
            # Get latest version
            yql = (
                f"select * from {self.schema_name} "
                f"where config_id contains {yql_quote(config_id)} "
                f"order by version desc limit 1"
            )
        else:
            # Get specific version. Query the indexed config_id + version
            # fields — Vespa has no queryable ``documentid`` field, so a
            # ``where documentid = ...`` clause fails with a 400.
            yql = (
                f"select * from {self.schema_name} "
                f"where config_id contains {yql_quote(config_id)} "
                f"and version = {int(version)}"
            )

        try:
            response = self.vespa_app.query(yql=yql)

            if not response.hits or len(response.hits) == 0:
                return None

            # Parse first hit
            hit = response.hits[0]["fields"]

            return ConfigEntry(
                tenant_id=hit["tenant_id"],
                scope=ConfigScope(hit["scope"]),
                service=hit["service"],
                config_key=hit["config_key"],
                config_value=json.loads(hit["config_value"]),
                version=hit["version"],
                created_at=datetime.fromisoformat(hit["created_at"]),
                updated_at=datetime.fromisoformat(hit["updated_at"]),
            )

        except Exception as e:
            # A genuinely-absent config already returned None above; reaching
            # here means the backend read FAILED. Raise so callers don't
            # silently fall back to default config during a Vespa outage.
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

        # Vespa rejects a YQL ``limit`` above its default max-hits (400); config
        # history is pruned to a small keep-window, so this ceiling never
        # truncates real data but keeps the query valid.
        query_limit = min(int(limit), 400)
        yql = (
            f"select * from {self.schema_name} "
            f"where config_id contains {yql_quote(config_id)} "
            f"order by version desc limit {query_limit}"
        )

        try:
            response = self.vespa_app.query(yql=yql)

            entries = []
            for hit in response.hits:
                fields = hit["fields"]
                entries.append(
                    ConfigEntry(
                        tenant_id=fields["tenant_id"],
                        scope=ConfigScope(fields["scope"]),
                        service=fields["service"],
                        config_key=fields["config_key"],
                        config_value=json.loads(fields["config_value"]),
                        version=fields["version"],
                        created_at=datetime.fromisoformat(fields["created_at"]),
                        updated_at=datetime.fromisoformat(fields["updated_at"]),
                    )
                )

            return entries

        except Exception as e:
            # A backend read FAILURE must not read as "no history" — raise,
            # matching get_config's contract.
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
        # Build YQL query with filters
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        conditions = [f"tenant_id contains {yql_quote(tenant_id)}"]

        if scope is not None:
            conditions.append(f"scope contains {yql_quote(scope.value)}")

        if service is not None:
            conditions.append(f"service contains {yql_quote(service)}")

        where_clause = " and ".join(conditions)

        # Query all matching configs, then filter to latest versions
        # Note: This is a simplified approach - for production, consider using
        # Vespa grouping or ranking to get only latest versions efficiently
        yql = f"select * from {self.schema_name} where {where_clause} limit 400"

        try:
            response = self.vespa_app.query(yql=yql)

            # Group by config_id and keep only latest version
            latest_configs: Dict[str, ConfigEntry] = {}

            for hit in response.hits:
                fields = hit["fields"]
                config_id = fields["config_id"]

                entry = ConfigEntry(
                    tenant_id=fields["tenant_id"],
                    scope=ConfigScope(fields["scope"]),
                    service=fields["service"],
                    config_key=fields["config_key"],
                    config_value=json.loads(fields["config_value"]),
                    version=fields["version"],
                    created_at=datetime.fromisoformat(fields["created_at"]),
                    updated_at=datetime.fromisoformat(fields["updated_at"]),
                )

                # Keep only latest version for each config_id
                if (
                    config_id not in latest_configs
                    or entry.version > latest_configs[config_id].version
                ):
                    latest_configs[config_id] = entry

            return list(latest_configs.values())

        except Exception as e:
            # A backend read FAILURE must not read as "no configs" — raise,
            # matching get_config's contract.
            logger.error(f"Failed to list configs from Vespa: {e!r}")
            raise

    def list_all_configs(
        self,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations across all tenants.

        Returns only latest versions.

        Args:
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)

        Returns:
            List of latest version ConfigEntry objects from all tenants

        Uses the Document v1 visit API for read-after-write consistency.
        Vespa's search endpoint is eventually consistent and races
        cross-process schema_registry writes.
        """
        import requests

        url = f"{self.vespa_app.url}/document/v1/"
        namespace = self.schema_name
        path = f"{url}{namespace}/{self.schema_name}/docid/"
        selection_parts = []
        if scope is not None:
            selection_parts.append(
                f"{self.schema_name}.scope == {yql_quote(scope.value)}"
            )
        if service is not None:
            selection_parts.append(
                f"{self.schema_name}.service == {yql_quote(service)}"
            )
        params: Dict[str, Any] = {"wantedDocumentCount": 400}
        if selection_parts:
            params["selection"] = " and ".join(selection_parts)

        latest_configs: Dict[str, ConfigEntry] = {}

        try:
            continuation: Optional[str] = None
            while True:
                if continuation:
                    params["continuation"] = continuation
                resp = requests.get(path, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                for doc in payload.get("documents") or []:
                    fields = doc.get("fields") or {}
                    if not fields:
                        continue
                    try:
                        config_id = fields["config_id"]
                        entry = ConfigEntry(
                            tenant_id=fields["tenant_id"],
                            scope=ConfigScope(fields["scope"]),
                            service=fields["service"],
                            config_key=fields["config_key"],
                            config_value=json.loads(fields["config_value"]),
                            version=fields["version"],
                            created_at=datetime.fromisoformat(fields["created_at"]),
                            updated_at=datetime.fromisoformat(fields["updated_at"]),
                        )
                    except (KeyError, ValueError, json.JSONDecodeError) as exc:
                        # Skip a single malformed peer document — must not
                        # bomb the whole list and leave the schema_registry
                        # at 0 entries. A bad config_value JSON or missing
                        # field on ONE doc otherwise dropped every other
                        # tenant's schemas, causing deploy_schema to refuse
                        # any new schema (Vespa-deployed > registry-known).
                        logger.warning(
                            "Skipping malformed config_metadata doc %s: %s",
                            doc.get("id"),
                            exc,
                        )
                        continue
                    if (
                        config_id not in latest_configs
                        or entry.version > latest_configs[config_id].version
                    ):
                        latest_configs[config_id] = entry
                continuation = payload.get("continuation")
                if not continuation:
                    break

            return list(latest_configs.values())

        except Exception as e:
            # An outage read as "no configs at all" makes the schema registry
            # load zero schemas and wipe its cache — raise so its designed
            # keep-existing-cache fallback fires instead.
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
            return False

        # Delete each version
        deleted_count = 0
        for entry in history:
            doc_id = f"{self.schema_name}::{config_id}::{entry.version}"
            try:
                self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete version {entry.version}: {e}")

        logger.info(
            f"Deleted {deleted_count} versions of config "
            f"{tenant_id}:{scope.value}:{service}:{config_key}"
        )

        return deleted_count > 0

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

        Returns:
            Dictionary with all configurations
        """
        if include_history:
            # Get all versions
            yql = f"select * from {self.schema_name} where tenant_id contains {yql_quote(tenant_id)} limit 400"
        else:
            # Get only latest versions
            configs = self.list_configs(tenant_id)
            return {
                "tenant_id": tenant_id,
                "include_history": False,
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
                "exported_at": datetime.now().isoformat(),
            }

        try:
            response = self.vespa_app.query(yql=yql)

            configs = []
            for hit in response.hits:
                fields = hit["fields"]
                configs.append(
                    {
                        "tenant_id": fields["tenant_id"],
                        "scope": fields["scope"],
                        "service": fields["service"],
                        "config_key": fields["config_key"],
                        "config_value": json.loads(fields["config_value"]),
                        "version": fields["version"],
                        "created_at": fields["created_at"],
                        "updated_at": fields["updated_at"],
                    }
                )

            return {
                "tenant_id": tenant_id,
                "include_history": True,
                "configs": configs,
                "exported_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to export configs from Vespa: {e}")
            return {
                "tenant_id": tenant_id,
                "include_history": include_history,
                "configs": [],
                "exported_at": datetime.now().isoformat(),
                "error": str(e),
            }

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
        """
        imported_count = 0

        for config_data in configs.get("configs", []):
            try:
                self.set_config(
                    tenant_id=tenant_id,
                    scope=ConfigScope(config_data["scope"]),
                    service=config_data["service"],
                    config_key=config_data["config_key"],
                    config_value=config_data["config_value"],
                )
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import config: {e}")

        logger.info(f"Imported {imported_count} configs for tenant {tenant_id}")
        return imported_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # Select all fields needed for stats
            yql_total = f"select config_id, tenant_id, scope from {self.schema_name} where true limit 400"
            response = self.vespa_app.query(yql=yql_total)

            total_versions = len(response.hits)
            unique_config_ids = len(
                set(hit["fields"]["config_id"] for hit in response.hits)
            )

            # Count tenants
            unique_tenants = len(
                set(hit["fields"]["tenant_id"] for hit in response.hits)
            )

            # Count per scope
            scope_counts: Dict[str, int] = {}
            for hit in response.hits:
                scope = hit["fields"]["scope"]
                scope_counts[scope] = scope_counts.get(scope, 0) + 1

            return {
                "total_configs": unique_config_ids,
                "total_versions": total_versions,
                "total_tenants": unique_tenants,
                "configs_per_scope": scope_counts,
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
            }

        except Exception as e:
            logger.error(f"Failed to get stats from Vespa: {e}")
            return {
                "total_configs": 0,
                "total_versions": 0,
                "total_tenants": 0,
                "configs_per_scope": {},
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
                "error": str(e),
            }

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
