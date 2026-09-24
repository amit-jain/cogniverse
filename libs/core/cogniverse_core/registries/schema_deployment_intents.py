"""Durable, non-destructive completion of interrupted schema registration."""

import copy
import json
from collections.abc import Callable
from time import time as _now
from typing import Any

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, canonical_tenant_id
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_sdk.interfaces.config_store import ConfigScope, ConfigStore

_SERVICE = "schema_deployment_intents"
_MAX_ATTEMPTS = 3
_REGISTRATION_FIELDS = {
    "tenant_id",
    "base_schema_name",
    "full_schema_name",
    "schema_definition",
    "config",
    "deployment_time",
}


class SchemaDeploymentIntents:
    """Reserve each full schema name and condition journal transitions on revision.

    Absent records retain their immutable definition for late activation.
    Registration uses a separate conditional write fenced by registry_version.
    """

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    @staticmethod
    def _validate(registration: dict[str, Any]) -> None:
        name = registration.get("full_schema_name")
        try:
            tenant = registration["tenant_id"]
            base = registration["base_schema_name"]
            expected = f"{base}_{canonical_tenant_id(tenant).replace(':', '_')}"
            definition = json.loads(registration["schema_definition"])
            if (
                set(registration) != _REGISTRATION_FIELDS
                or canonical_tenant_id(tenant) != tenant
                or name != expected
                or definition["name"] != name
            ):
                raise ValueError("registration identity and definition disagree")
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryStorageError(
                f"Invalid deployment intent for {name!r}: {exc}"
            ) from exc

    def _read(self, name: str) -> dict[str, Any] | None:
        try:
            entry = self._store.get_config(
                tenant_id=SYSTEM_TENANT_ID,
                scope=ConfigScope.SCHEMA,
                service=_SERVICE,
                config_key=name,
            )
        except Exception as exc:
            raise RegistryStorageError(
                f"Cannot read deployment intent for {name!r}: {exc}"
            ) from exc
        return (
            None
            if entry is None
            else {**entry.config_value, "_revision": entry.version}
        )

    def _save(self, record: dict[str, Any]) -> dict[str, Any] | None:
        from cogniverse_core.registries.schema_registry import (
            invalidate_deployed_schema_names,
        )

        value = {key: item for key, item in record.items() if key != "_revision"}
        name = value["registration"]["full_schema_name"]
        try:
            entry = self._store.compare_and_set_config(
                tenant_id=SYSTEM_TENANT_ID,
                scope=ConfigScope.SCHEMA,
                service=_SERVICE,
                config_key=name,
                config_value=value,
                expected_version=record.get("_revision", 0),
            )
            return None if entry is None else {**value, "_revision": entry.version}
        except Exception as exc:
            raise RegistryStorageError(
                f"Cannot persist deployment intent for {name!r}: {exc}"
            ) from exc
        finally:
            invalidate_deployed_schema_names(value["registration"]["tenant_id"])

    def prepare(
        self, registration: dict[str, Any], *, grace_s: float, registry_version: int = 0
    ) -> dict[str, Any]:
        """Reserve the name and persist the exact payload before activation."""
        self._validate(registration)
        name = registration["full_schema_name"]
        for _ in range(_MAX_ATTEMPTS):
            current = self._read(name)
            if current:
                old = current["registration"]
                if (old["tenant_id"], old["base_schema_name"]) != (
                    registration["tenant_id"],
                    registration["base_schema_name"],
                ):
                    raise RegistryStorageError(
                        f"Schema name {name!r} is reserved for another tenant"
                    )
                if registry_version < current["registry_version"]:
                    raise RegistryStorageError(
                        f"Stale registry version {registry_version} for deployment intent {name!r}; "
                        f"current generation uses {current['registry_version']}"
                    )
                if current["registry_version"] == registry_version:
                    if {k: v for k, v in old.items() if k != "deployment_time"} != {
                        k: v for k, v in registration.items() if k != "deployment_time"
                    }:
                        raise RegistryStorageError(
                            f"Unresolved deployment intent for {name!r} has a different payload"
                        )
                    return current
                if current["state"] != "complete":
                    raise RegistryStorageError(
                        f"Unresolved deployment intent for {name!r} cannot advance registry generation"
                    )
            record = {
                "registration": copy.deepcopy(registration),
                "recover_after": _now() + grace_s,
                "registry_version": registry_version,
                "state": "pending",
                "attempts": 0,
                "_revision": 0 if current is None else current["_revision"],
            }
            saved = self._save(record)
            if saved:
                return saved
        raise RegistryStorageError(
            f"Cannot reserve deployment intent for {name!r}: concurrent writers"
        )

    def records(self, config_key_suffix: str | None = None) -> list[dict[str, Any]]:
        """Read one latest record per reserved full schema name.

        ``config_key_suffix`` narrows the store read to the keys ending with
        it; the records it returns are validated exactly as an unnarrowed read
        validates the same rows.
        """
        try:
            entries = self._store.list_all_configs(
                scope=ConfigScope.SCHEMA,
                service=_SERVICE,
                config_key_suffix=config_key_suffix,
            )
            records = []
            for entry in entries:
                record = entry.config_value
                row = record["registration"]
                self._validate(row)
                if (
                    entry.tenant_id != SYSTEM_TENANT_ID
                    or entry.config_key != row["full_schema_name"]
                ):
                    raise ValueError("intent storage key and registration disagree")
                records.append({**record, "_revision": entry.version})
            return records
        except Exception as exc:
            raise RegistryStorageError(
                f"Cannot read deployment intents: {exc}"
            ) from exc

    def pending(self) -> list[dict[str, Any]]:
        """Return active intents; absent, complete and failed records are retired."""
        return [record for record in self.records() if record["state"] == "pending"]

    def pending_for_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
        """This tenant's active intents, read without carrying the others.

        Intents are keyed under the system tenant by the full schema name, and
        :meth:`_validate` holds that name to ``{base}_{sanitised tenant}`` — so
        every row this tenant owns, and no row of a tenant whose id is not a
        suffix of it, ends with that tenant's suffix. The store narrows the
        visit to those keys; which tenant a record belongs to is still decided
        by the registration's own ``tenant_id``, so a suffix that another
        tenant's id extends cannot be read as this tenant's.
        """
        tenant_id = canonical_tenant_id(tenant_id)
        suffix = f"_{tenant_id.replace(':', '_')}"
        return [
            record
            for record in self.records(config_key_suffix=suffix)
            if record["state"] == "pending"
            and record["registration"]["tenant_id"] == tenant_id
        ]

    def reserved(self, live_names: set[str]) -> dict[str, dict[str, Any]]:
        """Full schema names an in-flight activation owns, with their exact registration.

        A pending intent whose schema is live in Vespa is a registration in
        progress or awaiting recovery, never an orphan. A pending intent whose
        schema is not live yet counts only inside its grace: its activation is
        imminent, and a package built without it drops the schema the moment
        that activation lands. A pending intent past its grace with no live
        schema is an abandoned activation that recovery retires.
        """
        now = _now()
        return {
            record["registration"]["full_schema_name"]: record["registration"]
            for record in self.pending()
            if record["registration"]["full_schema_name"] in live_names
            or now < record["recover_after"]
        }

    def _transition(self, record: dict[str, Any], state: str) -> dict[str, Any] | None:
        current = record
        for _ in range(_MAX_ATTEMPTS):
            if current["state"] == "complete":
                return current
            updated = self._save({**current, "state": state})
            if updated:
                return updated
            current = self._read(record["registration"]["full_schema_name"])
            if current is None or (
                current["registration"],
                current["registry_version"],
            ) != (record["registration"], record["registry_version"]):
                return None
        raise RegistryStorageError(
            f"Cannot transition deployment intent for {record['registration']['full_schema_name']!r}: concurrent writers"
        )

    def retire(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Clear an unsuccessful activation's marker, retaining late-write recovery."""
        return self._transition(record, "absent")

    def complete(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Clear the active marker without overwriting a newer generation."""
        return self._transition(record, "complete")

    def reconcile(
        self,
        live_names: set[str],
        registered: dict[str, dict[str, Any]],
        write_registration: Callable[[dict[str, Any], int], None],
        fence: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Conditionally claim one of three attempts and complete only live schemas.

        The callback receives the exact payload and pre-activation registry
        version. A tombstone or newer registration must reject that version.
        ``fence`` runs before every journal and registry write and raises to
        stop a caller that no longer holds the deployment lease.
        """
        check = fence if fence is not None else (lambda: None)
        recovered = []
        for record in self.records():
            row = record["registration"]
            name = row["full_schema_name"]
            if record["state"] == "complete" or _now() < record["recover_after"]:
                continue
            current = registered.get(name)
            if current:
                if current != row:
                    raise RegistryStorageError(
                        f"Intent for {name!r} conflicts with registered ownership or payload"
                    )
                check()
                self.complete(record)
                continue
            if name not in live_names:
                if record["state"] == "pending":
                    check()
                    self.retire(record)
                continue
            if record["attempts"] == _MAX_ATTEMPTS:
                raise RegistryStorageError(
                    f"Recovery of {name!r} exhausted {_MAX_ATTEMPTS} attempts"
                )
            check()
            attempt = self._save({**record, "attempts": record["attempts"] + 1})
            if attempt is None:
                continue

            def failed(exc: Exception) -> RegistryStorageError:
                if attempt["attempts"] == _MAX_ATTEMPTS:
                    check()
                    self._transition({**attempt, "last_error": str(exc)}, "failed")
                return RegistryStorageError(
                    f"Recovery of {name!r} failed on attempt {attempt['attempts']}/{_MAX_ATTEMPTS}: {exc}"
                )

            try:
                check()
            except Exception as refused:
                # The claim was never used: hand the attempt back so repeated
                # takeovers cannot exhaust recovery. Conditional on the claim's
                # own revision, so a successor's write since is never undone.
                try:
                    self._save({**attempt, "attempts": record["attempts"]})
                except RegistryStorageError as release_exc:
                    refused.add_note(
                        f"The recovery attempt was not released: {release_exc}"
                    )
                raise
            try:
                write_registration(copy.deepcopy(row), record["registry_version"])
            except Exception as exc:
                raise failed(exc) from exc
            check()
            try:
                self.complete(attempt)
            except Exception as exc:
                raise failed(exc) from exc
            recovered.append(row)
        return recovered
