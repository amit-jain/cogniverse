"""Configuration read, edit, history, rollback, export and import.

The editable configs are the sections of ``cogniverse_foundation.config.
sections``: each is served with its dataclass's JSON schema, so a form built
from it shows every field. Writes are compare-and-set on the version the
editor read: a save made over a version someone else replaced is refused, not
applied over their change. Secrets are never sent back; a form leaves them
null to keep them.

Configs without a section (backend profiles, tenant instructions) can be
listed, browsed through their history and rolled back, but not edited here.

A save, restore or import answers once every runtime worker and ingestion
worker has dropped what it held for the tenant (``configs_changed`` on the
config events channel), so the next request anywhere reads the write.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.sections import (
    CONFIG_SECTIONS,
    SYSTEM_CONFIG_TENANT,
    ConfigSection,
    ConfigValueError,
    section_for,
)
from cogniverse_runtime.http_errors import canonical_tenant_or_400, failure_response
from cogniverse_runtime.routers.admin import get_config_manager_dependency
from cogniverse_sdk.interfaces.config_store import (
    ConfigEntry,
    ConfigScope,
    ConfigStoreUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# The most versions one history read returns.
HISTORY_LIMIT = 100

# How long a config write waits for every worker to drop what it held.
CONFIG_CHANGE_ACK_TIMEOUT_S = 15.0

# Delivers config changes to every runtime and ingestion worker; wired at
# startup.
_config_events = None


def set_config_events(config_events) -> None:
    """Wire the channel config changes reach every worker process through."""
    global _config_events
    _config_events = config_events


def _require_config_change_channel() -> None:
    """A config write refuses before storing anything when no channel can
    carry the change to the other workers."""
    if _config_events is None:
        raise RuntimeError("Config writes need the config events channel wired")


async def _publish_config_change(stored_tenant: str, what: str) -> None:
    """Have every runtime and ingestion worker drop what it holds for the
    tenant, so its next read on any of them is the write."""
    from cogniverse_runtime.cluster_events import CONFIGS_CHANGED, ClusterEventError

    try:
        await _config_events.publish(
            CONFIGS_CHANGED,
            {"tenant_id": stored_tenant},
            timeout_s=CONFIG_CHANGE_ACK_TIMEOUT_S,
        )
    except ClusterEventError as exc:
        raise failure_response(
            503,
            "config_change_not_propagated",
            f"{what} is stored for {stored_tenant}, but not every worker "
            "dropped the configs it held; those workers read it within a "
            "minute.",
            exc,
            tenant_id=stored_tenant,
        ) from exc


class SectionInfo(BaseModel):
    name: str
    title: str
    tenant_scoped: bool
    service: Optional[str] = Field(
        ..., description="The fixed service, or null when each agent is one"
    )
    schema_: Dict[str, Any] = Field(..., alias="schema")

    model_config = {"populate_by_name": True}


class SectionList(BaseModel):
    sections: List[SectionInfo]


class EntrySummary(BaseModel):
    scope: str
    service: str
    config_key: str
    version: int
    updated_at: str
    section: Optional[str] = Field(
        ..., description="The section that edits this entry, if any"
    )


class EntryList(BaseModel):
    tenant_id: str
    entries: List[EntrySummary]


class SectionValue(BaseModel):
    section: str
    tenant_id: str
    service: str
    version: int = Field(..., description="Stored version; 0 when none is stored")
    updated_at: Optional[str]
    value: Dict[str, Any] = Field(
        ..., description="The config as the form edits it; secrets are null"
    )
    secrets: Dict[str, bool] = Field(..., description="Which secrets hold a value")


class SectionWrite(BaseModel):
    tenant_id: Optional[str] = Field(
        None, description="Required for tenant sections; absent for system ones"
    )
    service: Optional[str] = Field(None, description="The agent, for the agent section")
    value: Dict[str, Any] = Field(
        ...,
        description=(
            "Fields to set; a field left out keeps its stored value, a secret "
            "left null keeps its value and an empty string clears it"
        ),
    )
    version: int = Field(..., ge=0, description="The version the editor read")


class HistoryVersion(BaseModel):
    version: int
    created_at: str
    updated_at: str
    value: Dict[str, Any]


class History(BaseModel):
    tenant_id: str
    scope: str
    service: str
    config_key: str
    section: Optional[str]
    versions: List[HistoryVersion]


class Rollback(BaseModel):
    tenant_id: Optional[str] = None
    scope: ConfigScope
    service: str
    config_key: str
    version: int = Field(..., ge=1, description="The version to restore")
    expected_version: int = Field(
        ..., ge=1, description="The latest version the editor read"
    )


class Written(BaseModel):
    version: int
    updated_at: str


class ImportRequest(BaseModel):
    tenant_id: str
    configs: Dict[str, Any] = Field(..., description="A configuration export")


class Imported(BaseModel):
    tenant_id: str
    imported: int


def _tenant(tenant_id: Optional[str], tenant_scoped: bool) -> str:
    """The stored tenant id: the canonical tenant, or the system tenant."""
    if not tenant_scoped:
        if tenant_id:
            raise HTTPException(
                status_code=400, detail="System configs take no tenant_id"
            )
        return SYSTEM_CONFIG_TENANT
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    return canonical_tenant_id(tenant_id)


def _entry_tenant(tenant_id: Optional[str]) -> str:
    return canonical_tenant_id(tenant_id) if tenant_id else SYSTEM_CONFIG_TENANT


def _section(name: str) -> ConfigSection:
    section = CONFIG_SECTIONS.get(name)
    if section is None:
        raise HTTPException(
            status_code=404,
            detail=f"No config section '{name}'; sections: {sorted(CONFIG_SECTIONS)}",
        )
    return section


def _service(section: ConfigSection, service: Optional[str]) -> str:
    try:
        return section.entry_service(service)
    except ConfigValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _read_failure(exc: Exception, what: str, **fields: Any) -> HTTPException:
    if isinstance(exc, ConfigStoreUnavailableError):
        return failure_response(
            503,
            "config_store_unavailable",
            f"The config store did not answer while {what}; retry.",
            exc,
            **fields,
        )
    return failure_response(
        500,
        "config_read_failed",
        f"Reading the config store failed while {what}; the runtime log "
        "names the cause.",
        exc,
        **fields,
    )


def _iso(value: datetime) -> str:
    return value.isoformat()


def _shown_value(section: Optional[ConfigSection], stored: Dict[str, Any]):
    """A stored value as it may be shown: through its section's form, so
    secrets are withheld; values of no section as stored."""
    if section is None:
        return stored
    return section.form_value(section.load(stored))


@router.get(
    "/config/sections", response_model=SectionList, response_model_by_alias=True
)
async def list_sections() -> SectionList:
    """The editable config sections, each with its form's JSON schema."""
    return SectionList(
        sections=[
            SectionInfo(
                name=section.name,
                title=section.title,
                tenant_scoped=section.tenant_scoped,
                service=section.service,
                schema=section.schema(),
            )
            for section in CONFIG_SECTIONS.values()
        ]
    )


@router.get("/config/entries", response_model=EntryList)
async def list_entries(
    tenant_id: Optional[str] = Query(None, description="Absent: system configs"),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> EntryList:
    """A tenant's stored configs (or the system ones), latest versions."""
    stored_tenant = _entry_tenant(tenant_id)
    try:
        entries = await asyncio.to_thread(
            config_manager.store.list_configs, tenant_id=stored_tenant
        )
    except Exception as exc:
        raise _read_failure(exc, "listing configs", tenant_id=stored_tenant)
    rows = []
    for entry in entries:
        if entry.scope == ConfigScope.SCHEMA:
            continue
        section = section_for(entry.scope, entry.config_key)
        rows.append(
            EntrySummary(
                scope=entry.scope.value,
                service=entry.service,
                config_key=entry.config_key,
                version=entry.version,
                updated_at=_iso(entry.updated_at),
                section=section.name if section else None,
            )
        )
    rows.sort(key=lambda row: (row.scope, row.service, row.config_key))
    return EntryList(tenant_id=stored_tenant, entries=rows)


@router.get("/config/sections/{section_name}", response_model=SectionValue)
async def read_section(
    section_name: str,
    tenant_id: Optional[str] = Query(None),
    service: Optional[str] = Query(None),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> SectionValue:
    """A section's stored config as its form edits it, with its version; the
    section's defaults at version 0 when nothing is stored."""
    section = _section(section_name)
    stored_tenant = _tenant(tenant_id, section.tenant_scoped)
    entry_service = _service(section, service)
    try:
        entry = await asyncio.to_thread(
            config_manager.store.get_config,
            stored_tenant,
            section.scope,
            entry_service,
            section.config_key,
        )
    except Exception as exc:
        raise _read_failure(
            exc,
            f"reading the {section.name} config",
            tenant_id=stored_tenant,
            service=entry_service,
        )
    config = (
        section.load(entry.config_value)
        if entry is not None
        else section.default(stored_tenant, entry_service)
    )
    return SectionValue(
        section=section.name,
        tenant_id=stored_tenant,
        service=entry_service,
        version=entry.version if entry is not None else 0,
        updated_at=_iso(entry.updated_at) if entry is not None else None,
        value=section.form_value(config),
        secrets=section.secrets_set(config),
    )


@router.put("/config/sections/{section_name}", response_model=SectionValue)
async def write_section(
    section_name: str,
    request: SectionWrite,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> SectionValue:
    """Apply a form's fields to the version the editor read.

    Raises:
        HTTPException 409: Another write replaced the version the editor read
        HTTPException 422: A field the section does not have, or a value its
            dataclass refuses
        HTTPException 503: The config store did not answer
    """
    _require_config_change_channel()
    section = _section(section_name)
    stored_tenant = _tenant(request.tenant_id, section.tenant_scoped)
    entry_service = _service(section, request.service)

    def _write() -> tuple[Any, ConfigEntry]:
        current_entry = config_manager.store.get_config(
            stored_tenant, section.scope, entry_service, section.config_key
        )
        current_version = current_entry.version if current_entry else 0
        if current_version != request.version:
            raise _conflict(request.version, current_version)
        current = (
            section.load(current_entry.config_value)
            if current_entry is not None
            else section.default(stored_tenant, entry_service)
        )
        updated = section.from_form(request.value, current)
        written = config_manager.compare_and_set_entry(
            stored_tenant,
            section.scope,
            entry_service,
            section.config_key,
            section.dump(updated, stored_tenant),
            expected_version=request.version,
        )
        if written is None:
            latest = config_manager.store.get_config(
                stored_tenant, section.scope, entry_service, section.config_key
            )
            raise _conflict(request.version, latest.version if latest else 0)
        return updated, written

    try:
        updated, written = await asyncio.to_thread(_write)
    except HTTPException:
        raise
    except ConfigValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "config_value_invalid",
                "message": f"The {section.name} config was not saved.",
                "errors": exc.errors,
            },
        ) from exc
    except Exception as exc:
        raise _read_failure(
            exc,
            f"saving the {section.name} config",
            tenant_id=stored_tenant,
            service=entry_service,
        )
    logger.info(
        "Saved %s config v%s for %s:%s",
        section.name,
        written.version,
        stored_tenant,
        entry_service,
    )
    await _publish_config_change(
        stored_tenant, f"The {section.name} config version {written.version}"
    )
    return SectionValue(
        section=section.name,
        tenant_id=stored_tenant,
        service=entry_service,
        version=written.version,
        updated_at=_iso(written.updated_at),
        value=section.form_value(updated),
        secrets=section.secrets_set(updated),
    )


def _conflict(read_version: int, current_version: int) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": "config_version_conflict",
            "message": (
                f"The config changed since it was read (version {read_version}, "
                f"now {current_version}); reload it and apply your edits again."
            ),
            "current_version": current_version,
        },
    )


@router.get("/config/history", response_model=History)
async def read_history(
    scope: ConfigScope,
    service: str,
    config_key: str,
    tenant_id: Optional[str] = Query(None, description="Absent: system configs"),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> History:
    """A config's versions, newest first, at most ``HISTORY_LIMIT``."""
    stored_tenant = _entry_tenant(tenant_id)
    try:
        entries = await asyncio.to_thread(
            config_manager.store.get_config_history,
            stored_tenant,
            scope,
            service,
            config_key,
            HISTORY_LIMIT,
        )
    except Exception as exc:
        raise _read_failure(
            exc, "reading config history", tenant_id=stored_tenant, service=service
        )
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No {scope.value} config {service}/{config_key} for {stored_tenant}",
        )
    section = section_for(scope, config_key)
    return History(
        tenant_id=stored_tenant,
        scope=scope.value,
        service=service,
        config_key=config_key,
        section=section.name if section else None,
        versions=[
            HistoryVersion(
                version=entry.version,
                created_at=_iso(entry.created_at),
                updated_at=_iso(entry.updated_at),
                value=_shown_value(section, entry.config_value),
            )
            for entry in entries
        ],
    )


@router.post("/config/rollback", response_model=Written)
async def rollback(
    request: Rollback,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Written:
    """Store an earlier version's value as the next version.

    Raises:
        HTTPException 404: The version is not in the config's history
        HTTPException 409: The latest version is not the one the editor read
        HTTPException 503: The restore is stored, but a worker did not confirm
            dropping what it held
    """
    _require_config_change_channel()
    stored_tenant = _entry_tenant(request.tenant_id)
    if request.version >= request.expected_version:
        raise HTTPException(
            status_code=400,
            detail="Only a version older than the latest can be restored",
        )

    def _restore() -> ConfigEntry:
        history = config_manager.store.get_config_history(
            stored_tenant,
            request.scope,
            request.service,
            request.config_key,
            request.expected_version - request.version + 1,
        )
        latest = history[0].version if history else 0
        if latest != request.expected_version:
            raise _conflict(request.expected_version, latest)
        restored = next((e for e in history if e.version == request.version), None)
        if restored is None:
            raise HTTPException(
                status_code=404,
                detail=f"Version {request.version} is no longer kept",
            )
        written = config_manager.compare_and_set_entry(
            stored_tenant,
            request.scope,
            request.service,
            request.config_key,
            restored.config_value,
            expected_version=request.expected_version,
        )
        if written is None:
            current = config_manager.store.get_config(
                stored_tenant, request.scope, request.service, request.config_key
            )
            raise _conflict(request.expected_version, current.version if current else 0)
        return written

    try:
        written = await asyncio.to_thread(_restore)
    except HTTPException:
        raise
    except Exception as exc:
        raise _read_failure(
            exc,
            "restoring a config version",
            tenant_id=stored_tenant,
            service=request.service,
        )
    logger.info(
        "Restored %s/%s/%s v%s as v%s for %s",
        request.scope.value,
        request.service,
        request.config_key,
        request.version,
        written.version,
        stored_tenant,
    )
    await _publish_config_change(
        stored_tenant,
        f"Version {request.version} of {request.scope.value}/{request.service}/"
        f"{request.config_key}, restored as version {written.version},",
    )
    return Written(version=written.version, updated_at=_iso(written.updated_at))


@router.get("/config/export")
async def export_configs(
    tenant_id: str,
    include_history: bool = False,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, Any]:
    """A tenant's configs as the store exports them, secrets included: the
    export is a backup that ``/config/import`` restores whole."""
    stored_tenant = canonical_tenant_or_400(tenant_id)
    try:
        return await asyncio.to_thread(
            config_manager.store.export_configs, stored_tenant, include_history
        )
    except Exception as exc:
        raise _read_failure(exc, "exporting configs", tenant_id=stored_tenant)


@router.post("/config/import", response_model=Imported)
async def import_configs(
    request: ImportRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Imported:
    """Write an export's configs into a tenant; tenant ids inside the export
    are ignored. The import lands whole or not at all.

    Raises:
        HTTPException 400: The export holds rows only the schema registry writes
        HTTPException 503: The import is stored, but a worker did not confirm
            dropping what it held
    """
    _require_config_change_channel()
    stored_tenant = canonical_tenant_or_400(request.tenant_id)

    def _import() -> int:
        try:
            return config_manager.store.import_configs(stored_tenant, request.configs)
        finally:
            config_manager.forget_held_configs(stored_tenant)

    try:
        count = await asyncio.to_thread(_import)
    except ValueError as exc:
        raise failure_response(
            400,
            "config_import_refused",
            "The export cannot be imported; the runtime log names the row.",
            exc,
            tenant_id=stored_tenant,
        )
    except Exception as exc:
        raise failure_response(
            503 if isinstance(exc, ConfigStoreUnavailableError) else 500,
            "config_import_failed",
            "The import failed and nothing was kept; the runtime log names the cause.",
            exc,
            tenant_id=stored_tenant,
        )
    logger.info("Imported %s configs into %s", count, stored_tenant)
    await _publish_config_change(stored_tenant, f"The import of {count} configs")
    return Imported(tenant_id=stored_tenant, imported=count)


@router.get("/config/stats")
async def store_stats(
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, Any]:
    """The config store's counts."""
    try:
        return await asyncio.to_thread(config_manager.store.get_stats)
    except Exception as exc:
        raise _read_failure(exc, "reading store statistics")


class StoreHealth(BaseModel):
    store: str = Field(..., description="The config store's implementation")
    healthy: bool = Field(..., description="Whether the store answered a query")


@router.get("/config/health", response_model=StoreHealth)
async def store_health(
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> StoreHealth:
    """Whether the config store answers a query now."""
    store = config_manager.store
    return StoreHealth(
        store=type(store).__name__,
        healthy=await asyncio.to_thread(store.health_check),
    )
