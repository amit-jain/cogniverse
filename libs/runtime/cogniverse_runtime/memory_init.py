"""Lazy Mem0 initialisation shared between the runtime FastAPI app and
out-of-process CLI entrypoints.

Both the runtime routers (e.g. ``POST /admin/tenant/{t}/memories``) and
the optimization workflow CLI (``optimization_cli --mode cleanup``)
need to bring up a ``Mem0MemoryManager`` for a tenant from the same
system-config-derived endpoints. Each callsite previously inlined the
setup, which split the truth across modules and caused the daily-cleanup
workflow to silently report ``failed: Mem0MemoryManager not initialized``
for every tenant — the CLI path skipped initialize() while the router
path used a local helper.

This module centralises the pattern. Callers pass a constructed
manager plus a config_manager; the helper enriches with environment
variables (``LLM_ENDPOINT``, ``VESPA_CONFIG_PORT``) and calls
``mgr.initialize`` with the right argument shape. On failure the
helper logs and raises with the concrete cause so callers can surface
it directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config

logger = logging.getLogger(__name__)


def lazy_init_memory(
    mgr: Mem0MemoryManager,
    tenant_id: str,
    config_manager: ConfigManager,
    auto_create_schema: bool = True,
) -> bool:
    """Initialise ``mgr`` for ``tenant_id`` from the system config.

    Returns True if the manager ended up with ``mgr.memory`` populated
    (already-initialised counts as True). Raises RuntimeError with the
    underlying cause if initialisation fails for any reason (missing
    system config, missing llm model id, missing denseon URL,
    Mem0/Vespa init failure).

    ``auto_create_schema`` MUST be False on read-only paths. Schema
    creation triggers a Vespa global app-redeploy; doing that while
    serving a read reconfigures the content cluster and can drop
    documents another process just fed but Vespa hasn't flushed yet —
    the reader then sees its target rows vanish. A reader connects to
    the already-existing tenant schema and returns empty when absent.
    """
    if mgr.memory:
        return True
    sc = config_manager.get_system_config()
    if not sc:
        msg = (
            f"Failed to lazy-init Mem0 for tenant {tenant_id}: "
            "no system config available"
        )
        logger.warning(msg)
        raise RuntimeError(msg)

    try:
        config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
        llm_cfg = (config.get("llm_config") or {}).get("primary") or {}
        model = llm_cfg.get("model")
        if not model:
            raise RuntimeError(
                "llm_config.primary.model missing — Mem0 lazy-init requires "
                "an explicit model id. Check configs/config.json or the chart "
                "values that populate it."
            )
        if "/" in model:
            model = model.split("/", 1)[1]

        llm_base_url = os.environ.get(
            "LLM_ENDPOINT",
            llm_cfg.get("api_base") or "http://localhost:11434",
        )
        denseon_url = sc.inference_service_urls.get("denseon")
        if not denseon_url:
            raise RuntimeError(
                "Mem0 lazy-init requires the 'denseon' inference service "
                f"to be present in system_config.inference_service_urls. "
                f"Available: {sorted(sc.inference_service_urls)}"
            )

        init_kwargs: dict[str, Any] = dict(
            backend_host=sc.backend_url,
            backend_port=sc.backend_port,
            backend_config_port=int(os.environ.get("VESPA_CONFIG_PORT", "19071")),
            base_schema_name="agent_memories",
            llm_model=model,
            embedding_model="lightonai/DenseOn",
            llm_base_url=llm_base_url,
            embedder_base_url=denseon_url,
            auto_create_schema=auto_create_schema,
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )
        mgr.initialize(**init_kwargs)
    except Exception as exc:
        logger.warning("Failed to lazy-init Mem0 for tenant %s: %s", tenant_id, exc)
        raise RuntimeError(
            f"Failed to lazy-init Mem0 for tenant {tenant_id}: {exc}"
        ) from exc

    if not mgr.memory:
        msg = (
            f"Failed to lazy-init Mem0 for tenant {tenant_id}: "
            "initialize() returned without a memory backend"
        )
        logger.warning(msg)
        raise RuntimeError(msg)

    logger.info("Lazy-initialised Mem0 for tenant %s", tenant_id)
    return True
