"""Bridge the live e2e Vespa endpoint into the shared backend-config env.

``tests/conftest.py::backend_config_env`` defaults ``BACKEND_PORT`` to a dead
sentinel (29071) so a unit test that resolves config without binding a real
store fails loudly instead of silently hitting an ambient Vespa. E2E runs
against a live cluster, so they must supply the real endpoint or every
in-process config read fails with ConnectionRefused on that sentinel.

An in-process ``ConfigManager`` in an e2e test also needs the host-side ports
in its system config where the cluster's stored one names in-cluster services;
``LocalSystemConfig`` answers the system config from the test process and every
other config from the cluster's store.

Kept out of ``conftest.py`` so it can be imported and tested without loading the
session fixtures, which provision a cluster.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope

DEFAULT_VESPA_URL = "http://localhost:33080"


def vespa_url() -> str:
    """The e2e cluster's Vespa endpoint."""
    return os.environ.get("VESPA_URL", DEFAULT_VESPA_URL)


def backend_env_from_vespa_url(url: str) -> tuple[str, str]:
    """Split a Vespa URL into the (BACKEND_URL, BACKEND_PORT) pair.

    The port is explicit rather than scheme-defaulted because the shared fixture
    compares it against the sentinel, and an empty value there would read as
    "unset" and fall back to the dead port.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"VESPA_URL must be an http(s) URL with a host: {url!r}")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme}://{parsed.hostname}", str(port)


def export_backend_env() -> tuple[str, str]:
    """Publish the live endpoint as TEST_BACKEND_URL/TEST_BACKEND_PORT.

    Never overrides values already set, so an explicit override still wins.
    """
    backend_url, backend_port = backend_env_from_vespa_url(vespa_url())
    os.environ.setdefault("TEST_BACKEND_URL", backend_url)
    os.environ.setdefault("TEST_BACKEND_PORT", backend_port)
    return backend_url, backend_port


class LocalSystemConfig:
    """A config store whose system config is this process's, never the store's.

    Storing a host-side system config would point the cluster's own pods at
    localhost ports, so the e2e manager answers it here, on every read
    including its background refreshes, and reads everything else from
    ``store``.
    """

    def __init__(self, store: Any, system_config: SystemConfig) -> None:
        self._store = store
        stamp = datetime.now(timezone.utc)
        self._entry = ConfigEntry(
            tenant_id=ConfigManager._SYSTEM_TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            config_value=system_config.to_dict(redact=False),
            version=1,
            created_at=stamp,
            updated_at=stamp,
        )

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        if (tenant_id, scope, service, config_key) == (
            self._entry.tenant_id,
            self._entry.scope,
            self._entry.service,
            self._entry.config_key,
        ):
            return self._entry
        return self._store.get_config(tenant_id, scope, service, config_key, version)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)
