"""Bridge the live e2e Vespa endpoint into the shared backend-config env.

``tests/conftest.py::backend_config_env`` defaults ``BACKEND_PORT`` to a dead
sentinel (29071) so a unit test that resolves config without binding a real
store fails loudly instead of silently hitting an ambient Vespa. E2E runs
against a live cluster, so they must supply the real endpoint or every
in-process config read fails with ConnectionRefused on that sentinel.

Kept out of ``conftest.py`` so it can be imported and tested without loading the
session fixtures, which provision a cluster.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

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
