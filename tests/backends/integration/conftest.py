"""Integration test configuration for backend tests.

The fixture is now a thin re-export of the project-wide ``shared_vespa``
container (``tests/conftest.py``). Schema-lifecycle tests in this package
already deploy schemas under unique tenant_ids (``acme``, ``startup``,
etc.) so they coexist with other packages' tenant-scoped schemas in the
same Vespa without conflict — Vespa is multi-tenant by design.

The schema-mutation tests' ``wipe_non_protected_schemas`` fixture (in
``test_tenant_schema_lifecycle.py``) is the one place that needs care
under sharing — it must NOT touch tenants owned by other packages.
That scoping is done in the test file itself, not here.
"""

import logging

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape backends/integration tests
    expect (``http_port``, ``config_port``, ``base_url``, ``container_name``)
    backed by the project-wide ``shared_vespa`` container.

    The original fixture started its own VespaDockerManager-managed
    container per module + deployed metadata schemas. ``shared_vespa``
    deploys the same metadata schemas at session start, so this shim
    just forwards the connection info — no setup work needed.
    """
    yield {
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
        "base_url": shared_vespa["base_url"],
        "container_name": shared_vespa["container_name"],
    }
    # Singleton clearing happens via the per-test autouse fixtures in
    # consumer conftests (e.g. tests/agents/integration/conftest.py).
    # No teardown here — shared_vespa owns the container lifecycle.
