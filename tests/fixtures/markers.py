"""Location-derived pytest markers and test-LM marker plumbing.

Tests under a ``unit/`` (``integration/``, ``e2e/``) directory carry the
``unit`` (``integration``, ``e2e``) marker from their location — the directory
IS the taxonomy, so a file or test that forgets the marker cannot silently fall
out of its directory's CI ``-m`` selection, nor into a selection it does not
belong to: an e2e file that marks itself ``integration`` gets collected by every
non-e2e sweep and errors there on the cluster it needs. ``local_only`` opts out:
it declares a deliberate exclusion from CI selections, so no location marker is
added.

Called from ``pytest_collection_modifyitems`` in every rootdir conftest —
``tests/conftest.py`` plus the nested roots (``tests/ingestion``,
``tests/routing``) whose own ``pytest.ini`` puts the project conftest outside
the discovery boundary. ``tests/runtime/unit/test_marker_coverage.py``
mirrors this rule when verifying CI selections.

``enforce_lm_gate`` is the shared half of the
``requires_lm`` convention: collection stamps the LM roles a test needs and
injects the session provisioner; the post-setup gate fails (never skips) a
marked test whose provisioned endpoint stopped answering. Both are idempotent
so a nested rootdir conftest and ``tests/conftest.py`` may each apply them.
"""

import pytest


def apply_location_markers(items) -> None:
    # get_closest_marker, not ``in item.keywords`` — keywords also contains
    # ancestor node NAMES, and the directory itself is named "unit".
    for item in items:
        if item.get_closest_marker("local_only") is not None:
            continue
        path = item.path.as_posix()
        if "/unit/" in path and item.get_closest_marker("unit") is None:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path and item.get_closest_marker("integration") is None:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in path and item.get_closest_marker("e2e") is None:
            item.add_marker(pytest.mark.e2e)


def enforce_lm_gate(item) -> None:
    """Fail (never skip) a ``requires_lm`` test whose endpoint is unreachable.

    Runs after fixture setup (register with ``trylast=True``) so the session
    provisioner has already exported the endpoint this probes. ``--setup-plan``
    prints the fixture plan without executing a single fixture, so no
    provisioner has run and there is no provisioned endpoint to gate.

    The failure names the URLs actually requested and where the endpoint came
    from, so an unprovisioned session is distinguishable from a dead server.
    """
    if item.get_closest_marker("requires_lm") is None:
        return
    if item.config.getoption("setupplan", default=False):
        return

    from tests.fixtures.llm import (
        is_test_lm_available,
        lm_endpoint_source,
        lm_probe_targets,
    )

    if not is_test_lm_available():
        pytest.fail(
            "Exact configured LLM endpoint not reachable. Probed "
            f"{', '.join(lm_probe_targets())} "
            f"(endpoint from {lm_endpoint_source()})",
            pytrace=False,
        )
