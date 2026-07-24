"""Location-derived pytest markers.

Tests under a ``unit/`` (``integration/``) directory carry the ``unit``
(``integration``) marker from their location — the directory IS the taxonomy,
so a file or test that forgets the marker cannot silently fall out of its
directory's CI ``-m`` selection. ``local_only`` opts out: it declares a
deliberate exclusion from CI selections, so no location marker is added.

Called from ``pytest_collection_modifyitems`` in every rootdir conftest —
``tests/conftest.py`` plus the nested roots (``tests/ingestion``,
``tests/routing``) whose own ``pytest.ini`` puts the project conftest outside
the discovery boundary. ``tests/runtime/unit/test_marker_coverage.py``
mirrors this rule when verifying CI selections.
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
