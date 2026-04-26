"""Auto-skip integration tests when their managed services aren't available."""

from __future__ import annotations

import pytest

from tests.utils.markers import is_docker_available


def pytest_collection_modifyitems(items):
    docker_ok = is_docker_available()
    for item in items:
        if "requires_docker" in item.keywords and not docker_ok:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available in this environment")
            )
