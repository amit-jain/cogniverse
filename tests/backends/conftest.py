"""Bind every test under ``tests/backends`` to the shared test Vespa with
config seeded.

The project-wide ``backend_config_env`` (``tests/conftest.py``) points the test
backend at a deliberate dead port so no test can silently resolve config
against an ambient Vespa (e.g. a developer's k3d on ``:8080``). Backend tests
DO exercise the real ``VespaConfigStore`` against real, present config, so this
override binds them to ``seeded_config_vespa`` — the shared test Vespa on a
non-obvious port with metadata schemas + baseline system/telemetry config
seeded. Real boundary, no mocks, identical local and CI.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def backend_config_env(seeded_config_vespa):  # noqa: F811
    yield
