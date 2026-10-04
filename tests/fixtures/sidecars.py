"""Sidecar fixtures and the session's sidecar report, shared by every subsuite.

``tests/conftest.py`` registers this plugin through ``pytest_plugins``.
``tests/ingestion/pytest.ini`` makes ``tests/ingestion`` the rootdir, which
leaves ``tests/conftest.py`` out of such a session, so
``tests/ingestion/conftest.py`` registers it from ``pytest_configure``.
Either way it is registered once and its hooks run once per session:

- ``pytest_sessionstart`` removes containers whose owning pytest process died
  without teardown, before collection, so a session that never provisions a
  sidecar still clears what a killed session left holding host RAM.
- ``pytest_terminal_summary`` prints that reap and every model endpoint
  decision the session made (``tests/utils/model_resolution.py``), whatever
  the capture mode and whether tests passed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_REAP_REPORT = pytest.StashKey[str | None]()


def reap_at_session_start() -> str | None:
    """Reap dead-owner containers and return the summary line, if there is one.

    Without a docker CLI no test container can exist, so there is nothing to
    reap. A docker that cannot list or remove is reported, not raised: the
    session's own tests decide whether they need docker.
    """
    from tests.utils.vllm_sidecar import reap_dead_owner_containers

    try:
        removed = reap_dead_owner_containers()
    except FileNotFoundError:
        return None
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return f"dead-owner container reap failed: {exc}"
    if not removed:
        return None
    return "reaped dead-owner containers: " + ", ".join(removed)


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config.stash[_REAP_REPORT] = reap_at_session_start()


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    lines = []
    reap = config.stash.get(_REAP_REPORT, None)
    if reap is not None:
        lines.append(reap)
    # A process that never imported the resolvers made no model decision.
    resolutions = sys.modules.get("tests.utils.model_resolution")
    if resolutions is not None:
        for resolution, calls in resolutions.resolution_counts():
            line = resolution.summary_line()
            lines.append(line if calls == 1 else f"{line} x{calls}")
    if not lines:
        return
    terminalreporter.section("test sidecars")
    for line in lines:
        terminalreporter.write_line(line)


@pytest.fixture(scope="session")
def vllm_sidecar():
    """Factory for spinning up real vLLM sidecars on demand. See
    tests/utils/vllm_sidecar.py for usage details."""
    from tests.utils.vllm_sidecar import VllmSidecarFactory

    factory = VllmSidecarFactory()
    try:
        yield factory
    finally:
        factory.teardown()


@pytest.fixture(scope="session")
def pylate_server(remote_inference):
    """LateOn served by the cluster's PyLate service (deploy/pylate) exposing
    the production ``/pooling`` contract.

    The service owns PyLate's exact encode for both directions: query
    expansion over masked padding positions and the document punctuation
    skiplist. Generic vLLM ``/pooling`` cannot reproduce the query side
    because its request schema carries no attention mask.
    """
    return remote_inference.resolve("colbert_pylate").base_url


@pytest.fixture(scope="module")
def served_code_colbert(remote_inference):
    """The pinned code encoder served by the cluster's PyLate service."""
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    endpoint = remote_inference.resolve("code_colbert_pylate")
    loader = RemoteColBERTLoader(
        model_name=endpoint.model_id,
        config={"remote_inference_url": endpoint.base_url},
        _resolved_headers=dict(endpoint.headers),
    )
    model, _ = loader.load_model()
    try:
        yield model
    finally:
        model._close()
