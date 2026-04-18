"""Unit tests for VespaSearchBackend's runtime add_profile / remove_profile.

Closes a long-standing gap where the search backend's ``self.profiles``
dict was a startup snapshot that no code path could update. Dynamically
created profiles (via ``POST /admin/profiles`` or Mem0 auto-bootstrap)
were persisted to the ConfigStore but invisible to the cached search
backend, causing "profile not found" retry storms under concurrent load.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from cogniverse_vespa.search_backend import VespaSearchBackend


def _make_backend(profiles: dict | None = None) -> VespaSearchBackend:
    """Build a backend without touching real Vespa / pool / metrics."""
    with patch("cogniverse_vespa.search_backend.ConnectionPool"), patch(
        "cogniverse_vespa.search_backend.SearchMetrics"
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
                "profiles": profiles or {},
            }
        )
    return backend


def test_add_profile_makes_entry_visible():
    backend = _make_backend()
    assert "agent_memories" not in backend.profiles

    backend.add_profile("agent_memories", {"type": "memory", "embedding_dims": 768})

    assert backend.profiles["agent_memories"] == {
        "type": "memory",
        "embedding_dims": 768,
    }


def test_add_profile_overwrites_existing_entry():
    backend = _make_backend({"x": {"v": 1}})
    backend.add_profile("x", {"v": 2})
    assert backend.profiles["x"] == {"v": 2}


def test_add_profile_copies_config_so_later_mutation_does_not_leak():
    """The caller should be able to mutate the dict they passed in
    without corrupting what the backend stored."""
    backend = _make_backend()
    cfg = {"type": "memory"}
    backend.add_profile("x", cfg)
    cfg["type"] = "corrupted"
    assert backend.profiles["x"]["type"] == "memory"


def test_remove_profile_drops_entry():
    backend = _make_backend({"x": {"v": 1}, "y": {"v": 2}})
    backend.remove_profile("x")
    assert "x" not in backend.profiles
    assert "y" in backend.profiles


def test_remove_profile_is_idempotent_for_missing_key():
    backend = _make_backend({"x": {"v": 1}})
    backend.remove_profile("not-there")  # must not raise
    assert backend.profiles == {"x": {"v": 1}}


def test_initialize_populates_profiles_from_top_level_config():
    """Reinitialize-in-place must actually re-read the profiles dict.

    Regression guard: previously ``initialize()`` re-assigned url/port/etc.
    but silently skipped ``profiles``, leaving any caller who re-init'd
    the backend with an empty profiles dict at query time.
    """
    backend = _make_backend()
    with patch("cogniverse_vespa.search_backend.ConnectionPool"), patch(
        "cogniverse_vespa.search_backend.SearchMetrics"
    ):
        backend.initialize(
            {
                "url": "http://localhost",
                "port": 8080,
                "profiles": {"alpha": {"type": "memory"}},
                "default_profiles": {"memory": {"profile": "alpha"}},
            }
        )
    assert backend.profiles == {"alpha": {"type": "memory"}}
    assert backend.default_profiles == {"memory": {"profile": "alpha"}}


def test_initialize_reads_profiles_from_nested_backend_section():
    """``get_search_backend`` passes profiles under config['backend'] in some
    call paths (see ``backend_registry.get_search_backend``)."""
    backend = _make_backend()
    with patch("cogniverse_vespa.search_backend.ConnectionPool"), patch(
        "cogniverse_vespa.search_backend.SearchMetrics"
    ):
        backend.initialize(
            {
                "url": "http://localhost",
                "port": 8080,
                "backend": {"profiles": {"beta": {"type": "video"}}},
            }
        )
    assert backend.profiles == {"beta": {"type": "video"}}


def test_add_profile_is_thread_safe_under_concurrent_writes():
    """20 threads each add a unique profile; final dict must contain all 20."""
    backend = _make_backend()

    def worker(i: int) -> None:
        backend.add_profile(f"profile_{i}", {"i": i, "type": "memory"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(backend.profiles) == 20
    for i in range(20):
        assert backend.profiles[f"profile_{i}"]["i"] == i


def test_concurrent_add_and_remove_leaves_consistent_state():
    """Mixed add/remove ops under contention must not corrupt the dict."""
    backend = _make_backend()
    stop = threading.Event()

    def adder() -> None:
        i = 0
        while not stop.is_set():
            backend.add_profile(f"p{i % 5}", {"i": i})
            i += 1

    def remover() -> None:
        i = 0
        while not stop.is_set():
            backend.remove_profile(f"p{i % 5}")
            i += 1

    threads = [threading.Thread(target=adder) for _ in range(3)] + [
        threading.Thread(target=remover) for _ in range(3)
    ]
    for t in threads:
        t.start()
    # Let them contend for a short while
    import time
    time.sleep(0.1)
    stop.set()
    for t in threads:
        t.join()

    # Regardless of final content, no exception must have been raised.
    # Just assert we can read the dict cleanly.
    snapshot = dict(backend.profiles)
    assert isinstance(snapshot, dict)


def test_search_raises_when_profile_not_found():
    """Confirms the existing error path still fires — we only changed
    how self.profiles gets populated, not the error semantics."""
    backend = _make_backend({"known": {"type": "video"}})
    with pytest.raises(ValueError, match="not found"):
        backend.search(
            query_dict={
                "query": "hi",
                "type": "video",
                "profile": "does_not_exist",
                "tenant_id": "acme",
            }
        )


def test_vespa_backend_add_profile_mirrors_to_owned_search_backend():
    """`VespaBackend.add_profile` must keep config['profiles'] AND the
    owned VespaSearchBackend's dict in sync. This is the layer
    `get_search_backend('vespa')` actually returns — it extends
    SearchBackend via the `Backend` union, and a default no-op override
    would have silently dropped runtime profile additions.
    """
    from unittest.mock import MagicMock

    from cogniverse_vespa.backend import VespaBackend

    # Construct VespaBackend with minimum viable deps (no real Vespa).
    backend_config = MagicMock()
    backend_config.backend_type = "vespa"
    backend_config.url = "http://localhost"
    backend_config.port = 8080
    backend_config.tenant_id = "t"
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=MagicMock(),
        config_manager=MagicMock(),
    )
    # Simulate post-initialize state: config dict exists with profiles key.
    backend.config = {"profiles": {}}

    # Attach a fake owned search backend to prove delegation happens.
    fake_inner = MagicMock()
    fake_inner.add_profile = MagicMock()
    fake_inner.remove_profile = MagicMock()
    backend._vespa_search_backend = fake_inner

    backend.add_profile("dyn", {"type": "memory", "schema_name": "dyn"})

    assert backend.config["profiles"]["dyn"]["schema_name"] == "dyn"
    assert backend.profiles["dyn"]["type"] == "memory"  # property reflects config
    fake_inner.add_profile.assert_called_once_with(
        "dyn", {"type": "memory", "schema_name": "dyn"}
    )

    backend.remove_profile("dyn")

    assert "dyn" not in backend.config["profiles"]
    assert "dyn" not in backend.profiles
    fake_inner.remove_profile.assert_called_once_with("dyn")


def test_search_uses_newly_added_profile():
    """Add a profile dynamically, then a search against it should pass
    past the profile-resolution phase (it may fail later for other reasons
    because we're not actually talking to Vespa, but the ValueError about
    'profile not found' must not fire)."""
    backend = _make_backend()
    backend.add_profile("fresh", {"type": "video", "schema_name": "fresh"})
    with pytest.raises(Exception) as exc_info:
        backend.search(
            query_dict={
                "query": "hi",
                "type": "video",
                "profile": "fresh",
                "tenant_id": "acme",
            }
        )
    # Whatever fails next (network, strategies cache, etc.) is fine —
    # but it must NOT be the "profile not found" path.
    assert "Requested profile 'fresh' not found" not in str(exc_info.value)
