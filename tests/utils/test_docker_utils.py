"""Unit tests for the Vespa test-port allocator.

``generate_unique_ports`` must hand back a pair that is actually bindable so a
leftover container from a crashed prior run (or a concurrent session) can't
cause an ``address already in use`` failure when the test starts its Vespa
container — the CI flake this allocator was hardened to prevent.
"""

import socket

from tests.utils import docker_utils
from tests.utils.docker_utils import _port_is_free, generate_unique_ports


def test_returns_free_pair_with_standard_offset():
    http_port, config_port = generate_unique_ports("tests.unit.docker_utils")
    # config is always http + 10991 (callers re-derive it from http).
    assert config_port == http_port + 10991
    # range keeps config_port < 65535.
    assert 40000 <= http_port <= 54544
    assert config_port < 65535
    # the returned ports are genuinely bindable right now.
    assert _port_is_free(http_port)
    assert _port_is_free(config_port)


def test_port_is_free_detects_a_bound_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        bound = s.getsockname()[1]
        assert _port_is_free(bound) is False
    # once the socket is closed the port frees up again.
    assert _port_is_free(bound) is True


def test_skips_a_busy_candidate(monkeypatch):
    """A candidate whose http or config port is in use is rejected; the next
    free candidate is returned."""
    import random

    candidates = iter([45000, 46000])
    monkeypatch.setattr(random, "randint", lambda a, b: next(candidates))

    def fake_free(port: int) -> bool:
        return port not in (45000, 45000 + 10991)

    monkeypatch.setattr(docker_utils, "_port_is_free", fake_free)

    http_port, config_port = generate_unique_ports("m")
    assert (http_port, config_port) == (46000, 46000 + 10991)


def test_falls_back_to_deterministic_hash_when_nothing_is_free(monkeypatch):
    """If probing never finds a free pair, fall back to the module+PID hash so
    behaviour degrades to the old deterministic allocation rather than hanging."""
    monkeypatch.setattr(docker_utils, "_port_is_free", lambda port: False)

    http_port, config_port = generate_unique_ports("tests.fallback")
    assert config_port == http_port + 10991
    assert 40000 <= http_port < 54544
    # deterministic for a fixed module+PID.
    assert generate_unique_ports("tests.fallback") == (http_port, config_port)
