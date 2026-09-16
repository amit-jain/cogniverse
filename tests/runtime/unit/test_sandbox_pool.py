"""Unit tests for SandboxSessionPool leasing per-task sandbox sessions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.sandbox_pool import (
    SandboxPoolConfig,
    SandboxSessionPool,
)


class _FakeSession:
    def __init__(self, name: str):
        self.id = name
        self.sandbox = MagicMock()
        self.sandbox.name = name
        self.delete_count = 0

    def delete(self):
        self.delete_count += 1


class _CountingClient:
    """Records create_session + wait_ready calls so tests can assert reuse."""

    def __init__(self):
        self.create_calls = 0
        self.wait_calls = 0
        self._next = 0
        self.created: list[_FakeSession] = []

    def create_session(self) -> _FakeSession:
        self._next += 1
        s = _FakeSession(name=f"sandbox-{self._next}")
        self.created.append(s)
        self.create_calls += 1
        return s

    def wait_ready(self, name: str, timeout_seconds: int = 0):
        self.wait_calls += 1


class TestPoolConfig:
    def test_defaults(self):
        cfg = SandboxPoolConfig()
        assert cfg.max_pool_size == 8

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_SIZE", "3")
        cfg = SandboxPoolConfig.from_environment()
        assert cfg.max_pool_size == 3


class TestCloseAllLocking:
    def test_hung_delete_during_close_does_not_block_a_new_lease(self):
        """``close_all`` destroys sessions OUTSIDE the pool lock: a gateway
        that never answers ``session.delete()`` must not freeze every other
        task behind it."""
        import threading

        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        hung_entered = threading.Event()
        release = threading.Event()
        leased: list = []

        def lease() -> None:
            with pool.task_session() as fresh:
                leased.append(fresh.id)

        with pool.task_session() as doomed:

            def hang():
                hung_entered.set()
                assert release.wait(30) is True

            doomed.delete = hang
            closer = threading.Thread(target=pool.close_all)
            closer.start()
            assert hung_entered.wait(5) is True

            leaser = threading.Thread(target=lease)
            leaser.start()
            leaser.join(2)
            still_blocked = leaser.is_alive()
            release.set()
            leaser.join(5)
            closer.join(5)

        assert still_blocked is False, (
            "close_all held the pool lock across session.delete(); a hung "
            "gateway would freeze every other task"
        )
        assert leased == ["sandbox-2"]


class TestManagerPoolLifecycle:
    """SandboxManager's lazy pool init and reconnect must keep exactly one
    pool, bound to the CURRENT client — a pool built on a pre-reconnect
    client keeps failing auth after a cert rotation while health looks
    green, and racing cold inits orphan pools whose live gateway sessions
    shutdown never reaps."""

    def test_concurrent_cold_init_builds_exactly_one_pool(self, monkeypatch):
        import threading
        import time as _time

        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        built: list = []

        class _SlowPool:
            def __init__(self, client, config=None, gateway_breaker=None):
                _time.sleep(0.02)
                built.append(self)
                self.client = client

            def close_all(self):
                pass

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _SlowPool)
        mgr = SandboxManager(policy="disabled")
        mgr._client = object()
        mgr._available = True

        results: list = []

        def grab():
            results.append(mgr._get_or_create_pool())

        threads = [threading.Thread(target=grab) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1
        assert len(results) == 16
        assert all(r is results[0] for r in results)

    def test_reconnect_rebuilds_pool_on_the_new_client(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_manager as sm_mod
        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        new_clients: list = []

        class _FakeClient:
            def __init__(self, endpoint=None, tls=None):
                new_clients.append(self)

        class _FakeTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                pass

        monkeypatch.setitem(
            sys.modules,
            "openshell",
            SimpleNamespace(SandboxClient=_FakeClient, TlsConfig=_FakeTls),
        )
        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "gw.invalid:19999")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(sm_mod, "_probe_gateway_endpoint", lambda ep: None)

        class _Pool:
            def __init__(self, client, config=None, gateway_breaker=None):
                self.client = client
                self.closed = False

            def close_all(self):
                self.closed = True

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _Pool)

        mgr = SandboxManager(policy="optional")
        old_client = object()
        mgr._client = old_client
        mgr._available = True
        stale = mgr._get_or_create_pool()
        assert stale is mgr._pool
        assert stale.client is old_client

        assert mgr.reconnect() is True

        assert stale.closed
        assert mgr._pool is None

        rebuilt = mgr._get_or_create_pool()
        assert rebuilt is not stale
        assert rebuilt.client is mgr._client
        assert rebuilt.client is new_clients[-1]

    def test_close_waits_for_in_flight_pool_build_and_reaps_it(self, monkeypatch):
        """close() racing a cold pool build must block on the pool lock and
        then reap the freshly built pool — an unlocked close slips past the
        builder and leaves a live pool behind on a closed manager."""
        import threading
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        build_entered = threading.Event()
        build_release = threading.Event()
        close_all_calls: list = []

        class _BlockingPool:
            def __init__(self, client, config=None, gateway_breaker=None):
                self.config = config
                build_entered.set()
                assert build_release.wait(5)

            def close_all(self):
                close_all_calls.append(self)

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _BlockingPool)
        mgr = SandboxManager(policy="disabled")
        mgr._client = SimpleNamespace(close=lambda: None)
        mgr._available = True

        builder = threading.Thread(target=mgr._get_or_create_pool)
        builder.start()
        try:
            assert build_entered.wait(5)

            closer = threading.Thread(target=mgr.close)
            closer.start()
            closer.join(0.3)
            assert closer.is_alive(), "close() must block until the pool build ends"
        finally:
            build_release.set()
        builder.join(5)
        closer.join(5)
        assert not builder.is_alive()
        assert not closer.is_alive()
        assert mgr._pool is None
        assert len(close_all_calls) == 1
        assert mgr._client is None


class TestManagerConnectSerialization:
    """Concurrent _connect calls (cert-rotation tick vs exec-error trigger)
    must serialize, and the client each reconnect displaces must be closed
    exactly once — otherwise the loser's grpc channel leaks."""

    @pytest.fixture
    def gateway_env(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_manager as sm_mod

        clients: list = []

        class _FakeClient:
            def __init__(self, endpoint=None, tls=None):
                self.close_calls = 0
                clients.append(self)

            def close(self):
                self.close_calls += 1

        class _FakeTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                pass

        monkeypatch.setitem(
            sys.modules,
            "openshell",
            SimpleNamespace(SandboxClient=_FakeClient, TlsConfig=_FakeTls),
        )
        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "gw.invalid:19999")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(sm_mod, "_probe_gateway_endpoint", lambda ep: None)
        return SimpleNamespace(clients=clients, client_cls=_FakeClient)

    def test_reconnect_closes_displaced_client_exactly_once(self, gateway_env):
        from cogniverse_runtime.sandbox_manager import SandboxManager

        mgr = SandboxManager(policy="disabled")
        mgr._connect()
        mgr._connect()

        clients = gateway_env.clients
        assert len(clients) == 2
        assert mgr._client is clients[1]
        assert clients[0].close_calls == 1
        assert clients[1].close_calls == 0
        assert mgr._available is True

    def test_concurrent_connects_serialize_and_close_the_loser(
        self, gateway_env, monkeypatch
    ):
        import threading

        from cogniverse_runtime.sandbox_manager import SandboxManager

        first_entered = threading.Event()
        second_entered = threading.Event()
        release = threading.Event()
        guard = threading.Lock()

        orig_init = gateway_env.client_cls.__init__

        def blocking_init(client_self, endpoint=None, tls=None):
            orig_init(client_self, endpoint=endpoint, tls=tls)
            with guard:
                first = len(gateway_env.clients) == 1
            if first:
                first_entered.set()
                assert release.wait(5)
            else:
                second_entered.set()

        monkeypatch.setattr(gateway_env.client_cls, "__init__", blocking_init)

        mgr = SandboxManager(policy="disabled")

        t_rotation = threading.Thread(target=mgr._connect)
        t_rotation.start()
        try:
            assert first_entered.wait(5)

            t_exec_error = threading.Thread(target=mgr._connect)
            t_exec_error.start()
            assert not second_entered.wait(0.3), (
                "second _connect must wait for the first, not dial concurrently"
            )
        finally:
            release.set()
        t_rotation.join(5)
        t_exec_error.join(5)
        assert not t_rotation.is_alive()
        assert not t_exec_error.is_alive()

        clients = gateway_env.clients
        assert len(clients) == 2
        assert mgr._client is clients[1]
        assert clients[0].close_calls == 1
        assert clients[1].close_calls == 0
        assert mgr._available is True


class TestResolveTlsConfig:
    """Client mTLS certs must be read from the same OPENSHELL_CONFIG_DIR tree
    the cert rotator watches — a hardcoded home path makes connect read certs
    rotation never refreshes."""

    def test_certs_resolved_from_openshell_config_dir(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime.sandbox_manager import SandboxManager

        class _RecordingTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                self.ca_path = ca_path
                self.cert_path = cert_path
                self.key_path = key_path

        monkeypatch.setitem(
            sys.modules, "openshell", SimpleNamespace(TlsConfig=_RecordingTls)
        )
        mtls = tmp_path / "gateways" / "gw-a" / "mtls"
        mtls.mkdir(parents=True)
        for name in ("ca.crt", "tls.crt", "tls.key"):
            (mtls / name).write_text("pem")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))

        mgr = SandboxManager(policy="disabled")
        tls = mgr._resolve_tls_config()

        assert isinstance(tls, _RecordingTls)
        assert tls.ca_path == mtls / "ca.crt"
        assert tls.cert_path == mtls / "tls.crt"
        assert tls.key_path == mtls / "tls.key"

    def test_incomplete_cert_set_returns_none(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime.sandbox_manager import SandboxManager

        monkeypatch.setitem(sys.modules, "openshell", SimpleNamespace(TlsConfig=object))
        mtls = tmp_path / "gateways" / "gw-a" / "mtls"
        mtls.mkdir(parents=True)
        (mtls / "ca.crt").write_text("pem")  # tls.crt / tls.key missing
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))

        mgr = SandboxManager(policy="disabled")
        assert mgr._resolve_tls_config() is None


def test_wait_ready_budget_is_not_below_the_sdk_default() -> None:
    """The pool must not undercut the SDK's own readiness budget.

    A sandbox provisions cold in ~168s on this host (the gateway controller
    logs pod-created 16:52:41 -> Pod is Ready 16:55:29), dominated by pulling
    the sandbox base image on first use. A budget below the SDK default
    turns that normal cold start into a timeout, which is what made
    test_coding_agent_full_execution_with_sandbox error in every sweep.

    The SDK default is read from its signature rather than restated here, so
    this pin follows the SDK instead of drifting from it.
    """
    import inspect

    from openshell.sandbox import SandboxClient

    sdk_default = (
        inspect.signature(SandboxClient.wait_ready)
        .parameters["timeout_seconds"]
        .default
    )
    assert isinstance(sdk_default, (int, float)), sdk_default

    pool_default = (
        inspect.signature(SandboxSessionPool.__init__)
        .parameters["wait_ready_timeout_s"]
        .default
    )
    assert pool_default >= sdk_default, (
        f"pool wait_ready budget {pool_default}s undercuts the openshell SDK "
        f"default {sdk_default}s; a cold sandbox start measured 168s"
    )


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("readiness unavailable"), TimeoutError("readiness expired")],
)
def test_readiness_failure_deletes_created_session(failure):
    client = _CountingClient()

    def fail_ready(name, timeout_seconds):
        raise failure

    client.wait_ready = fail_ready
    pool = SandboxSessionPool(client)
    leased = []
    with pytest.raises(type(failure), match=str(failure)) as raised:
        with pool.task_session() as session:
            leased.append(session.id)
    pool.close_all()
    assert raised.value is failure
    assert leased == []
    assert [session.delete_count for session in client.created] == [1]
    assert pool.stats() == {"max_pool_size": 8, "task_sessions": 0}


def test_failed_readiness_does_not_orphan_during_concurrent_recovery():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    client = _CountingClient()
    barrier = Barrier(2)

    def wait_ready(name, timeout_seconds):
        barrier.wait(timeout=5)
        if name == "sandbox-1":
            raise RuntimeError("first readiness failed")

    client.wait_ready = wait_ready
    pool = SandboxSessionPool(client)

    def lease() -> str:
        with pool.task_session() as session:
            return session.id

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lease)
        second = executor.submit(lease)
        with pytest.raises(RuntimeError, match="first readiness failed"):
            first.result(timeout=5)
        assert second.result(timeout=5) == "sandbox-2"
    pool.close_all()
    assert [session.delete_count for session in client.created] == [1, 1]
    assert pool.stats() == {"max_pool_size": 8, "task_sessions": 0}


@pytest.mark.asyncio
async def test_task_leases_are_exclusive_and_destroyed_across_tenants():
    import asyncio

    from cogniverse_runtime.sandbox_manager import SandboxManager

    client = _CountingClient()
    manager = SandboxManager(policy="disabled")
    manager._client = client
    manager._available = True
    both_acquired = asyncio.Barrier(2)

    async def run(tenant_id):
        async with manager.task_session("coding_agent", tenant_id) as session:
            await both_acquired.wait()
            assert [s.delete_count for s in client.created] == [0, 0]
            return session.session_name

    sessions = await asyncio.gather(run("prodfixagents:a"), run("prodfixagents:b"))
    assert set(sessions) == {"sandbox-1", "sandbox-2"}
    assert [s.delete_count for s in client.created] == [1, 1]
    async with manager.task_session("coding_agent", "prodfixagents:a") as session:
        assert session.session_name == "sandbox-3"
    assert [s.delete_count for s in client.created] == [1, 1, 1]


@pytest.mark.asyncio
async def test_task_cancellation_waits_for_execution_before_destroying():
    import asyncio
    import threading

    from openshell.sandbox import ExecResult

    from cogniverse_runtime.sandbox_manager import SandboxManager

    client = _CountingClient()
    manager = SandboxManager(policy="disabled")
    manager._client = client
    manager._available = True
    started = threading.Event()
    finish = threading.Event()
    events = []

    def execute(command, timeout_seconds):
        started.set()
        assert finish.wait(5) is True
        events.append("exec finished")
        return ExecResult(stdout="done\n", stderr="", exit_code=0)

    async def run():
        async with manager.task_session(
            "coding_agent", "prodfixagents:cancel"
        ) as session:
            client.created[0].exec = execute
            original_delete = client.created[0].delete

            def delete():
                events.append("deleted")
                original_delete()

            client.created[0].delete = delete
            await session.exec(["true"], timeout_seconds=5)

    task = asyncio.create_task(run())
    try:
        assert await asyncio.to_thread(started.wait, 5) is True
        task.cancel()
        await asyncio.sleep(0)
        assert events == []
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert events == ["exec finished", "deleted"]
    assert client.created[0].delete_count == 1


class TestTaskSessionCapacity:
    """Task sessions are bounded by the pool's capacity, not by concurrency."""

    def test_concurrent_tasks_beyond_capacity_are_refused(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, Lock

        from cogniverse_runtime.sandbox_pool import SandboxCapacityError

        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=3))
        attempted = Barrier(12)
        live = {"now": 0, "peak": 0}
        counter = Lock()

        def task() -> str:
            try:
                with pool.task_session() as session:
                    with counter:
                        live["now"] += 1
                        live["peak"] = max(live["peak"], live["now"])
                    # No holder releases before every worker has attempted,
                    # so the refusals cannot be an artefact of fast turnover.
                    attempted.wait(timeout=10)
                    with counter:
                        live["now"] -= 1
                    return session.id
            except SandboxCapacityError as exc:
                attempted.wait(timeout=10)
                return str(exc)

        with ThreadPoolExecutor(max_workers=12) as executor:
            outcomes = [
                f.result(timeout=15) for f in [executor.submit(task) for _ in range(12)]
            ]

        refusals = [o for o in outcomes if o.startswith("Sandbox task capacity")]
        assert sorted(o for o in outcomes if o not in refusals) == [
            "sandbox-1",
            "sandbox-2",
            "sandbox-3",
        ]
        assert refusals == ["Sandbox task capacity reached: 3 sessions in use"] * 9
        assert live["peak"] == 3
        assert client.create_calls == 3
        assert [s.delete_count for s in client.created] == [1, 1, 1]
        assert pool.stats()["task_sessions"] == 0

    def test_a_finished_task_frees_its_slot(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=1))

        names = []
        for _ in range(3):
            with pool.task_session() as session:
                assert pool.stats()["task_sessions"] == 1
                names.append(session.id)

        assert names == ["sandbox-1", "sandbox-2", "sandbox-3"]
        assert [s.delete_count for s in client.created] == [1, 1, 1]
        assert pool.stats()["task_sessions"] == 0

    def test_a_failed_creation_frees_its_slot(self):
        """A readiness failure must not consume capacity forever."""
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=1))
        failures = {"left": 1}
        ready = client.wait_ready

        def wait_ready(name, timeout_seconds):
            if failures["left"]:
                failures["left"] -= 1
                raise RuntimeError("readiness unavailable")
            ready(name, timeout_seconds)

        client.wait_ready = wait_ready

        with pytest.raises(RuntimeError, match="readiness unavailable"):
            with pool.task_session():
                pytest.fail("session yielded")
        assert pool.stats()["task_sessions"] == 0

        with pool.task_session() as session:
            assert session.id == "sandbox-2"
        assert [s.delete_count for s in client.created] == [1, 1]


class TestTaskSessionShutdown:
    """close_all reclaims a task's sandbox while the client is still open."""

    def test_close_all_destroys_a_live_task_session(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))

        with pool.task_session() as session:
            pool.close_all()
            assert session.delete_count == 1
            assert pool.stats()["task_sessions"] == 0

        assert session.delete_count == 1

    def test_manager_close_destroys_the_task_sandbox_before_the_client(self):
        from cogniverse_runtime.sandbox_manager import SandboxManager

        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        order = []

        class _Client:
            def close(self):
                order.append("client closed")

        mgr = SandboxManager(policy="disabled")
        mgr._pool = pool
        mgr._client = _Client()
        mgr._available = True

        with pool.task_session() as session:
            original_delete = session.delete

            def delete():
                order.append("sandbox deleted")
                original_delete()

            session.delete = delete
            mgr.close()

        assert order == ["sandbox deleted", "client closed"]
        assert session.delete_count == 1

    def test_a_failing_delete_does_not_replace_the_task_s_error(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))

        with pytest.raises(ValueError, match="task blew up"):
            with pool.task_session() as session:
                session.delete = lambda: (_ for _ in ()).throw(
                    RuntimeError("gateway refused the delete")
                )
                raise ValueError("task blew up")
        assert pool.stats()["task_sessions"] == 0

    def test_a_failing_delete_does_not_fail_a_finished_task(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        seen = []

        with pool.task_session() as session:
            session.delete = lambda: (_ for _ in ()).throw(
                RuntimeError("gateway refused the delete")
            )
            seen.append(session.id)

        assert seen == ["sandbox-1"]
        assert pool.stats()["task_sessions"] == 0
