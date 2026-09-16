"""Each coding task owns its sandbox, and owns it alone.

Against the shared host OpenShell gateway the runtime itself dials, this pins
the three properties a per-task lease has to hold:

  * a task cannot read the previous task's files or see its processes, even
    when the previous task belonged to another tenant;
  * two tasks running at once write and read in their own sandbox, so a file
    one of them wrote is never the file the other runs;
  * a session that never became ready is destroyed and its slot freed, and a
    task that finds every slot taken is refused rather than served with an
    unbounded new container.

The production ``SandboxManager`` / ``SandboxSessionPool`` are the objects
under test; only the gateway process is the host's rather than the pod's, and
it is the same gateway the deployed runtime execs through.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Iterator

import pytest
from openshell.sandbox import SandboxError

from cogniverse_runtime.sandbox_pool import (
    SandboxCapacityError,
    SandboxPoolConfig,
    SandboxSessionPool,
)
from tests.e2e.conftest import _ensure_host_sandbox_gateway, run_async, unique_id

pytestmark = pytest.mark.e2e

EXEC_TIMEOUT_S = 120
AGENT_TYPE = "coding_agent"


@pytest.fixture(scope="module")
def sandbox_manager() -> Iterator[object]:
    """A manager bound to the live host gateway, closed when the module ends."""
    _ensure_host_sandbox_gateway()
    os.environ.pop("OPENSHELL_GATEWAY_ENDPOINT", None)

    from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

    manager = SandboxManager(policy=SandboxPolicy.REQUIRED)
    assert manager._available is True, (
        "the host OpenShell gateway must be reachable; the module fixture bootstraps it"
    )
    try:
        yield manager
    finally:
        manager.close()


def _listing(result: dict) -> set[str]:
    """The entry names one ``ls -1`` produced."""
    assert result["exit_code"] == 0, result
    return {line for line in result["stdout"].split("\n") if line}


class TestATaskCannotReachTheTaskBefore:
    """A finished task leaves nothing behind for the next tenant's task."""

    def test_the_next_tenant_sees_neither_the_files_nor_the_processes(
        self, sandbox_manager
    ):
        tenant_a = f"{unique_id('sbx_isoa')}:t1"
        tenant_b = f"{unique_id('sbx_isob')}:t1"
        marker_a = f"cogniverse-e2e-{uuid.uuid4().hex}"
        marker_b = f"cogniverse-e2e-{uuid.uuid4().hex}"

        async def first_task() -> tuple[str, set[str], set[str]]:
            async with sandbox_manager.task_session(AGENT_TYPE, tenant_a) as session:
                before = _listing(
                    await session.exec(["sh", "-c", "ls -1 /tmp"], EXEC_TIMEOUT_S)
                )
                # The marker rides in the process's own argv, so the next
                # task's process table names it if the container is reused.
                started = await session.exec(
                    [
                        "sh",
                        "-c",
                        f"touch /tmp/{marker_a} && "
                        f"cp /bin/sleep /tmp/{marker_a}.proc && "
                        f"(/tmp/{marker_a}.proc 600 &) && "
                        "sleep 1 && ls -1 /tmp",
                    ],
                    EXEC_TIMEOUT_S,
                )
                return session.session_name, before, _listing(started)

        async def second_task() -> tuple[str, set[str], set[str], dict]:
            async with sandbox_manager.task_session(AGENT_TYPE, tenant_b) as session:
                before = _listing(
                    await session.exec(["sh", "-c", "ls -1 /tmp"], EXEC_TIMEOUT_S)
                )
                after = _listing(
                    await session.exec(
                        ["sh", "-c", f"touch /tmp/{marker_b} && ls -1 /tmp"],
                        EXEC_TIMEOUT_S,
                    )
                )
                processes = await session.exec(
                    ["sh", "-c", "ps -eo args || ps -ef"], EXEC_TIMEOUT_S
                )
                return session.session_name, before, after, processes

        name_a, a_before, a_after = run_async(first_task(), timeout_s=900.0)
        name_b, b_before, b_after, processes = run_async(second_task(), timeout_s=900.0)

        # Each task's own write is the only thing that appeared in its sandbox.
        assert a_after - a_before == {marker_a, f"{marker_a}.proc"}, (
            a_before,
            a_after,
        )
        assert b_after - b_before == {marker_b}, (b_before, b_after)
        # The second task starts from a sandbox the first one never touched.
        assert marker_a not in b_before, b_before
        assert marker_a not in b_after, b_after
        assert marker_b not in a_after, a_after
        # The long-lived process the first task started is not in the second
        # task's process table.
        assert processes["exit_code"] == 0, processes
        assert marker_a not in processes["stdout"], processes
        assert name_a != name_b, (name_a, name_b)


class TestConcurrentTasksRunTheFileTheyWrote:
    """Two tasks for one tenant at once each stay inside their own sandbox."""

    def test_each_task_reads_back_exactly_its_own_write(self, sandbox_manager):
        tenant_id = f"{unique_id('sbx_conc')}:t1"
        markers = (
            f"cogniverse-e2e-{uuid.uuid4().hex}",
            f"cogniverse-e2e-{uuid.uuid4().hex}",
        )

        async def one_task(marker: str) -> tuple[str, dict, set[str]]:
            async with sandbox_manager.task_session(AGENT_TYPE, tenant_id) as session:
                await session.exec(
                    ["sh", "-c", f"printf %s {marker} > /tmp/{marker}.txt"],
                    EXEC_TIMEOUT_S,
                )
                read = await session.exec(
                    ["sh", "-c", f"cat /tmp/{marker}.txt"], EXEC_TIMEOUT_S
                )
                listing = _listing(
                    await session.exec(["sh", "-c", "ls -1 /tmp"], EXEC_TIMEOUT_S)
                )
                return session.session_name, read, listing

        async def both() -> list:
            return await asyncio.gather(*(one_task(marker) for marker in markers))

        results = run_async(both(), timeout_s=900.0)

        names = [name for name, _read, _listing in results]
        assert len(set(names)) == len(markers), names
        for marker, (_name, read, listing) in zip(markers, results):
            assert read["exit_code"] == 0, read
            assert read["stdout"] == marker, read
            peers = {f"{candidate}.txt" for candidate in markers if candidate != marker}
            assert listing & peers == set(), (marker, listing)
            assert f"{marker}.txt" in listing, (marker, listing)


class TestALeaseThatNeverBecameReadyIsReclaimed:
    """A readiness failure destroys its container and frees its slot."""

    def test_a_failed_readiness_leaves_no_session_and_no_lease(self, sandbox_manager):
        pool = SandboxSessionPool(
            sandbox_manager._client,
            config=SandboxPoolConfig(max_pool_size=1),
            wait_ready_timeout_s=0,
        )
        before = set(sandbox_manager._client.list_ids(limit=100))

        with pytest.raises(SandboxError) as raised:
            with pool.task_session():
                raise AssertionError("the lease must not be handed out")

        message = str(raised.value)
        assert message.startswith("sandbox "), message
        assert message.endswith(" was not ready within timeout"), message
        created = message[len("sandbox ") : -len(" was not ready within timeout")]
        # The container the pool made before the wait is gone, and the slot
        # it held is back.
        assert created not in set(sandbox_manager._client.list_ids(limit=100)), created
        assert set(sandbox_manager._client.list_ids(limit=100)) - before == set()
        assert pool.stats() == {"max_pool_size": 1, "task_sessions": 0}, pool.stats()

    def test_a_task_past_capacity_is_refused_without_a_new_container(
        self, sandbox_manager
    ):
        pool = SandboxSessionPool(
            sandbox_manager._client,
            config=SandboxPoolConfig(max_pool_size=1),
        )
        with pool.task_session():
            assert pool.stats() == {"max_pool_size": 1, "task_sessions": 1}, (
                pool.stats()
            )
            held = set(sandbox_manager._client.list_ids(limit=100))
            with pytest.raises(SandboxCapacityError) as raised:
                with pool.task_session():
                    raise AssertionError("the second lease must not be handed out")
            assert str(raised.value) == (
                "Sandbox task capacity reached: 1 sessions in use"
            ), str(raised.value)
            # The refusal never dialled the gateway.
            assert set(sandbox_manager._client.list_ids(limit=100)) == held
        assert pool.stats() == {"max_pool_size": 1, "task_sessions": 0}, pool.stats()
