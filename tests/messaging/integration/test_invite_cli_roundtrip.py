"""``cogniverse admin invite`` must mint a token that actually registers.

The CLI speaks plain HTTP to the runtime, so this serves the real admin router
over a real socket with uvicorn and drives the real CLI function against it.
The store behind the route is a real Vespa-backed ConfigManager — the same one
production writes invite tokens to.

The assertion is the outcome the command exists to produce: the token it prints
validates through a real InviteTokenManager and resolves to the exact tenant the
operator asked for. A command that returned 200 and printed something, but wrote
a token nobody could redeem, fails here.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn
from cogniverse_cli.admin import cmd_create_invite
from fastapi import FastAPI

from cogniverse_core.messaging_auth import InviteTokenManager
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.integration]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def config_manager(shared_vespa):
    """Real Vespa-backed config store — where invite tokens actually live."""
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=shared_vespa["http_port"]
    )
    return ConfigManager(store=store)


@pytest.fixture(scope="module")
def runtime_url(config_manager):
    """The real admin router served over a real socket."""
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
        config_manager
    )

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.started:
            break
        time.sleep(0.1)
    else:
        pytest.fail("admin router did not start within 30s")

    try:
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)


class TestInviteCommand:
    def test_minted_token_registers_the_requested_tenant(
        self, runtime_url, config_manager, capsys
    ):
        tenant = f"acme:{uuid.uuid4().hex[:8]}"

        code = cmd_create_invite(runtime_url, tenant, expires_in_hours=6)

        assert code == 0
        printed = capsys.readouterr().out
        assert tenant in printed

        # The token the operator would copy out of the terminal. It appears
        # twice — once labelled, once inside the /start line — but must be one
        # distinct value.
        tokens = {w for w in printed.split() if len(w) == 32 and w.isalnum()}
        assert len(tokens) == 1, f"expected one distinct token, got {tokens}"
        token = tokens.pop()

        # It resolves, through the real manager against the real store, to the
        # tenant that was asked for — this is what makes /start <token> work.
        assert InviteTokenManager(config_manager).validate_token(token) == tenant

    def test_prints_the_start_command_the_user_sends(self, runtime_url, capsys):
        tenant = f"acme:{uuid.uuid4().hex[:8]}"

        cmd_create_invite(runtime_url, tenant, expires_in_hours=1)

        printed = capsys.readouterr().out
        tokens = [w for w in printed.split() if len(w) == 32 and w.isalnum()]
        assert f"/start {tokens[0]}" in printed

    def test_each_invocation_mints_a_distinct_token(self, runtime_url, capsys):
        tenant = f"acme:{uuid.uuid4().hex[:8]}"

        cmd_create_invite(runtime_url, tenant, expires_in_hours=1)
        first = capsys.readouterr().out
        cmd_create_invite(runtime_url, tenant, expires_in_hours=1)
        second = capsys.readouterr().out

        def only_token(text):
            found = {w for w in text.split() if len(w) == 32 and w.isalnum()}
            assert len(found) == 1, found
            return found.pop()

        assert only_token(first) != only_token(second)

    def test_unreachable_runtime_exits_nonzero_without_a_token(self, capsys):
        """A dead runtime must fail loudly, not print a token nobody can redeem."""
        dead = f"http://127.0.0.1:{_free_port()}"

        code = cmd_create_invite(dead, "acme:nobody", expires_in_hours=1)

        assert code == 2
        printed = capsys.readouterr().out
        assert "Failed to reach runtime" in printed
        assert not [w for w in printed.split() if len(w) == 32 and w.isalnum()]

    def test_runtime_error_response_exits_nonzero(self, capsys):
        """A runtime that answers but rejects the request is a failure too."""
        app = FastAPI()

        @app.post("/admin/messaging/invite")
        async def _boom():
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"detail": "store down"})

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.1)

        try:
            with httpx.Client(timeout=5.0) as probe:
                probe.post(f"http://127.0.0.1:{port}/admin/messaging/invite", json={})
            code = cmd_create_invite(
                f"http://127.0.0.1:{port}", "acme:x", expires_in_hours=1
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10)

        assert code == 3
        printed = capsys.readouterr().out
        assert "503" in printed
