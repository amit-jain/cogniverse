"""``register_tenant_and_wait`` returns only once every requested schema is live.

The helper polls Vespa's config-server schemas list. When the caller names
``base_schemas`` the poll must wait for all of them; returning on the first
one to appear would report a multi-schema tenant ready while the rest are
still deploying. A real HTTP server stands in for the config-server and the
runtime here, serving the payload shapes both really return, so the helper's
own request loop is what is under test.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.e2e import conftest as e2e_conftest

_BASES = ("wiki_pages", "provenance", "agent_memories")
_TENANT = "regwait_a1b2c3d4:t1"
_SUFFIX = _TENANT.replace(":", "_")
_SCHEMAS_PATH = (
    "/application/v2/tenant/default/application/default/"
    "environment/prod/region/default/instance/default/content/schemas/"
)
_ROW = {
    "tenant_full_id": _TENANT,
    "org_id": "regwait_a1b2c3d4",
    "tenant_name": "t1",
    "created_at": 1787000000000,
    "created_by": "regwait",
    "status": "active",
    "schemas_deployed": list(_BASES),
}


class _Runtime(BaseHTTPRequestHandler):
    """Config-server schemas list plus the two tenant routes the helper calls."""

    def __init__(self, state, *args, **kwargs):
        self._state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args):  # noqa: ANN002
        return

    def _send(self, status: int, body: object) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == _SCHEMAS_PATH:
            visible = self._state["visible"](len(self._state["polls"]))
            self._state["polls"].append(visible)
            self._send(
                200,
                [f"http://config/{name}.sd" for name in sorted(visible)],
            )
            return
        if self.path == f"/admin/tenants/{_TENANT}":
            self._send(200, _ROW)
            return
        self._send(404, {"detail": self.path})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self._state["posted"].append(json.loads(self.rfile.read(length)))
        self._send(201, _ROW)


@pytest.fixture
def served(monkeypatch):
    """Start the stand-in server and point the helper's URLs at it."""

    def start(visible):
        state = {"visible": visible, "polls": [], "posted": [], "owned": []}
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Runtime, state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setattr(e2e_conftest, "RUNTIME", base)
        monkeypatch.setattr(
            e2e_conftest, "_VESPA_SCHEMAS_LIST_URL", f"{base}{_SCHEMAS_PATH}"
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_TENANT_OWNERS",
            [("function:test", state["owned"].append)],
        )
        state["close"] = server.shutdown
        return state

    started: list = []
    try:
        yield lambda visible: started.append(start(visible)) or started[-1]
    finally:
        for state in started:
            state["close"]()


def _full() -> set[str]:
    return {f"{base}_{_SUFFIX}" for base in _BASES}


def test_named_bases_wait_for_every_one_of_them(served):
    partial_view = {f"{_BASES[0]}_{_SUFFIX}"}
    state = served(lambda n: partial_view if n < 2 else _full())

    row = e2e_conftest.register_tenant_and_wait(
        _TENANT, created_by="regwait", base_schemas=list(_BASES), timeout_s=60.0
    )

    assert row == _ROW
    assert state["polls"] == [partial_view, partial_view, _full()]
    assert state["posted"] == [
        {
            "tenant_id": _TENANT,
            "created_by": "regwait",
            "base_schemas": list(_BASES),
        }
    ]


def test_unnamed_bases_wait_for_the_first_schema_the_runtime_chose(served):
    chosen = {f"video_colpali_smol500_mv_frame_{_SUFFIX}"}
    state = served(lambda n: set() if n < 1 else chosen)

    row = e2e_conftest.register_tenant_and_wait(
        _TENANT, created_by="regwait", timeout_s=60.0
    )

    assert row == _ROW
    assert state["polls"] == [set(), chosen]
    assert state["posted"] == [{"tenant_id": _TENANT, "created_by": "regwait"}]


def test_a_partial_deploy_never_reports_ready(served):
    partial_view = {f"{base}_{_SUFFIX}" for base in _BASES[:2]}
    served(lambda n: partial_view)

    with pytest.raises(RuntimeError) as raised:
        e2e_conftest.register_tenant_and_wait(
            _TENANT, created_by="regwait", base_schemas=list(_BASES), timeout_s=6.0
        )

    assert str(raised.value) == (
        f"register_tenant_and_wait: tenant {_TENANT!r} not ready after "
        f"6 s — saw_schema=False saw_metadata=True"
    )


def test_the_created_tenant_is_owned_for_teardown(served):
    state = served(lambda n: _full())

    e2e_conftest.register_tenant_and_wait(
        _TENANT, created_by="regwait", base_schemas=list(_BASES), timeout_s=60.0
    )

    assert [(f.func, f.args, f.keywords) for f in state["owned"]] == [
        (e2e_conftest.delete_minted_tenant_and_wait, (_TENANT,), {})
    ]
