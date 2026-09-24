"""A conflicted deploy reposts a rebuilt package, never the stale one.

Vespa's prepare-and-activate replaces the WHOLE application. When the config
server answers 409 because another process activated meanwhile, reposting the
same zip activates a package that predates that peer — with the content-type
removal override, that drops the peer's schema and its documents.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vespa.package import ApplicationPackage, Document, Schema

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Lease:
    def __init__(self):
        self.events: list[str] = []

    def acquire(self):
        self.events.append("acquire")
        return self

    def ensure_owned(self):
        self.events.append("ensure_owned")

    def release(self):
        self.events.append("release")


class _Registry:
    def __init__(self, lease):
        self._lease = lease

    def deployment_lease(self, **kwargs):
        return self._lease


def _manager(lease):
    return VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=19071,
        schema_registry=_Registry(lease),
    )


def _package(names):
    return ApplicationPackage(
        name="cogniverse",
        schema=[Schema(name=name, document=Document()) for name in names],
    )


def _response(status_code):
    return SimpleNamespace(status_code=status_code, content=b"{}")


def test_conflict_reposts_the_survivor_set_read_after_the_conflict(monkeypatch):
    lease = _Lease()
    manager = _manager(lease)
    live = [["knowledge_graph_acme_acme"]]
    built: list[list[str]] = []

    def build_package():
        names = list(live[-1])
        built.append(names)
        return _package(names)

    posted: list[list[str]] = []
    statuses = [409, 200]

    def post(tenant_url, app_zip, fence=None):
        assert tenant_url == "http://localhost:19071/application/v2/tenant/default"
        # The fence re-checks ownership immediately before the activate.
        fence()
        posted.append(list(built[-1]))
        if len(posted) == 1:
            # A peer activates its own schema while our first attempt conflicts.
            live.append([*live[-1], "wiki_pages_globex_globex"])
        return _response(statuses[len(posted) - 1])

    monkeypatch.setattr(manager, "_post_package", post)
    monkeypatch.setattr(
        "cogniverse_vespa.vespa_schema_manager.time.sleep", lambda _: None
    )

    manager._deploy_package(build_package, allow_schema_removal=True)

    assert built == [
        ["knowledge_graph_acme_acme"],
        ["knowledge_graph_acme_acme", "wiki_pages_globex_globex"],
    ]
    assert posted == built
    assert lease.events == [
        "acquire",
        "ensure_owned",
        "ensure_owned",
        "ensure_owned",
        "ensure_owned",
        "release",
    ]


def test_lost_lease_refuses_to_activate(monkeypatch):
    class _LostLease(_Lease):
        def ensure_owned(self):
            super().ensure_owned()
            raise RuntimeError("Vespa deployment lease expired or was replaced")

    lease = _LostLease()
    manager = _manager(lease)
    posted: list[bytes] = []

    monkeypatch.setattr(
        manager,
        "_post_package",
        lambda url, zipped, fence=None: posted.append(zipped),
    )

    with pytest.raises(RuntimeError, match="lease expired or was replaced"):
        manager._deploy_package(lambda: _package(["knowledge_graph_acme_acme"]))

    assert posted == []
    assert lease.events == ["acquire", "ensure_owned", "release"]
