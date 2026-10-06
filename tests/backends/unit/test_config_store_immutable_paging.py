"""``list_immutable_configs`` pages are bounded even when a visit overshoots.

Vespa treats a visit's ``wantedDocumentCount`` as a hint: a visit returns
whole buckets, so it can carry more documents than asked. The store must still
return at most ``page_size`` records per page, and following the cursor must
return every record exactly once. The fake below answers document/v1 the way
Vespa does: a visit takes whole buckets until it reaches the wanted count, and
a single-document GET answers 404 for an absent id.
"""

from __future__ import annotations

import itertools
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
)
from cogniverse_vespa.config.config_store import VespaConfigStore

SCHEMA = "config_metadata"
URL = "http://vespa:8080"
TENANT = "system:system"
SERVICE = "harness_keys"


def _fields(key: str) -> dict:
    return {
        "tenant_id": TENANT,
        "scope": "system",
        "service": SERVICE,
        "config_key": key,
        "config_value": json.dumps({"key": key}),
        "version": 1,
        "created_at": "2026-10-06T00:00:00+00:00",
        "updated_at": "2026-10-06T00:00:00+00:00",
    }


class _Response:
    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status_code = status
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> dict:
        return self._body


class FakeVespa:
    """Documents grouped into buckets; a visit returns whole buckets."""

    def __init__(self, buckets: list[list[str]]) -> None:
        self.buckets = buckets
        self.deleted: set[str] = set()
        self.visits: list[dict] = []
        self.lock = threading.Lock()

    def get_document_v1_path(self, id: str, schema: str, namespace: str) -> str:
        return f"/document/v1/{namespace}/{schema}/docid/{id}"

    def get(self, path: str, params: dict | None = None, timeout: int = 0):
        prefix = f"{URL}/document/v1/{SCHEMA}/{SCHEMA}/docid/"
        assert path.startswith(prefix), path
        doc_id = path[len(prefix) :]
        if doc_id:
            key = doc_id.split("::")[1].rsplit(":", 1)[1]
            if key in self.deleted or not any(key in b for b in self.buckets):
                return _Response(404)
            return _Response(200, {"fields": _fields(key)})
        start = int(params.get("continuation", 0))
        wanted = params["wantedDocumentCount"]
        documents: list[dict] = []
        bucket = start
        while bucket < len(self.buckets) and len(documents) < wanted:
            documents.extend(
                {"id": f"id:{SCHEMA}:{SCHEMA}::{k}", "fields": _fields(k)}
                for k in self.buckets[bucket]
                if k not in self.deleted
            )
            bucket += 1
        with self.lock:
            self.visits.append({**params, "returned": len(documents)})
        body: dict = {"documents": documents}
        if bucket < len(self.buckets):
            body["continuation"] = str(bucket)
        return _Response(200, body)


@pytest.fixture
def make_store(monkeypatch):
    def make(buckets: list[list[str]]) -> tuple[VespaConfigStore, FakeVespa]:
        fake = FakeVespa(buckets)
        fake.url = URL
        monkeypatch.setattr(requests, "get", fake.get)
        return VespaConfigStore(vespa_app=fake, schema_name=SCHEMA), fake

    return make


def _scan(store: VespaConfigStore, page_size: int) -> list[list[str]]:
    pages = []
    continuation = None
    while True:
        entries, continuation = store.list_immutable_configs(
            TENANT,
            ConfigScope.SYSTEM,
            SERVICE,
            page_size=page_size,
            continuation=continuation,
        )
        pages.append([entry.config_key for entry in entries])
        if continuation is None:
            return pages


def test_a_visit_that_overshoots_the_page_size_is_cut_to_the_page(make_store):
    store, fake = make_store([["a", "b"], ["c"]])

    entries, continuation = store.list_immutable_configs(
        TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1
    )

    # The visit asked for one document and got the whole first bucket.
    assert [(v["wantedDocumentCount"], v["returned"]) for v in fake.visits] == [(1, 2)]
    assert [entry.config_key for entry in entries] == ["a"]
    assert continuation is not None
    assert _scan(store, 1) == [["a"], ["b"], ["c"]]


def test_extra_documents_are_served_before_the_visit_resumes(make_store):
    store, fake = make_store([["a", "b", "c", "d", "e"], ["f"], ["g", "h"]])

    assert _scan(store, 2) == [["a", "b"], ["c", "d"], ["e"], ["f", "g"], ["h"]]
    # The visit is resumed from Vespa's own continuation only after the
    # carried documents are served: three visits, never a re-visit.
    assert [(v.get("continuation"), v["returned"]) for v in fake.visits] == [
        (None, 5),
        ("1", 3),
    ]


@pytest.mark.parametrize(
    "buckets",
    [
        [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]],
        [["a", "b", "c", "d", "e", "f", "g"]],
        [[], ["a", "b", "c"], [], [], ["d"], ["e", "f", "g", "h", "i", "j"], []],
        [["a", "b"], ["c", "d", "e"], ["f"], ["g", "h", "i", "j", "k"], ["l"]],
    ],
)
@pytest.mark.parametrize("page_size", [1, 2, 3, 4, 5, 7, 100])
def test_every_page_size_returns_each_record_exactly_once(
    make_store, buckets, page_size
):
    store, _ = make_store(buckets)
    expected = [key for bucket in buckets for key in bucket]

    pages = _scan(store, page_size)

    assert all(len(page) <= page_size for page in pages)
    assert list(itertools.chain.from_iterable(pages)) == expected


def test_a_carried_record_deleted_before_its_page_is_dropped(make_store):
    store, fake = make_store([["a", "b", "c"], ["d"]])

    first, continuation = store.list_immutable_configs(
        TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1
    )
    fake.deleted.add("b")
    second, continuation = store.list_immutable_configs(
        TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1, continuation=continuation
    )
    rest = []
    while continuation is not None:
        page, continuation = store.list_immutable_configs(
            TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1, continuation=continuation
        )
        rest.extend(entry.config_key for entry in page)

    assert [e.config_key for e in first] == ["a"]
    assert [e.config_key for e in second] == []
    assert rest == ["c", "d"]


@pytest.mark.parametrize(
    "cursor",
    [
        "AAAAAQ==",  # a raw Vespa continuation is not a page cursor
        "v1.not-base64!",
        "v1.eyJ2aXNpdCI6IDF9",  # {"visit": 1}
        "v1.eyJ2aXNpdCI6bnVsbCwicGVuZGluZyI6WzFdfQ==",  # pending [1]
    ],
)
def test_a_cursor_this_store_did_not_issue_is_refused(make_store, cursor):
    store, _ = make_store([["a"]])

    with pytest.raises(ValueError) as excinfo:
        store.list_immutable_configs(
            TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1, continuation=cursor
        )
    assert str(excinfo.value) == "continuation is not a config-store page cursor"


def test_concurrent_scans_of_one_store_each_see_every_record_once(make_store):
    store, _ = make_store([["a", "b", "c"], ["d"], ["e", "f"], ["g"]])
    barrier = threading.Barrier(8)
    results: list[list[str]] = [[] for _ in range(8)]

    def scan(slot: int) -> None:
        barrier.wait()
        results[slot] = list(itertools.chain.from_iterable(_scan(store, 1 + slot % 3)))

    threads = [threading.Thread(target=scan, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [list("abcdefg")] * 8


def test_an_unreachable_backend_raises_instead_of_an_empty_page(monkeypatch):
    store = VespaConfigStore(vespa_app=SimpleNamespace(url=URL), schema_name=SCHEMA)

    def refuse(path, params=None, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(
        "cogniverse_vespa.config.config_store._config_store_visit_backoff_seconds",
        lambda attempt: 0.0,
    )
    with pytest.raises(ConfigStoreUnavailableError) as excinfo:
        store.list_immutable_configs(TENANT, ConfigScope.SYSTEM, SERVICE, page_size=1)
    assert str(excinfo.value).startswith(
        "Failed to read Vespa config visit after 5 attempts over "
    )
    assert str(excinfo.value).endswith(": ConnectionError: connection refused")
