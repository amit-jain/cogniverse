"""Each config version is handed to exactly one writer.

Pruning deletes old version documents, and Vespa applies a conditional put
with ``create`` to a missing document whatever its condition says. A writer
that picked its version, stalled while others appended and pruned past it,
and then wrote would therefore re-create a version another writer already
won. The fake below answers the way Vespa does for the operations the store
issues: per-document test-and-set, create-if-missing that skips the condition,
document GET, namespace-scoped visits, and the prune query.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional
from urllib.parse import unquote

import pytest
import requests

import cogniverse_vespa.config.config_store as config_store_module
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

SCHEMA = "config_metadata"
URL = "http://vespa:8080"
COORDINATES = ("acme:prod", ConfigScope.BACKEND, "probe", "k1")
CONFIG_ID = "acme:prod:backend:probe:k1"
_CONDITION = re.compile(rf"^{SCHEMA}\.version (<|==) (-?\d+)$")


def _condition_miss() -> requests.HTTPError:
    return requests.HTTPError(
        "HTTP 412: condition not met", response=SimpleNamespace(status_code=412)
    )


class FakeVespa:
    """Shared document store with Vespa's per-document write semantics."""

    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.operations: list[tuple[str, str, Optional[str]]] = []
        self.between_match_and_fill: Callable[[], None] = lambda: None

    @staticmethod
    def _holds(condition: str, fields: dict[str, Any]) -> bool:
        operator, value = _CONDITION.match(condition).groups()
        version = int(fields["version"])
        return version < int(value) if operator == "<" else version == int(value)

    def put(self, namespace, data_id, fields, condition=None, create=False) -> None:
        with self.lock:
            self.operations.append(("put", namespace, condition))
            key = (namespace, data_id)
            existing = self.docs.get(key)
            if condition is not None:
                if existing is None and not create:
                    raise _condition_miss()
                if existing is not None and not self._holds(condition, existing):
                    raise _condition_miss()
            self.docs[key] = dict(fields)

    def update(self, namespace, data_id, fields, condition=None) -> None:
        with self.lock:
            self.operations.append(("update", namespace, condition))
            existing = self.docs.get((namespace, data_id))
            if existing is None or (
                condition is not None and not self._holds(condition, existing)
            ):
                raise _condition_miss()
            existing.update(fields)

    def delete(self, namespace, data_id) -> None:
        with self.lock:
            self.operations.append(("delete", namespace, None))
            self.docs.pop((namespace, data_id), None)

    def get(self, namespace: str, data_id: str) -> tuple[int, dict]:
        with self.lock:
            fields = self.docs.get((namespace, data_id))
        if fields is None:
            return 404, {}
        return 200, {"fields": dict(fields)}

    def http_get(self, path: str, params=None, timeout=None):
        match = re.match(rf"^{URL}/document/v1/([^/]+)/{SCHEMA}/docid/(.*)$", path)
        namespace, doc_id = match.groups()
        with self.lock:
            if not doc_id:
                body = {
                    "documents": [
                        {"id": f"id:{ns}:{SCHEMA}::{i}", "fields": dict(f)}
                        for (ns, i), f in self.docs.items()
                        if ns == namespace
                    ]
                }
                return _Response(200, body)
            fields = self.docs.get((namespace, unquote(doc_id)))
        if fields is None:
            return _Response(404, {})
        return _Response(200, {"fields": dict(fields)})

    def prune_query(self, yql: str):
        """Match, then fill summaries. A document deleted in between comes
        back as a hit without fields and no ``root.errors``, as Vespa answers."""
        config_id = re.search(r'contains "([^"]+)"', yql).group(1)
        limit = int(re.search(r"limit (\d+)", yql).group(1))
        with self.lock:
            matched = sorted(
                (
                    (int(f["version"]), data_id)
                    for (ns, data_id), f in self.docs.items()
                    if ns == SCHEMA and f.get("config_id") == config_id
                ),
                reverse=True,
            )[:limit]
        self.between_match_and_fill()
        with self.lock:
            hits = [
                {"id": _hit_id(version), "fields": {"version": version}}
                if (SCHEMA, data_id) in self.docs
                else {"id": _hit_id(version), "relevance": 0.0}
                for version, data_id in matched
            ]
        return SimpleNamespace(hits=hits, json={"root": {"children": hits}})


def _hit_id(version: int) -> str:
    return f"index:cogniverse_content/0/{version:024x}"


class _Response:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> dict:
        return self._body


class App:
    """One store's client of the shared fake; ``before_write`` runs ahead of
    every version-document write, which is where a stall is injected."""

    def __init__(self, fake: FakeVespa) -> None:
        self.fake = fake
        self.url = URL
        self.before_write: Callable[[str], None] = lambda data_id: None

    def get_document_v1_path(self, id: str, schema: str, namespace: str) -> str:
        return f"/document/v1/{namespace}/{schema}/docid/{id}"

    def get_data(self, schema, data_id, namespace=None, raise_on_not_found=False):
        status, body = self.fake.get(namespace or schema, data_id)
        return SimpleNamespace(status_code=status, get_json=lambda: body)

    def feed_data_point(self, schema, data_id, fields, namespace=None, **kwargs):
        if "::" in data_id:
            self.before_write(data_id)
        self.fake.put(
            namespace or schema,
            data_id,
            fields,
            kwargs.get("condition"),
            kwargs.get("create", False),
        )

    def update_data(self, schema, data_id, fields, namespace=None, **kwargs):
        self.fake.update(namespace or schema, data_id, fields, kwargs.get("condition"))

    def delete_data(self, schema, data_id, namespace=None, **kwargs):
        self.fake.delete(namespace or schema, data_id)

    def query(self, yql: str):
        return self.fake.prune_query(yql)


@pytest.fixture
def fake(monkeypatch) -> FakeVespa:
    backend = FakeVespa()
    monkeypatch.setattr(requests, "get", backend.http_get)
    return backend


def _store(fake: FakeVespa, keep: int = 3) -> tuple[VespaConfigStore, App]:
    app = App(fake)
    return VespaConfigStore(vespa_app=app, keep_versions=keep), app


def test_a_writer_stalled_past_a_pruned_version_does_not_win_it_again(fake):
    fast, _ = _store(fake)
    slow, slow_app = _store(fake)
    first = fast.set_config(*COORDINATES, {"writer": "fast", "write": 0})
    fast_versions: list[int] = []

    def others_append_and_prune_meanwhile(data_id: str) -> None:
        slow_app.before_write = lambda data_id: None
        fast_versions.extend(
            fast.set_config(*COORDINATES, {"writer": "fast", "write": n}).version
            for n in range(1, 7)
        )

    slow_app.before_write = others_append_and_prune_meanwhile
    slow_version = slow.set_config(*COORDINATES, {"writer": "slow"}).version

    assert sorted([first.version, *fast_versions, slow_version]) == list(range(1, 9))


def test_set_config_reserves_on_the_counter_before_writing_the_version(fake):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.operations.clear()

    entry = store.set_config(*COORDINATES, {"n": 2})

    assert entry.version == 2
    assert fake.operations == [
        ("update", "config_version_counter", f"{SCHEMA}.version == 1"),
        ("put", SCHEMA, f"{SCHEMA}.version < 2"),
    ]
    assert fake.docs[("config_version_counter", CONFIG_ID)]["version"] == 2


def test_a_missing_counter_starts_at_the_stored_latest_version(fake):
    store, _ = _store(fake)
    for version in (1, 2, 3):
        fake.put(
            SCHEMA,
            f"{SCHEMA}::{CONFIG_ID}::{version}",
            {
                "config_id": CONFIG_ID,
                "tenant_id": "acme:prod",
                "scope": "backend",
                "service": "probe",
                "config_key": "k1",
                "config_value": json.dumps({"seed": version}),
                "version": version,
                "created_at": "2026-10-06T00:00:00+00:00",
                "updated_at": "2026-10-06T00:00:00+00:00",
            },
        )

    assert store.set_config(*COORDINATES, {"n": 4}).version == 4
    assert fake.docs[("config_version_counter", CONFIG_ID)]["version"] == 4


def test_compare_and_set_reports_contention_on_an_unwritten_reservation(fake):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.update(
        "config_version_counter",
        CONFIG_ID,
        {"version": 2, "updated_at": datetime.now(timezone.utc).isoformat()},
    )

    assert (
        store.compare_and_set_config(*COORDINATES, {"cas": 1}, expected_version=1)
        is None
    )
    assert fake.docs[("config_version_counter", CONFIG_ID)]["version"] == 2


def test_compare_and_set_reserves_past_an_abandoned_reservation(fake):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    abandoned_at = datetime.now(timezone.utc) - timedelta(
        seconds=config_store_module._ABANDONED_RESERVATION_SECONDS + 1
    )
    fake.update(
        "config_version_counter",
        CONFIG_ID,
        {"version": 2, "updated_at": abandoned_at.isoformat()},
    )

    written = store.compare_and_set_config(*COORDINATES, {"cas": 1}, expected_version=1)

    assert (written.version, written.config_value) == (3, {"cas": 1})
    assert store.get_config(*COORDINATES).version == 3


def test_a_late_write_to_a_reservation_compare_and_set_passed_is_not_reported(fake):
    """The abandoned reservation's writer was only slow: its write lands below
    the version compare-and-set wrote, and its own read-back reports None."""
    store, _ = _store(fake)
    late, late_app = _store(fake)
    store.compare_and_set_config(*COORDINATES, {"n": 1}, expected_version=0)

    def pass_the_reservation(data_id: str) -> None:
        late_app.before_write = lambda data_id: None
        fake.update(
            "config_version_counter",
            CONFIG_ID,
            {
                "updated_at": (
                    datetime.now(timezone.utc)
                    - timedelta(
                        seconds=config_store_module._ABANDONED_RESERVATION_SECONDS + 1
                    )
                ).isoformat()
            },
        )
        assert (
            store.compare_and_set_config(
                *COORDINATES, {"cas": 2}, expected_version=1
            ).version
            == 3
        )

    late_app.before_write = pass_the_reservation

    assert (
        late.compare_and_set_config(*COORDINATES, {"late": 2}, expected_version=1)
        is None
    )
    assert store.get_config(*COORDINATES).config_value == {"cas": 2}


def test_concurrent_stalling_writers_each_win_distinct_versions(fake):
    writers, writes = 6, 15
    stores = [_store(fake, keep=2) for _ in range(writers)]
    for _, app in stores:
        app.before_write = lambda data_id: time.sleep(random.uniform(0, 0.004))
    barrier = threading.Barrier(writers)
    versions: list[int] = []
    lock = threading.Lock()

    def write(store: VespaConfigStore) -> None:
        barrier.wait()
        for n in range(writes):
            version = store.set_config(*COORDINATES, {"n": n}).version
            with lock:
                versions.append(version)

    threads = [threading.Thread(target=write, args=(s,)) for s, _ in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(versions) == list(range(1, writers * writes + 1))
    assert stores[0][0].get_config(*COORDINATES).version == writers * writes


def test_delete_config_removes_the_counter_so_versions_restart(fake):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    store.set_config(*COORDINATES, {"n": 2})

    assert store.delete_config(*COORDINATES) is True
    assert ("config_version_counter", CONFIG_ID) not in fake.docs
    assert store.set_config(*COORDINATES, {"n": 3}).version == 1


def test_an_unreadable_counter_raises_before_any_write(fake):
    store, app = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.operations.clear()

    def refuse(**kwargs):
        raise requests.ConnectionError("connection refused")

    app.get_data = refuse

    with pytest.raises(config_store_module.ConfigStoreUnavailableError) as raised:
        store.set_config(*COORDINATES, {"n": 2})
    assert str(raised.value) == (
        f"Failed to read the version counter of config {CONFIG_ID}: "
        "ConnectionError: connection refused"
    )
    assert fake.operations == []


def _answer_counter_reads_with(app: App, answers: list[tuple[int, dict]]) -> list[int]:
    """Serve the next counter GETs from ``answers``, then the fake's own; the
    returned list grows by one entry per counter GET."""
    real_get = app.get_data
    reads: list[int] = []

    def get_data(schema, data_id, namespace=None, raise_on_not_found=False):
        if namespace == config_store_module._VERSION_COUNTER_NAMESPACE:
            reads.append(len(reads))
            if answers:
                status, body = answers.pop(0)
                return SimpleNamespace(status_code=status, get_json=lambda: body)
        return real_get(schema, data_id, namespace, raise_on_not_found)

    app.get_data = get_data
    return reads


@pytest.fixture
def backoffs(monkeypatch) -> list[int]:
    attempts: list[int] = []

    def no_wait(attempt: int) -> float:
        attempts.append(attempt)
        return 0.0

    monkeypatch.setattr(
        config_store_module, "_config_store_visit_backoff_seconds", no_wait
    )
    return attempts


def test_an_overloaded_counter_read_is_retried_and_the_write_lands(fake, backoffs):
    store, app = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    overloaded = {"pathId": f"/document/v1/{CONFIG_ID}", "message": "overloaded"}
    reads = _answer_counter_reads_with(app, [(503, overloaded), (429, overloaded)])

    entry = store.set_config(*COORDINATES, {"n": 2})

    assert entry.version == 2
    assert store.get_config(*COORDINATES).config_value == {"n": 2}
    assert backoffs == [1, 2]
    assert reads == [0, 1, 2]


def test_a_counter_read_that_stays_overloaded_raises_before_any_write(fake, backoffs):
    store, app = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.operations.clear()
    overloaded = {"message": "overloaded"}
    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    _answer_counter_reads_with(app, [(503, overloaded)] * attempts)

    with pytest.raises(config_store_module.ConfigStoreUnavailableError) as raised:
        store.set_config(*COORDINATES, {"n": 2})

    assert str(raised.value) == (
        f"Failed to read the version counter of config {CONFIG_ID} after "
        f"{attempts} attempts: HTTP 503: {overloaded}"
    )
    assert backoffs == list(range(1, attempts))
    assert fake.operations == []


def test_a_rejected_counter_read_raises_without_retrying(fake, backoffs):
    store, app = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.operations.clear()
    rejected = {"message": "bad request"}
    reads = _answer_counter_reads_with(app, [(400, rejected)])

    with pytest.raises(config_store_module.ConfigStoreUnavailableError) as raised:
        store.set_config(*COORDINATES, {"n": 2})

    assert str(raised.value) == (
        f"Failed to read the version counter of config {CONFIG_ID} after "
        f"1 attempts: HTTP 400: {rejected}"
    )
    assert reads == [0]
    assert backoffs == []
    assert fake.operations == []


def test_a_counter_answer_without_fields_raises_before_any_write(fake, backoffs):
    store, app = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    fake.operations.clear()
    bare = {"pathId": f"/document/v1/{CONFIG_ID}", "id": CONFIG_ID}
    _answer_counter_reads_with(app, [(200, bare)])

    with pytest.raises(config_store_module.ConfigStoreUnavailableError) as raised:
        store.set_config(*COORDINATES, {"n": 2})

    assert str(raised.value) == (
        f"Version counter of config {CONFIG_ID} came back without fields: {bare}"
    )
    assert backoffs == []
    assert fake.operations == []


def _versions(fake: FakeVespa) -> list[int]:
    return sorted(
        int(fields["version"])
        for (namespace, _), fields in fake.docs.items()
        if namespace == SCHEMA
    )


def test_a_version_pruned_between_another_prunes_match_and_fill_prunes_nothing(
    fake, caplog
):
    """Writer A's prune matches v3..v1; writer B appends v4 and prunes v2 and
    v1 before A's summaries fill, so A's listing carries two hits without
    fields. A's write still lands and A deletes nothing."""
    store_a, _ = _store(fake, keep=2)
    store_b, _ = _store(fake, keep=2)
    store_b.set_config(*COORDINATES, {"writer": "b", "n": 1})
    store_b.set_config(*COORDINATES, {"writer": "b", "n": 2})
    fake.operations.clear()
    a_matched, b_pruned = threading.Event(), threading.Event()
    held: list[str] = []

    def hold_the_first_listing() -> None:
        if threading.current_thread().name == "writer-a" and not held:
            held.append("writer-a")
            a_matched.set()
            assert b_pruned.wait(timeout=10)

    fake.between_match_and_fill = hold_the_first_listing
    outcome: dict[str, object] = {}

    def write_a() -> None:
        try:
            outcome["a"] = store_a.set_config(*COORDINATES, {"writer": "a"}).version
        except Exception as exc:
            outcome["a"] = f"{type(exc).__name__}: {exc}"

    thread_a = threading.Thread(target=write_a, name="writer-a")
    with caplog.at_level("WARNING", logger=config_store_module.__name__):
        thread_a.start()
        assert a_matched.wait(timeout=10)
        outcome["b"] = store_b.set_config(*COORDINATES, {"writer": "b", "n": 4}).version
        b_pruned.set()
        thread_a.join(timeout=10)

    assert outcome == {"a": 3, "b": 4}
    assert _versions(fake) == [3, 4]
    assert [op for op in fake.operations if op[0] == "delete"] == [
        ("delete", SCHEMA, None),
        ("delete", SCHEMA, None),
    ]
    assert [r.getMessage() for r in caplog.records] == [
        f"Could not list versions to prune {CONFIG_ID!r}: Vespa returned hit "
        f"{_hit_id(2)} for config {CONFIG_ID} without summary fields ['version']"
    ]


def _strip_fields(monkeypatch, stripped: Callable[[str], bool]) -> None:
    """Drop ``fields`` from a Document v1 GET answer, or from each visited
    document, whose id ``stripped`` selects; the rest is served as is."""
    real_get = requests.get

    def get(path, params=None, timeout=None):
        response = real_get(path, params=params, timeout=timeout)
        body = response.json()
        if "fields" in body and stripped(unquote(path)):
            del body["fields"]
        for document in body.get("documents", []):
            if stripped(document["id"]):
                del document["fields"]
        return response

    monkeypatch.setattr(requests, "get", get)


def test_an_immutable_config_answer_without_fields_raises_unavailable(
    fake, monkeypatch
):
    store, _ = _store(fake)
    fake.put(
        SCHEMA,
        f"{SCHEMA}::{CONFIG_ID}::1",
        {
            "config_id": CONFIG_ID,
            "tenant_id": "acme:prod",
            "scope": "backend",
            "service": "probe",
            "config_key": "k1",
            "config_value": json.dumps({"n": 1}),
            "version": 1,
            "created_at": "2026-10-06T00:00:00+00:00",
            "updated_at": "2026-10-06T00:00:00+00:00",
        },
    )
    assert store.get_immutable_config(*COORDINATES).config_value == {"n": 1}
    _strip_fields(monkeypatch, lambda doc_id: doc_id.endswith(f"{CONFIG_ID}::1"))

    with pytest.raises(config_store_module.ConfigStoreUnavailableError) as raised:
        store.get_immutable_config(*COORDINATES)

    assert str(raised.value) == (
        f"Immutable config {CONFIG_ID} came back without fields: {{}}"
    )


def test_a_visited_document_without_fields_fails_a_strict_read_by_its_id(
    fake, monkeypatch
):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    store.set_config(*COORDINATES, {"n": 2})
    stripped_id = f"id:{SCHEMA}:{SCHEMA}::{SCHEMA}::{CONFIG_ID}::1"
    _strip_fields(monkeypatch, lambda doc_id: doc_id == stripped_id)

    with pytest.raises(ValueError) as raised:
        store.get_config(*COORDINATES)

    assert str(raised.value) == (
        f"config_metadata document {stripped_id} came back without fields"
    )


def test_a_visited_document_without_fields_is_skipped_by_a_tolerant_listing(
    fake, monkeypatch, caplog
):
    store, _ = _store(fake)
    store.set_config(*COORDINATES, {"n": 1})
    store.set_config(*COORDINATES, {"n": 2})
    stripped_id = f"id:{SCHEMA}:{SCHEMA}::{SCHEMA}::{CONFIG_ID}::1"
    _strip_fields(monkeypatch, lambda doc_id: doc_id == stripped_id)

    with caplog.at_level("WARNING", logger=config_store_module.__name__):
        listed = store.list_all_configs()

    assert [(e.config_key, e.version, e.config_value) for e in listed] == [
        ("k1", 2, {"n": 2})
    ]
    assert [r.getMessage() for r in caplog.records] == [
        f"Skipping malformed config_metadata doc {stripped_id}: "
        f"config_metadata document {stripped_id} came back without fields"
    ]
