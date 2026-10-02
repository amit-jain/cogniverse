"""RefreshingCache answers within its staleness ceiling and refreshes off the reader."""

import logging
import sys
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from cogniverse_foundation.caching import RefreshingCache
from cogniverse_foundation.caching import refreshing_cache as refreshing_cache_module

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

NAME = "test-cache"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Source:
    """Keyed backing values whose reads are recorded and can be held or failed.

    A read captures its value when it starts, as a store read does, then waits
    while the source is held.
    """

    def __init__(self, values: dict) -> None:
        self.values = dict(values)
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None
        self.released = threading.Event()
        self.released.set()
        self.started = threading.Event()
        self._lock = threading.Lock()

    def hold(self) -> None:
        self.released.clear()
        self.started.clear()

    def reader(self, key: str):
        def read():
            with self._lock:
                self.calls.append((key, threading.current_thread().name))
                value = self.values[key]
                error = self.error
            self.started.set()
            if not self.released.wait(timeout=10):
                raise TimeoutError("the test never released a held read")
            if error is not None:
                raise error
            return value

        return read

    def keys_read(self) -> Counter:
        with self._lock:
            return Counter(key for key, _ in self.calls)


def _cache(clock, **overrides) -> RefreshingCache:
    bounds = {
        "refresh_after_s": 10.0,
        "max_staleness_s": 60.0,
        "max_entries": 16,
    }
    bounds.update(overrides)
    return RefreshingCache(name=NAME, clock=clock, **bounds)


def _join_refreshes() -> None:
    for thread in threading.enumerate():
        if thread.name == f"{NAME}-refresh":
            thread.join(timeout=10)
            assert thread.is_alive() is False


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


def _waiting_on_a_shared_read(thread: threading.Thread) -> bool:
    frame = sys._current_frames().get(thread.ident)
    while frame is not None:
        code = frame.f_code
        if code.co_name == "result" and code.co_filename.endswith("_base.py"):
            return True
        frame = frame.f_back
    return False


def _read_concurrently(cache, source, keys: list[str]) -> list[tuple[str, str, str]]:
    ready = threading.Barrier(len(keys))
    answers: list[tuple[str, str, str]] = []
    lock = threading.Lock()

    def run(key: str) -> None:
        ready.wait(timeout=10)
        value = cache.get(key, source.reader(key))
        with lock:
            answers.append((key, value, threading.current_thread().name))

    threads = [
        threading.Thread(target=run, args=(key,), name=f"reader-{index}")
        for index, key in enumerate(keys)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert thread.is_alive() is False
    return answers


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"refresh_after_s": -1.0}, "refresh_after_s must be >= 0, got -1.0"),
        (
            {"refresh_after_s": 2.0, "max_staleness_s": 1.0},
            "max_staleness_s (1.0) must be >= refresh_after_s (2.0)",
        ),
        ({"max_entries": 0}, "max_entries must be >= 1, got 0"),
        ({"max_background_reads": 0}, "max_background_reads must be >= 1, got 0"),
    ],
)
def test_rejects_inconsistent_bounds(bounds, message):
    with pytest.raises(ValueError) as caught:
        _cache(_Clock(), **bounds)
    assert str(caught.value) == message


def test_entry_younger_than_refresh_after_is_served_without_a_read():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)

    assert cache.get("acme", source.reader("acme")) == "acme-1"
    source.values["acme"] = "acme-2"
    clock.now = 9.99

    assert cache.get("acme", source.reader("acme")) == "acme-1"
    assert source.calls == [("acme", threading.current_thread().name)]


def test_stale_entry_is_served_to_concurrent_readers_while_one_background_read_runs():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = "acme-2"
    clock.now = 10.0
    source.hold()

    answers = _read_concurrently(cache, source, ["acme"] * 16)

    # Every reader returned while the refresh was still held.
    assert source.released.is_set() is False
    assert sorted(value for _, value, _ in answers) == ["acme-1"] * 16
    assert source.started.wait(timeout=10)
    assert source.calls[1] == ("acme", f"{NAME}-refresh")
    assert len(source.calls) == 2
    source.released.set()
    _join_refreshes()

    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert len(source.calls) == 2


def test_concurrent_stale_reads_of_two_tenants_refresh_each_once_with_no_bleed():
    clock = _Clock()
    source = _Source({"acme": "acme-1", "globex": "globex-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    cache.get("globex", source.reader("globex"))
    source.values.update({"acme": "acme-2", "globex": "globex-2"})
    clock.now = 30.0
    source.hold()

    answers = _read_concurrently(cache, source, ["acme", "globex"] * 8)

    assert source.released.is_set() is False
    assert Counter((key, value) for key, value, _ in answers) == Counter(
        {("acme", "acme-1"): 8, ("globex", "globex-1"): 8}
    )
    source.released.set()
    _join_refreshes()
    assert source.keys_read() == Counter({"acme": 2, "globex": 2})
    assert [thread for _, thread in source.calls[2:]] == [f"{NAME}-refresh"] * 2
    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert cache.get("globex", source.reader("globex")) == "globex-2"
    assert source.keys_read() == Counter({"acme": 2, "globex": 2})


def test_entry_at_max_staleness_is_read_on_the_callers_thread():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = "acme-2"
    clock.now = 60.0

    assert cache.get("acme", source.reader("acme")) == "acme-2"
    caller = threading.current_thread().name
    assert source.calls == [("acme", caller), ("acme", caller)]


def test_concurrent_cold_reads_share_one_read():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    source.hold()
    releaser = threading.Thread(
        target=lambda: source.started.wait(timeout=10) and source.released.set()
    )
    releaser.start()

    answers = _read_concurrently(cache, source, ["acme"] * 12)
    releaser.join(timeout=10)

    assert [value for _, value, _ in answers] == ["acme-1"] * 12
    assert len(source.calls) == 1
    assert source.calls[0][1] in {thread for _, _, thread in answers}


def test_failed_background_read_serves_last_good_value_then_raises_at_max_staleness(
    caplog,
):
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.error = ConnectionError("config store down")
    clock.now = 10.0

    with caplog.at_level(logging.ERROR, logger=refreshing_cache_module.__name__):
        assert cache.get("acme", source.reader("acme")) == "acme-1"
        _join_refreshes()
    assert [record.getMessage() for record in caplog.records] == [
        f"{NAME}: refreshing 'acme' failed with ConnectionError: config store "
        "down; serving the value read 10.0s ago until it is 60.0s old"
    ]

    # The failure backs the key off for refresh_after_s; no read until then.
    clock.now = 19.9
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    assert len(source.calls) == 2
    clock.now = 20.0
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    _join_refreshes()
    assert len(source.calls) == 3

    clock.now = 60.0
    for _ in range(2):
        with pytest.raises(ConnectionError) as caught:
            cache.get("acme", source.reader("acme"))
        assert str(caught.value) == "config store down"
    caller = threading.current_thread().name
    assert source.calls[3:] == [("acme", caller), ("acme", caller)]

    source.error = None
    source.values["acme"] = "acme-2"
    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert len(source.calls) == 6


def test_reader_past_max_staleness_shares_a_failing_background_read():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.error = ConnectionError("config store down")
    clock.now = 59.0
    source.hold()
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    assert source.started.wait(timeout=10)
    clock.now = 60.0
    outcome: list[BaseException] = []

    def read_past_ceiling() -> None:
        try:
            cache.get("acme", source.reader("acme"))
        except BaseException as exc:
            outcome.append(exc)

    waiter = threading.Thread(target=read_past_ceiling)
    waiter.start()
    _wait_for(lambda: _waiting_on_a_shared_read(waiter), "the reader to join")
    source.released.set()
    waiter.join(timeout=10)
    _join_refreshes()

    assert waiter.is_alive() is False
    assert [(type(exc), str(exc)) for exc in outcome] == [
        (ConnectionError, "config store down")
    ]
    assert len(source.calls) == 2


def test_invalidate_detaches_a_background_read_in_flight():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = "acme-2"
    clock.now = 10.0
    source.hold()
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    assert source.started.wait(timeout=10)

    cache.invalidate(lambda key: key == "acme")
    source.values["acme"] = "acme-3"
    source.released.set()
    _join_refreshes()

    assert cache.get("acme", source.reader("acme")) == "acme-3"
    assert cache.get("acme", source.reader("acme")) == "acme-3"
    assert [thread for _, thread in source.calls] == [
        threading.current_thread().name,
        f"{NAME}-refresh",
        threading.current_thread().name,
    ]


def test_invalidate_leaves_other_keys_cached():
    clock = _Clock()
    source = _Source({"acme": "acme-1", "globex": "globex-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    cache.get("globex", source.reader("globex"))

    cache.invalidate(lambda key: key == "acme")

    assert cache.keys() == ["globex"]
    assert cache.get("globex", source.reader("globex")) == "globex-1"
    assert source.keys_read() == Counter({"acme": 1, "globex": 1})


def test_background_reads_are_bounded():
    clock = _Clock()
    source = _Source({"a": "a-1", "b": "b-1", "c": "c-1"})
    cache = _cache(clock, max_background_reads=2)
    for key in ("a", "b", "c"):
        cache.get(key, source.reader(key))
    source.values.update({"a": "a-2", "b": "b-2", "c": "c-2"})
    clock.now = 10.0
    source.hold()

    assert [cache.get(key, source.reader(key)) for key in ("a", "b", "c")] == [
        "a-1",
        "b-1",
        "c-1",
    ]
    _wait_for(lambda: len(source.calls) == 5, "both refreshes to start")
    assert source.keys_read() == Counter({"a": 2, "b": 2, "c": 1})
    source.released.set()
    _join_refreshes()

    assert cache.get("c", source.reader("c")) == "c-1"
    _join_refreshes()
    assert [cache.get(key, source.reader(key)) for key in ("a", "b", "c")] == [
        "a-2",
        "b-2",
        "c-2",
    ]
    assert source.keys_read() == Counter({"a": 2, "b": 2, "c": 2})


def test_a_refresh_that_cannot_start_leaves_the_key_refreshable(monkeypatch, caplog):
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = "acme-2"
    clock.now = 10.0

    class _Unstartable:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        refreshing_cache_module,
        "threading",
        SimpleNamespace(Thread=_Unstartable, Lock=threading.Lock),
    )
    with caplog.at_level(logging.ERROR, logger=refreshing_cache_module.__name__):
        assert cache.get("acme", source.reader("acme")) == "acme-1"
    assert [record.getMessage() for record in caplog.records] == [
        f"{NAME}: refreshing 'acme' failed with RuntimeError: can't start new "
        "thread; serving the value read 10.0s ago until it is 60.0s old"
    ]
    assert source.calls == [("acme", threading.current_thread().name)]

    monkeypatch.setattr(refreshing_cache_module, "threading", threading)
    clock.now = 20.0
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    _join_refreshes()
    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert len(source.calls) == 2


def test_entries_are_bounded_least_recently_used_first():
    clock = _Clock()
    source = _Source({"a": "a-1", "b": "b-1", "c": "c-1", "d": "d-1"})
    cache = _cache(clock, max_entries=3)
    for key in ("a", "b", "c"):
        cache.get(key, source.reader(key))
    cache.get("a", source.reader("a"))
    cache.get("d", source.reader("d"))

    assert cache.keys() == ["c", "a", "d"]
    assert cache.items() == [("c", "c-1"), ("a", "a-1"), ("d", "d-1")]
    assert len(cache) == 3


def test_zero_max_staleness_reads_on_every_call():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock, refresh_after_s=0.0, max_staleness_s=0.0)

    assert cache.get("acme", source.reader("acme")) == "acme-1"
    source.values["acme"] = "acme-2"
    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert cache.get("acme", source.reader("acme")) == "acme-2"
    assert source.calls == [("acme", threading.current_thread().name)] * 3


def test_a_held_value_the_caller_rejects_is_read_again_and_the_read_is_shared():
    clock = _Clock()
    source = _Source({"acme": frozenset({"a"})})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = frozenset({"a", "b"})

    def wants(name):
        return lambda held: name in held

    assert cache.get("acme", source.reader("acme"), accept=wants("a")) == frozenset(
        {"a"}
    )
    assert len(source.calls) == 1
    source.hold()
    releaser = threading.Thread(
        target=lambda: source.started.wait(timeout=10) and source.released.set()
    )
    releaser.start()
    ready = threading.Barrier(8)
    answers: list = []
    lock = threading.Lock()

    def ask() -> None:
        ready.wait(timeout=10)
        value = cache.get("acme", source.reader("acme"), accept=wants("b"))
        with lock:
            answers.append(value)

    askers = [threading.Thread(target=ask, name=f"asker-{i}") for i in range(8)]
    for thread in askers:
        thread.start()
    for thread in askers:
        thread.join(timeout=10)
        assert thread.is_alive() is False
    releaser.join(timeout=10)

    assert answers == [frozenset({"a", "b"})] * 8
    assert len(source.calls) == 2
    assert source.calls[1][1] in {thread.name for thread in askers}
    assert cache.get("acme", source.reader("acme"), accept=wants("b")) == frozenset(
        {"a", "b"}
    )
    assert len(source.calls) == 2


def test_a_value_keep_rejects_is_returned_unheld_and_drops_the_entry():
    clock = _Clock()
    source = _Source({"acme": frozenset({"a"})})
    cache = _cache(clock, keep=bool)
    assert cache.get("acme", source.reader("acme")) == frozenset({"a"})
    source.values["acme"] = frozenset()
    clock.now = 10.0

    assert cache.get("acme", source.reader("acme")) == frozenset({"a"})
    _join_refreshes()
    assert cache.items() == []
    assert cache.get("acme", source.reader("acme")) == frozenset()
    assert cache.items() == []
    caller = threading.current_thread().name
    assert source.calls == [
        ("acme", caller),
        ("acme", f"{NAME}-refresh"),
        ("acme", caller),
    ]


def test_put_holds_the_written_value_and_detaches_a_background_read_in_flight():
    clock = _Clock()
    source = _Source({"acme": "acme-1"})
    cache = _cache(clock)
    cache.get("acme", source.reader("acme"))
    source.values["acme"] = "acme-2"
    clock.now = 10.0
    source.hold()
    assert cache.get("acme", source.reader("acme")) == "acme-1"
    # The refresh has read the pre-write value and is held there.
    assert source.started.wait(timeout=10)

    cache.put("acme", "acme-written")
    source.released.set()
    _join_refreshes()

    assert cache.items() == [("acme", "acme-written")]
    clock.now = 19.9
    assert cache.get("acme", source.reader("acme")) == "acme-written"
    assert [thread for _, thread in source.calls] == [
        threading.current_thread().name,
        f"{NAME}-refresh",
    ]


def test_put_during_a_cold_read_answers_that_reader_and_holds_the_written_value():
    clock = _Clock()
    source = _Source({"acme": "acme-before"})
    cache = _cache(clock)
    source.hold()
    answers: list[str] = []
    reader = threading.Thread(
        target=lambda: answers.append(cache.get("acme", source.reader("acme"))),
        name="cold-reader",
    )
    reader.start()
    assert source.started.wait(timeout=10)

    cache.put("acme", "acme-written")
    source.released.set()
    reader.join(timeout=10)

    assert reader.is_alive() is False
    assert answers == ["acme-before"]
    assert cache.items() == [("acme", "acme-written")]
    assert cache.get("acme", source.reader("acme")) == "acme-written"
    assert source.calls == [("acme", "cold-reader")]


def test_put_of_a_value_keep_rejects_drops_the_entry():
    clock = _Clock()
    source = _Source({"acme": frozenset({"a"}), "globex": frozenset({"g"})})
    cache = _cache(clock, keep=bool)
    cache.get("acme", source.reader("acme"))
    cache.get("globex", source.reader("globex"))

    cache.put("acme", frozenset())

    assert cache.items() == [("globex", frozenset({"g"}))]


def test_put_evicts_least_recently_used_beyond_max_entries():
    clock = _Clock()
    source = _Source({"acme": "acme-1", "globex": "globex-1"})
    cache = _cache(clock, max_entries=2)
    cache.get("acme", source.reader("acme"))
    cache.get("globex", source.reader("globex"))

    cache.put("initech", "initech-written")

    assert cache.items() == [("globex", "globex-1"), ("initech", "initech-written")]
