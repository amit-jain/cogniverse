"""SearchMetrics must bound its latency window (no per-query memory leak).

VespaSearchBackend instances are cached for the process lifetime, so the old
unbounded ``search_latencies`` list grew by one float per query forever. The
window is now a bounded deque; the lifetime average comes from running totals.
"""

from cogniverse_vespa.search_backend import _LATENCY_WINDOW, SearchMetrics


def test_search_latencies_window_is_bounded():
    m = SearchMetrics()
    n = _LATENCY_WINDOW + 500
    for i in range(n):
        m.record_search(success=True, latency_ms=float(i), strategy="binary")

    assert m.total_searches == n  # lifetime count is unbounded/accurate
    assert len(m.search_latencies) == _LATENCY_WINDOW  # window is capped
    # The deque holds only the most-recent window.
    assert list(m.search_latencies)[-1] == float(n - 1)
    assert list(m.search_latencies)[0] == float(n - _LATENCY_WINDOW)


def test_avg_latency_is_lifetime_not_windowed():
    m = SearchMetrics()
    # Record more than one window; every latency == 10.0.
    n = _LATENCY_WINDOW + 200
    for _ in range(n):
        m.record_search(success=True, latency_ms=10.0, strategy="binary")
    # Average over ALL n searches, not just the window.
    assert m.avg_latency_ms == 10.0
    assert m.total_latency_ms == 10.0 * n


def test_avg_latency_reflects_all_values_beyond_window():
    m = SearchMetrics()
    # First window of 100ms calls, then a window of 0ms calls.
    for _ in range(_LATENCY_WINDOW):
        m.record_search(success=True, latency_ms=100.0, strategy="binary")
    for _ in range(_LATENCY_WINDOW):
        m.record_search(success=True, latency_ms=0.0, strategy="binary")
    # Lifetime average = (100*W + 0*W) / (2W) = 50, even though the window now
    # holds only the 0ms calls.
    assert m.avg_latency_ms == 50.0
    assert m.p95_latency_ms == 0.0  # window holds only the recent 0ms calls


def test_p95_over_window():
    m = SearchMetrics()
    for i in range(100):
        m.record_search(success=True, latency_ms=float(i), strategy="binary")
    # 95th percentile of 0..99 lands at 95.
    assert m.p95_latency_ms == 95.0


def test_empty_metrics_zero():
    m = SearchMetrics()
    assert m.avg_latency_ms == 0.0
    assert m.p95_latency_ms == 0.0


def test_record_search_exact_under_concurrent_threads():
    """Counters must be exact under concurrent recording — searches record
    from pool worker threads, and an unguarded ``+=`` loses counts."""
    import sys
    import threading

    metrics = SearchMetrics()
    threads_n, per_thread = 8, 500
    barrier = threading.Barrier(threads_n)

    def hammer(worker: int):
        barrier.wait()
        for i in range(per_thread):
            metrics.record_search(
                success=(i % 2 == 0),
                latency_ms=1.0,
                strategy=f"s{worker % 2}",
            )

    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=hammer, args=(n,)) for n in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(original)

    total = threads_n * per_thread
    assert metrics.total_searches == total
    assert metrics.successful_searches == total // 2
    assert metrics.failed_searches == total // 2
    assert metrics.total_latency_ms == float(total)
    assert sum(metrics.strategy_usage.values()) == total


def test_p95_stable_while_another_thread_records():
    """The percentile sorts a locked snapshot — sorting the live deque while
    another thread appends raised mid-iteration."""
    import threading

    metrics = SearchMetrics()
    for _ in range(100):
        metrics.record_search(success=True, latency_ms=5.0, strategy="s")

    stop = threading.Event()

    def writer():
        while not stop.is_set():
            metrics.record_search(success=True, latency_ms=5.0, strategy="s")

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(200):
            assert metrics.p95_latency_ms == 5.0
    finally:
        stop.set()
        thread.join()


def test_get_metrics_exports_failed_search_count():
    """A failure is countable in the exported stats, not only via the rate."""
    from unittest.mock import MagicMock, patch

    from cogniverse_vespa.search_backend import VespaSearchBackend

    with patch("cogniverse_vespa.search_backend.ConnectionPool"):
        backend = VespaSearchBackend(
            config={"url": "http://localhost", "port": 8080, "profiles": {}}
        )
    backend._tenant_schema_exists = MagicMock(return_value=True)
    backend.metrics.record_search(success=True, latency_ms=5.0, strategy="binary")
    backend.metrics.record_search(
        success=False, latency_ms=7.0, strategy="binary", error=ValueError("boom")
    )

    search = backend.get_metrics()["search_metrics"]
    assert search["total_searches"] == 2
    assert search["failed_searches"] == 1
    assert search["success_rate"] == 50.0
    assert search["error_types"] == {"ValueError": 1}
