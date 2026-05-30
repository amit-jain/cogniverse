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
