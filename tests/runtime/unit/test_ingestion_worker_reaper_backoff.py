"""Re-drive schedule for graph-pending ingestion entries.

A graph-stage failure keeps its stream entry pending — acking or
dead-lettering would strand content in Vespa without its graph — so the
only bound on a deterministic failure is how often the reaper re-runs the
full pipeline for it. The hold before each re-drive doubles from the
sweep's idle threshold and clamps at ``GRAPH_REDRIVE_HOLD_CAP_MS``.
"""

import pytest

from cogniverse_runtime.ingestion_worker.reaper import (
    GRAPH_REDRIVE_HOLD_CAP_MS,
    graph_redrive_hold_ms,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

FIVE_MINUTES_MS = 300_000


def test_hold_doubles_from_the_idle_threshold_and_clamps_at_the_cap():
    assert GRAPH_REDRIVE_HOLD_CAP_MS == 6 * 60 * 60 * 1000
    assert [graph_redrive_hold_ms(k, base_ms=FIVE_MINUTES_MS) for k in range(11)] == [
        300_000,
        600_000,
        1_200_000,
        2_400_000,
        4_800_000,
        9_600_000,
        19_200_000,
        GRAPH_REDRIVE_HOLD_CAP_MS,
        GRAPH_REDRIVE_HOLD_CAP_MS,
        GRAPH_REDRIVE_HOLD_CAP_MS,
        GRAPH_REDRIVE_HOLD_CAP_MS,
    ]


def test_hold_stays_at_the_cap_for_any_redrive_count():
    assert (
        graph_redrive_hold_ms(10_000, base_ms=FIVE_MINUTES_MS)
        == GRAPH_REDRIVE_HOLD_CAP_MS
    )


def test_zero_idle_threshold_never_holds():
    assert [graph_redrive_hold_ms(k, base_ms=0) for k in range(8)] == [0] * 8


def test_negative_redrive_count_is_rejected():
    with pytest.raises(ValueError, match=r"^redrives must be >= 0, got -1$"):
        graph_redrive_hold_ms(-1, base_ms=FIVE_MINUTES_MS)
