"""Holdout must be served AND scoreable — never the appended synthetic tail."""

from cogniverse_runtime.optimization_cli import _split_served_holdout, is_scoreable


def _rec(i, example_id, source_text="src text", grounding=""):
    return {
        "example_id": example_id,
        "query": f"q{i}",
        "source_text": source_text,
        "grounding_context": grounding,
        "trainable": True,
    }


def test_holdout_is_served_scoreable_tail_and_synthetic_never_held_out():
    served = [_rec(i, f"span:{i:03d}") for i in range(12)]
    synthetic = [
        _rec(i, f"approved:{i:03d}", source_text="", grounding="")
        for i in range(12, 20)
    ]
    train, holdout = _split_served_holdout(served + synthetic, min_holdout=3)
    assert [r["example_id"] for r in holdout] == [
        "span:009",
        "span:010",
        "span:011",
    ]
    assert [r["example_id"] for r in train] == [f"span:{i:03d}" for i in range(9)] + [
        f"approved:{i:03d}" for i in range(12, 20)
    ]


def test_unscoreable_served_records_train_but_never_judge():
    rows = [_rec(i, f"span:{i}") for i in range(8)] + [
        _rec(9, "span:9", source_text="", grounding="")
    ]
    train, holdout = _split_served_holdout(rows, min_holdout=2)
    assert all(is_scoreable(r) for r in holdout)
    assert {"span:9"} == {r["example_id"] for r in train} - {
        r["example_id"] for r in rows[:6]
    }


def test_below_min_holdout_returns_empty_holdout():
    rows = [_rec(i, f"span:{i}") for i in range(4)] + [
        _rec(i, f"approved:{i}") for i in range(4, 104)
    ]
    train, holdout = _split_served_holdout(rows, min_holdout=10)
    assert holdout == []
    assert len(train) == 104
