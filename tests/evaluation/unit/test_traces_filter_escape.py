"""get_traces_by_experiment must safely quote profile/strategy in the filter.

Phoenix evaluates filter_condition as a Python expression; a raw single-quoted
interpolation broke on any value containing a quote.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cogniverse_evaluation.data.traces import TraceManager


def test_filter_uses_repr_quoting():
    mgr = object.__new__(TraceManager)
    captured = {}
    storage = MagicMock()

    def _get(*, start_time, filter_condition, limit):
        captured["filter"] = filter_condition
        return MagicMock(__len__=lambda self: 0)

    storage.get_traces_for_evaluation.side_effect = _get
    mgr.storage = storage

    mgr.get_traces_by_experiment(profile="pro'file", strategy="float_float")

    assert repr("pro'file") in captured["filter"]
    assert repr("float_float") in captured["filter"]
