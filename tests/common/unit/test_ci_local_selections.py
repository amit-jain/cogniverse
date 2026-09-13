"""The local runner includes the chart's CI selection."""

from __future__ import annotations

import pytest

from scripts.ci_local import build_argv, discover

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_chart_validation_is_a_selectable_local_ci_module(tmp_path):
    (tmp_path / "chart-validation.yml").write_text(
        """on: [push, pull_request]
jobs:
  chart-validation:
    steps:
      - run: uv run --no-sync python -m pytest tests/charts -q --tb=short
"""
    )
    selections = discover(tmp_path)
    assert selections == [
        {
            "module": "chart-validation",
            "paths": ["tests/charts"],
            "ignores": [],
            "marker": None,
            "env": {},
            "unit": True,
        }
    ]
    assert build_argv(selections[0]) == [
        "uv",
        "run",
        "python",
        "-m",
        "pytest",
        "tests/charts",
        "-v",
        "-p",
        "no:cacheprovider",
        "--tb=long",
    ]


def test_a_chart_integration_job_is_selected_as_a_service_job(tmp_path):
    (tmp_path / "chart-validation.yml").write_text(
        """on: [push, pull_request]
jobs:
  integration:
    steps:
      - run: uv run python -m pytest tests/foundation/integration -v
"""
    )
    assert discover(tmp_path) == [
        {
            "module": "chart-validation",
            "paths": ["tests/foundation/integration"],
            "ignores": [],
            "marker": None,
            "env": {},
            "unit": False,
        }
    ]
