"""A job's timeout must cover its own steps' budgets plus setup.

GitHub cancels the whole job at ``timeout-minutes`` regardless of how much
budget its steps still hold, and the cancellation reads as a red workflow
with no failing test. Measured setup across successful integration runs
spans 3.0-8.4 minutes, dominated by the shared disk-reclaim step.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[3] / ".github" / "workflows"

# Worst measured setup (8.4 min on runtime-tests) with margin for runner variance.
SETUP_ALLOWANCE_MINUTES = 15


def _step_budget_total(job: dict) -> int:
    return sum(
        step["timeout-minutes"]
        for step in job.get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("timeout-minutes"), int)
    )


def underbudgeted_jobs(
    workflows: dict[str, dict], *, allowance: int = SETUP_ALLOWANCE_MINUTES
) -> dict[tuple[str, str], tuple[int | None, int]]:
    """Map (workflow, job) -> (declared, required) for every job that cannot
    honour its own steps plus ``allowance`` minutes of setup."""
    offenders: dict[tuple[str, str], tuple[int | None, int]] = {}
    for name, document in workflows.items():
        for job_name, job in (document.get("jobs") or {}).items():
            steps_total = _step_budget_total(job)
            if steps_total == 0:
                continue
            required = steps_total + allowance
            declared = job.get("timeout-minutes")
            if declared is None or declared < required:
                offenders[(name, job_name)] = (declared, required)
    return offenders


def _shipped_workflows() -> dict[str, dict]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def test_every_job_budget_covers_its_steps_and_setup():
    offenders = underbudgeted_jobs(_shipped_workflows())
    assert offenders == {}, "\n".join(
        f"{workflow} job {job}: timeout-minutes={declared} but its steps "
        f"declare {required - SETUP_ALLOWANCE_MINUTES} minutes and setup "
        f"measures up to {SETUP_ALLOWANCE_MINUTES}; needs >= {required}"
        for (workflow, job), (declared, required) in sorted(offenders.items())
    )


# The repo-wide assertion above cannot protect its own detector: once the last
# offender is fixed, gutting underbudgeted_jobs leaves it green. These drive
# the detector on synthetic input instead.

_GOOD = {"jobs": {"ok": {"timeout-minutes": 25, "steps": [{"timeout-minutes": 10}]}}}
_BAD = {"jobs": {"tight": {"timeout-minutes": 15, "steps": [{"timeout-minutes": 10}]}}}
_MISSING = {"jobs": {"unbounded": {"steps": [{"timeout-minutes": 10}]}}}
_NEGATIVE = {
    "jobs": {
        "impossible": {
            "timeout-minutes": 35,
            "steps": [{"timeout-minutes": 30}, {"timeout-minutes": 25}],
        }
    }
}
_NO_STEP_BUDGET = {"jobs": {"free": {"timeout-minutes": 5, "steps": [{"run": "true"}]}}}


@pytest.mark.parametrize(
    ("workflows", "expected"),
    [
        ({"good.yml": _GOOD}, {}),
        ({"bad.yml": _BAD}, {("bad.yml", "tight"): (15, 25)}),
        ({"missing.yml": _MISSING}, {("missing.yml", "unbounded"): (None, 25)}),
        ({"neg.yml": _NEGATIVE}, {("neg.yml", "impossible"): (35, 70)}),
        ({"free.yml": _NO_STEP_BUDGET}, {}),
    ],
    ids=["sufficient", "tight", "unbounded", "negative-slack", "no-step-budget"],
)
def test_detector_names_exactly_the_underbudgeted_jobs(workflows, expected):
    assert underbudgeted_jobs(workflows) == expected


def test_allowance_is_the_bound_the_detector_applies():
    workflows = {
        "w.yml": {
            "jobs": {"j": {"timeout-minutes": 20, "steps": [{"timeout-minutes": 10}]}}
        }
    }
    assert underbudgeted_jobs(workflows, allowance=10) == {}
    assert underbudgeted_jobs(workflows, allowance=15) == {("w.yml", "j"): (20, 25)}
