"""Job gating and step order of ``.github/workflows/publish-packages.yml``.

The job ``if:`` conditions are evaluated with the GitHub expression rules they
use: ``!`` binds tighter than ``==``/``!=``, which bind tighter than ``&&``,
which binds tighter than ``||``; string comparison and ``contains`` /
``startsWith`` ignore case.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "publish-packages.yml"
)

_TOKEN = re.compile(r"\s*(\|\||&&|==|!=|!|\(|\)|,|'[^']*'|[A-Za-z_][\w.\-]*)")


def _tokens(expression: str) -> list[str]:
    tokens, position = [], 0
    expression = expression.strip()
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        assert match, f"unparsed expression at {expression[position:]!r}"
        tokens.append(match.group(1))
        position = match.end()
        while position < len(expression) and expression[position].isspace():
            position += 1
    return tokens


def _evaluate(expression: str, context: dict[str, str]) -> bool:
    tokens = _tokens(expression)
    index = 0

    def peek() -> str | None:
        return tokens[index] if index < len(tokens) else None

    def take(expected: str | None = None) -> str:
        nonlocal index
        token = tokens[index]
        assert expected is None or token == expected, (token, expected)
        index += 1
        return token

    def either():
        value = both()
        while peek() == "||":
            take()
            right = both()
            value = value or right
        return value

    def both():
        value = negation()
        while peek() == "&&":
            take()
            right = negation()
            value = value and right
        return value

    def negation():
        if peek() == "!":
            take()
            return not negation()
        return comparison()

    def comparison():
        left = primary()
        if peek() in ("==", "!="):
            operator = take()
            right = primary()
            equal = str(left).lower() == str(right).lower()
            return equal if operator == "==" else not equal
        return left

    def primary():
        token = take()
        if token == "(":
            value = either()
            take(")")
            return value
        if token.startswith("'"):
            return token[1:-1]
        if peek() == "(":
            take("(")
            first = either()
            take(",")
            second = either()
            take(")")
            haystack, needle = str(first).lower(), str(second).lower()
            if token == "contains":
                return needle in haystack
            if token == "startsWith":
                return haystack.startswith(needle)
            raise AssertionError(f"unsupported function {token}")
        return context[token]

    value = either()
    assert index == len(tokens), tokens[index:]
    return bool(value)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _runs(job: str, context: dict[str, str]) -> bool:
    return _evaluate(_workflow()["jobs"][job]["if"], context)


def _push(ref: str) -> dict[str, str]:
    return {"github.event_name": "push", "github.ref": ref, "inputs.target": ""}


def _dispatch(ref: str, target: str) -> dict[str, str]:
    return {
        "github.event_name": "workflow_dispatch",
        "github.ref": ref,
        "inputs.target": target,
    }


@pytest.mark.parametrize(
    ("context", "testpypi", "pypi", "release"),
    [
        (_push("refs/tags/v1.2.3"), False, True, True),
        (_push("refs/tags/v1.2.3-alpha.0"), True, False, False),
        (_push("refs/tags/v1.2.3-beta.1"), True, False, False),
        (_push("refs/tags/v1.2.3-rc.1"), True, False, False),
        (_dispatch("refs/tags/v1.2.3", "testpypi"), True, False, False),
        (_dispatch("refs/tags/v1.2.3", "pypi"), False, True, False),
        (_dispatch("refs/tags/v1.2.3-rc.1", "pypi"), False, True, False),
        (_dispatch("refs/tags/v1.2.3-beta.1", "pypi"), False, True, False),
        (_dispatch("refs/heads/fix-alpha-notes", "pypi"), False, True, False),
    ],
    ids=[
        "release-tag",
        "alpha-tag",
        "beta-tag",
        "rc-tag",
        "dispatch-testpypi",
        "dispatch-pypi",
        "dispatch-pypi-from-rc-tag",
        "dispatch-pypi-from-beta-tag",
        "dispatch-pypi-from-alpha-named-branch",
    ],
)
def test_each_trigger_publishes_to_exactly_its_target(context, testpypi, pypi, release):
    assert _runs("publish-testpypi", context) is testpypi
    assert _runs("publish-pypi", context) is pypi
    assert _runs("create-release", context) is release


def test_the_package_test_job_frees_disk_before_installing_the_release():
    names = [step.get("name") for step in _workflow()["jobs"]["test"]["steps"]]

    assert "Free up disk space" in names
    assert names.index("Free up disk space") < names.index("Install packages from dist")


def test_dry_run_input_names_the_tag_ref_it_needs():
    workflow = _workflow()
    dry_run = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"][
        "dry_run"
    ]

    assert "run from a v* tag ref" in dry_run["description"]


def test_the_release_install_admits_only_the_graphql_core_pre_release():
    """The wheel install names the one pre-release arize-phoenix needs, as the
    runtime and dashboard READMEs do, instead of admitting every pre-release."""
    steps = {step.get("name"): step for step in _workflow()["jobs"]["test"]["steps"]}
    [install] = [
        line.strip()
        for line in steps["Install packages from dist"]["run"].splitlines()
        if line.strip().startswith("uv pip install")
    ]

    assert "--prerelease" not in install
    assert install.endswith('"${wheels[@]}" "graphql-core>=3.3.0a0"')
    for readme in ("runtime", "dashboard"):
        text = (WORKFLOW.parents[2] / "libs" / readme / "README.md").read_text()
        assert '"graphql-core>=3.3.0a0"`' in text, readme
