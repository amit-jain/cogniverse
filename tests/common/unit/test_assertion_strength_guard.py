"""Guard against fixes that make a test pass by weakening what it proves.

A change under ``tests/`` may not reduce the number of assertions in a
file, and may not introduce a skip, an xfail, or one of the unbounded
assertion forms the project bans (``is not None``, ``>= 1``, ``> 0``,
bare truthiness). Those forms pass when ranking is inverted, when the
wrong document comes back, and when the value is empty.

The comparison base defaults to ``HEAD~1`` and is overridable with
``ASSERTION_GUARD_BASE`` so CI can point it at a merge base.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# ``expect(...)`` raises on failure exactly as ``assert`` does, and against a
# rendered page it is the stronger form: it retries until the condition holds
# rather than sampling once. Counting only ``assert`` made this guard reward
# the sampling form it exists to discourage.
_ASSERT = re.compile(r"^[+-]\s*(assert\b|expect\()")
_WEAK_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("skip", re.compile(r"pytest\.skip\(")),
    ("xfail", re.compile(r"pytest\.mark\.xfail")),
    ("is-not-none", re.compile(r"^\s*assert\s+[^=<>!]+\bis not None\s*(#.*)?$")),
    ("len-at-least-one", re.compile(r"^\s*assert\s+len\([^)]*\)\s*>=\s*1\s*(#.*)?$")),
    ("greater-than-zero", re.compile(r"^\s*assert\s+[^=<>!]+>\s*0\s*(#.*)?$")),
)


def analyze_diff(diff: str) -> dict[str, dict[str, object]]:
    """Return per-file assertion deltas and newly introduced weak forms.

    Only files present in both revisions are reported; a wholly added or
    deleted file has no "before" to weaken.
    """
    findings: dict[str, dict[str, object]] = {}
    path: str | None = None
    created = False
    deleted = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1]
            created = deleted = False
            findings[path] = {"removed": 0, "added": 0, "weak": []}
            continue
        if path is None:
            continue
        if line.startswith("new file mode"):
            created = True
        elif line.startswith("deleted file mode"):
            deleted = True
        if created or deleted:
            findings.pop(path, None)
            path = None
            continue
        if line.startswith(("+++", "---")):
            continue
        if _ASSERT.match(line):
            key = "added" if line.startswith("+") else "removed"
            findings[path][key] = int(findings[path][key]) + 1  # type: ignore[arg-type]
        if line.startswith("+"):
            body = line[1:]
            added = findings[path].setdefault("added_lines", [])
            assert isinstance(added, list)
            added.append(body)
            for name, pattern in _WEAK_FORMS:
                if pattern.search(body):
                    weak = findings[path]["weak"]
                    assert isinstance(weak, list)
                    weak.append((name, body.strip()))

    result: dict[str, dict[str, object]] = {}
    for path, entry in findings.items():
        added_lines = entry.pop("added_lines", [])
        if not path.startswith("tests/"):
            continue
        assert isinstance(added_lines, list)
        weak = entry["weak"]
        assert isinstance(weak, list)
        entry["weak"] = [
            (name, text)
            for name, text in weak
            if not _is_guarded_none_check(name, text, added_lines)
        ]
        result[path] = entry
    return result


def _is_guarded_none_check(name: str, text: str, added_lines: list[str]) -> bool:
    """Report whether an ``is not None`` line is a diagnostic guard.

    It is one when the same expression is pinned exactly elsewhere in the
    change, e.g. ``assert ev is not None`` followed by
    ``assert ev["state"] == "complete"`` — the None check only buys a
    readable failure instead of a TypeError.
    """
    if name != "is-not-none":
        return False
    match = re.search(r"assert\s+(.+?)\s+is not None", text)
    if not match:
        return False
    expr = re.escape(match.group(1).strip())
    pinned = re.compile(rf"assert\s+{expr}\s*(\[[^\]]*\]|\.[A-Za-z_]\w*)?\s*==")
    return any(
        pinned.search(other) for other in added_lines if other.strip() != text.strip()
    )


def _git_diff(base: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"assertion-strength guard could not diff against {base!r}: "
            f"{proc.stderr.strip()[:400]}. Set ASSERTION_GUARD_BASE to a "
            f"reachable ref; the guard fails closed rather than skip."
        )
    return proc.stdout


def test_no_net_assertion_loss_in_changed_tests():
    base = os.environ.get("ASSERTION_GUARD_BASE", "HEAD~1")
    offenders = {
        path: f
        for path, f in analyze_diff(_git_diff(base)).items()
        if int(f["removed"]) > int(f["added"])  # type: ignore[arg-type]
    }
    assert offenders == {}, (
        "these test files lost assertions; a fix may not reduce what a test "
        f"proves (base={base}): "
        + "; ".join(
            f"{p}: -{f['removed']} +{f['added']}" for p, f in sorted(offenders.items())
        )
    )


def test_no_weak_assertion_forms_introduced():
    base = os.environ.get("ASSERTION_GUARD_BASE", "HEAD~1")
    offenders = {
        path: f["weak"]
        for path, f in analyze_diff(_git_diff(base)).items()
        if f["weak"]
    }
    assert offenders == {}, (
        f"banned skip/xfail/unbounded assertion forms introduced (base={base}): "
        + "; ".join(f"{p}: {w}" for p, w in sorted(offenders.items()))
    )


def test_detector_catches_a_removed_assertion():
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "-    assert result == {'a': 1}\n"
        "-    assert order == ['a', 'b']\n"
        "+    assert result == {'a': 1}\n"
    )
    assert analyze_diff(diff) == {
        "tests/foo/test_x.py": {"removed": 2, "added": 1, "weak": []}
    }


def test_expect_counts_as_an_assertion():
    """``expect(...)`` raises on failure, so it is an assertion.

    Counting only ``assert`` made the guard reward the weaker form: swapping
    a one-shot ``assert x.count() > 0`` for a retrying
    ``expect(x).to_have_count(1)`` read as a loss, so the guard pushed a fix
    toward the sampling form it exists to discourage.
    """
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "-    assert widgets.count() > 0\n"
        "+    expect(widgets).to_have_count(1, timeout=INTERACTION_TIMEOUT)\n"
    )
    assert analyze_diff(diff) == {
        "tests/foo/test_x.py": {"removed": 1, "added": 1, "weak": []}
    }


def test_detector_does_not_count_non_assertion_lines():
    """A loss must be visible even when the change adds other lines.

    Every other fixture here is made entirely of assertion lines, so a
    detector that counted *any* changed line scored identically on all of
    them and its own suite stayed green. This is the case that separates
    them: one assertion removed while three ordinary lines are added. A
    correct detector reports the loss; one that counts lines sees a gain.
    """
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "-    assert result == {'a': 1}\n"
        "+    # explain the setup\n"
        "+    helper = build_helper()\n"
        "+    value = helper.compute()\n"
    )
    assert analyze_diff(diff) == {
        "tests/foo/test_x.py": {"removed": 1, "added": 0, "weak": []}
    }


def test_removing_an_expect_is_still_a_loss():
    """Counting ``expect`` must not become a way to drop coverage."""
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "-    expect(rows).to_have_count(3)\n"
        "-    expect(title).to_have_text('Results')\n"
        "+    expect(rows).to_have_count(3)\n"
    )
    assert analyze_diff(diff) == {
        "tests/foo/test_x.py": {"removed": 2, "added": 1, "weak": []}
    }


def test_detector_catches_each_weak_form():
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "+    assert value is not None\n"
        "+    assert len(hits) >= 1\n"
        "+    assert count > 0\n"
        "+        pytest.skip('backend unavailable')\n"
        "+@pytest.mark.xfail\n"
    )
    names = [name for name, _ in analyze_diff(diff)["tests/foo/test_x.py"]["weak"]]
    assert names == [
        "is-not-none",
        "len-at-least-one",
        "greater-than-zero",
        "skip",
        "xfail",
    ]


def test_detector_ignores_added_and_deleted_files():
    diff = (
        "diff --git a/tests/foo/test_new.py b/tests/foo/test_new.py\n"
        "new file mode 100644\n"
        "+    assert value is not None\n"
        "diff --git a/tests/foo/test_gone.py b/tests/foo/test_gone.py\n"
        "deleted file mode 100644\n"
        "-    assert result == 1\n"
    )
    assert analyze_diff(diff) == {}


def test_detector_ignores_non_test_paths():
    diff = (
        "diff --git a/libs/core/thing.py b/libs/core/thing.py\n"
        "--- a/libs/core/thing.py\n"
        "+++ b/libs/core/thing.py\n"
        "-    assert x == 1\n"
    )
    assert analyze_diff(diff) == {}


def test_none_check_guarding_an_exact_pin_is_not_flagged():
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "+    assert second.final_event is not None\n"
        '+    assert second.final_event["state"] == "complete"\n'
    )
    assert analyze_diff(diff)["tests/foo/test_x.py"]["weak"] == []


def test_bare_none_check_without_a_pin_is_still_flagged():
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "+    assert second.final_event is not None\n"
        "+    assert other_thing == 3\n"
    )
    names = [name for name, _ in analyze_diff(diff)["tests/foo/test_x.py"]["weak"]]
    assert names == ["is-not-none"]


def test_exact_comparisons_are_not_flagged_as_weak():
    diff = (
        "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
        "--- a/tests/foo/test_x.py\n"
        "+++ b/tests/foo/test_x.py\n"
        "+    assert hits[0].id == 'doc-1'\n"
        "+    assert len(hits) == 3\n"
        "+    assert elapsed > 0.0 or exact is False\n"
    )
    assert analyze_diff(diff)["tests/foo/test_x.py"]["weak"] == []
