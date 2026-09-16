"""Guard against fixes that make a test pass by weakening what it proves.

A change under ``tests/`` may not reduce the number of assertions in a
file, and may not introduce a skip, an xfail, or one of the unbounded
assertion forms the project bans (``is not None``, ``>= 1``, ``> 0``,
bare truthiness). Those forms pass when ranking is inverted, when the
wrong document comes back, and when the value is empty.

The comparison base defaults to ``HEAD~1`` and is overridable with
``ASSERTION_GUARD_BASE`` so CI can point it at a merge base.

A file the branch adds has no state at the base, so the range diff cannot
see a later commit strip its assertions. Such a file is therefore also
checked commit by commit against its own previous state.
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


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _run_git(args: list[str], repo: Path = REPO_ROOT, hint: str = "") -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(
            f"assertion-strength guard could not run `git {' '.join(args)}` in "
            f"{repo}: {proc.stderr.strip()[:400]}.{hint}"
        )
    return proc.stdout


def _git_diff(base: str, repo: Path = REPO_ROOT) -> str:
    return _run_git(
        ["diff", "--unified=0", f"{base}...HEAD", "--", "tests/"],
        repo,
        hint=(
            " Set ASSERTION_GUARD_BASE to a reachable ref; the guard fails "
            "closed rather than skip."
        ),
    )


def added_test_files(base: str, repo: Path = REPO_ROOT) -> list[str]:
    """Return the ``tests/`` files that exist at HEAD but not at ``base``."""
    listing = _run_git(
        ["diff", "--name-only", "--diff-filter=A", f"{base}...HEAD", "--", "tests/"],
        repo,
    )
    return [line for line in listing.splitlines() if line.endswith(".py")]


def _assertion_count(sha: str, path: str, repo: Path = REPO_ROOT) -> int:
    """Count the assertions ``path`` holds at ``sha``."""
    text = _run_git(["show", f"{sha}:{path}"], repo)
    return sum(1 for line in text.splitlines() if _ASSERT.match("+" + line))


def intra_branch_weakening(
    base: str, repo: Path = REPO_ROOT
) -> dict[str, dict[str, object]]:
    """Return files the branch added that end weaker than the branch made them.

    Keyed ``"<sha> <path>"`` by the commit that held the peak. The range diff
    against ``base`` reports such a file as wholly added, so every commit that
    touched it is counted and the final state must hold at least the peak.
    """
    offenders: dict[str, dict[str, object]] = {}
    for path in added_test_files(base, repo):
        shas = _run_git(
            ["log", "--no-merges", "--format=%H", f"{base}..HEAD", "--", path], repo
        ).split()
        counts = [(sha, _assertion_count(sha, path, repo)) for sha in reversed(shas)]
        final = _assertion_count("HEAD", path, repo)
        peak_sha, peak = max(counts, key=lambda item: item[1])
        if final < peak:
            offenders[f"{peak_sha} {path}"] = {"peak": peak, "final": final}
    return offenders


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


def test_no_intra_branch_weakening_of_tests_the_branch_added():
    base = os.environ.get("ASSERTION_GUARD_BASE", "HEAD~1")
    offenders = intra_branch_weakening(base)
    assert offenders == {}, (
        "these test files the branch added end with fewer assertions than the "
        f"branch once gave them (base={base}): "
        + "; ".join(
            f"{k}: peak {f['peak']}, final {f['final']}"
            for k, f in sorted(offenders.items())
        )
    )


def _commit(repo: Path, message: str, path: str) -> str:
    _run_git(["add", "--", path], repo)
    _run_git(["commit", "-q", "-m", message, "--", path], repo)
    return _run_git(["rev-parse", "HEAD"], repo).strip()


def _branch_repo(
    tmp_path: Path, second_version: str, *versions: str
) -> tuple[Path, str]:
    """Build a repo whose branch adds a test file and then rewrites it.

    Returns the repository and the sha of the commit that added the file.
    Further ``versions`` are committed after ``second_version``. This is the
    shape the range diff cannot see: at ``main`` the file does not exist, so
    the whole branch reads as one wholly added file.
    """
    repo = tmp_path / "repo"
    (repo / "tests" / "foo").mkdir(parents=True)
    _run_git(["init", "-q", "-b", "main", str(repo)], tmp_path)
    _run_git(["config", "user.email", "guard@example.invalid"], repo)
    _run_git(["config", "user.name", "Guard"], repo)
    (repo / "README.md").write_text("base\n")
    _commit(repo, "Seed the repository", "README.md")

    _run_git(["checkout", "-q", "-b", "work"], repo)
    target = repo / "tests" / "foo" / "test_x.py"
    target.write_text(
        "def test_notices():\n"
        "    assert [m.value for m in app.info] == []\n"
        "    assert len(notices) == 1\n"
        "    assert '503' in notices[0]\n"
    )
    added = _commit(repo, "Add the test", "tests/foo/test_x.py")

    for index, version in enumerate((second_version, *versions)):
        target.write_text(version)
        _commit(repo, f"Rewrite the test {index}", "tests/foo/test_x.py")
    return repo, added


_WEAKENED = (
    "def test_notices():\n"
    "    notices = [element.value for element in app.error]\n"
    "    assert notices == [_EMPTY_WINDOW_NOTICE]\n"
)

_RESTORED_SHORT = (
    "def test_notices():\n"
    "    notices = [element.value for element in app.error]\n"
    "    assert [m.value for m in app.info] == []\n"
    "    assert len(notices) == 1\n"
    "    assert notices == [_EMPTY_WINDOW_NOTICE]\n"
)

_PRESERVED = (
    "def test_notices():\n"
    "    notices = [element.value for element in app.error]\n"
    "    assert [m.value for m in app.info] == []\n"
    "    assert len(notices) == 1\n"
    "    assert '503' in notices[0]\n"
    "    assert notices == [_EMPTY_WINDOW_NOTICE]\n"
)


def test_detector_catches_a_later_commit_stripping_a_branch_added_test(tmp_path):
    repo, sha = _branch_repo(tmp_path, _WEAKENED)
    assert intra_branch_weakening("main", repo) == {
        f"{sha} tests/foo/test_x.py": {"peak": 3, "final": 1}
    }


def test_detector_passes_a_stripped_test_the_branch_restores(tmp_path):
    repo, _ = _branch_repo(tmp_path, _WEAKENED, _PRESERVED)
    assert intra_branch_weakening("main", repo) == {}


def test_detector_catches_a_restore_that_stops_short_of_the_peak(tmp_path):
    repo, sha = _branch_repo(tmp_path, _PRESERVED, _WEAKENED, _RESTORED_SHORT)
    offenders = intra_branch_weakening("main", repo)
    assert list(offenders.values()) == [{"peak": 4, "final": 3}]
    (key,) = offenders
    assert key.endswith(" tests/foo/test_x.py")
    assert key.split(" ")[0] != sha


def test_detector_passes_a_rewrite_that_keeps_every_assertion(tmp_path):
    repo, _ = _branch_repo(tmp_path, _PRESERVED)
    assert intra_branch_weakening("main", repo) == {}


def test_range_diff_alone_misses_the_branch_added_weakening(tmp_path):
    """The blind spot the per-commit walk exists to cover.

    Against the merge base the weakened file is wholly added, so the range
    diff reports nothing at all — the loss is only visible commit by commit.
    """
    repo, _ = _branch_repo(tmp_path, _WEAKENED)
    assert analyze_diff(_git_diff("main", repo)) == {}
    assert added_test_files("main", repo) == ["tests/foo/test_x.py"]


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
