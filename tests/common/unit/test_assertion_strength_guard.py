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

An assertion moved verbatim into a file the change creates (a helper moved
to a new module) is not a loss: each such line in a created ``tests/`` file
credits one identical removal elsewhere, and the per-commit check above keeps
the created file from shedding it later.

A net loss is accepted only when ``tests/common/assertion_waivers.toml``
waives it: the lost assertions tested code that no longer exists. Each waiver
names the test file, the largest net loss it may have, and the removed symbols
(``path:Name``, a top-level name) the assertions tested. The guard checks every
waiver: each symbol must be gone at HEAD and must have existed in HEAD's
history, the file's net loss must not exceed the waived count, and a waiver
whose file lost nothing fails as stale. A waiver applies only when one of its
symbols existed at the base; one whose symbols were all gone before the range
covered an earlier change and waives nothing.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import tomllib
from collections import Counter
from dataclasses import dataclass
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


def moved_assertions(diff: str) -> dict[str, int]:
    """Return, per existing ``tests/`` file, how many of its removed assertions
    a file the same change creates adds back verbatim.

    Lines compare with surrounding whitespace stripped. Each added line in a
    created file credits at most one removal.
    """
    created: Counter[str] = Counter()
    removed: dict[str, list[str]] = {}
    path: str | None = None
    created_file = deleted_file = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1]
            created_file = deleted_file = False
            continue
        if path is None or not path.startswith("tests/"):
            continue
        if line.startswith("new file mode"):
            created_file = True
        elif line.startswith("deleted file mode"):
            deleted_file = True
        if line.startswith(("+++", "---")) or not _ASSERT.match(line):
            continue
        if created_file and line.startswith("+"):
            created[line[1:].strip()] += 1
        elif not (created_file or deleted_file) and line.startswith("-"):
            removed.setdefault(path, []).append(line[1:].strip())

    moved: dict[str, int] = {}
    for path in sorted(removed):
        for text in removed[path]:
            if created[text]:
                created[text] -= 1
                moved[path] = moved.get(path, 0) + 1
    return moved


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


def net_assertion_losses(diff: str) -> dict[str, dict[str, int]]:
    """``{path: {"removed", "moved", "added"}}`` for every changed test file
    whose removals, less those moved into created files, exceed its additions."""
    moved = moved_assertions(diff)
    losses: dict[str, dict[str, int]] = {}
    for path, f in analyze_diff(diff).items():
        removed, added = int(f["removed"]), int(f["added"])  # type: ignore[arg-type]
        if removed - moved.get(path, 0) > added:
            losses[path] = {
                "removed": removed,
                "moved": moved.get(path, 0),
                "added": added,
            }
    return losses


WAIVERS_PATH = REPO_ROOT / "tests" / "common" / "assertion_waivers.toml"


@dataclass(frozen=True)
class Waiver:
    """An accepted net assertion loss in one test file."""

    file: str
    max_net_loss: int
    removed_symbols: tuple[str, ...]
    reason: str


def load_waivers(path: Path = WAIVERS_PATH) -> list[Waiver]:
    entries = tomllib.loads(path.read_text(encoding="utf-8")).get("waiver", [])
    waivers = []
    for entry in entries:
        if set(entry) != {"file", "max_net_loss", "removed_symbols", "reason"}:
            raise ValueError(f"malformed assertion waiver: {entry!r}")
        if not entry["removed_symbols"] or not entry["reason"].strip():
            raise ValueError(f"waiver for {entry['file']} names no symbol or reason")
        waivers.append(
            Waiver(
                file=entry["file"],
                max_net_loss=int(entry["max_net_loss"]),
                removed_symbols=tuple(entry["removed_symbols"]),
                reason=entry["reason"],
            )
        )
    return waivers


def _defines(source: str, name: str) -> bool:
    """Whether ``source`` binds ``name`` at module level."""
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
    return False


def symbol_defined(sha: str, symbol: str, repo: Path = REPO_ROOT) -> bool:
    """Whether ``path:Name`` is a top-level binding of ``path`` at ``sha``."""
    path, _, name = symbol.partition(":")
    if not path or not name:
        raise ValueError(f"waiver symbol must be 'path:Name', got {symbol!r}")
    proc = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=repo, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return False
    return _defines(proc.stdout, name)


def symbol_ever_defined(symbol: str, repo: Path = REPO_ROOT) -> bool:
    """Whether any commit reachable from HEAD defined ``path:Name``."""
    path = symbol.partition(":")[0]
    commits = _run_git(["rev-list", "HEAD", "--", path], repo).split()
    return any(symbol_defined(f"{sha}~1", symbol, repo) for sha in commits) or any(
        symbol_defined(sha, symbol, repo) for sha in commits
    )


def apply_waivers(
    losses: dict[str, dict[str, int]],
    waivers: list[Waiver],
    base: str,
    repo: Path = REPO_ROOT,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Return the losses no waiver covers and every waiver that fails its check."""
    remaining = dict(losses)
    errors: list[str] = []
    for waiver in waivers:
        removed_in_range = False
        broken = False
        for symbol in waiver.removed_symbols:
            if symbol_defined("HEAD", symbol, repo):
                errors.append(f"{waiver.file}: {symbol} still exists at HEAD")
                broken = True
            elif symbol_defined(base, symbol, repo):
                removed_in_range = True
            elif not symbol_ever_defined(symbol, repo):
                errors.append(f"{waiver.file}: {symbol} never existed")
                broken = True
        if broken:
            continue
        if not removed_in_range:
            # Every symbol was removed before this range: the waiver covered
            # an earlier change and waives nothing here.
            continue
        loss = losses.get(waiver.file)
        if loss is None:
            errors.append(f"{waiver.file}: waiver is unused, the file lost nothing")
            continue
        net = loss["removed"] - loss["moved"] - loss["added"]
        if net > waiver.max_net_loss:
            errors.append(
                f"{waiver.file}: net loss {net} exceeds the waived {waiver.max_net_loss}"
            )
            continue
        remaining.pop(waiver.file)
    return remaining, errors


def test_no_net_assertion_loss_in_changed_tests():
    base = os.environ.get("ASSERTION_GUARD_BASE", "HEAD~1")
    offenders, waiver_errors = apply_waivers(
        net_assertion_losses(_git_diff(base)), load_waivers(), base
    )
    assert waiver_errors == [], f"assertion waivers fail their checks: {waiver_errors}"
    assert offenders == {}, (
        "these test files lost assertions; a fix may not reduce what a test "
        f"proves (base={base}): "
        + "; ".join(
            f"{p}: -{f['removed']} (moved {f['moved']}) +{f['added']}"
            for p, f in sorted(offenders.items())
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


_MOVE_SOURCE = (
    "diff --git a/tests/foo/test_x.py b/tests/foo/test_x.py\n"
    "--- a/tests/foo/test_x.py\n"
    "+++ b/tests/foo/test_x.py\n"
    "-    assert result == {'a': 1}\n"
    "-    assert order == ['a', 'b']\n"
)


def _created(path: str, *lines: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n" + "".join(f"+{line}\n" for line in lines)
    )


def test_assertions_moved_into_a_created_module_are_not_a_loss():
    diff = _MOVE_SOURCE + _created(
        "tests/foo/helpers.py",
        "def check(result, order):",
        "    assert result == {'a': 1}",
        "    assert order == ['a', 'b']",
    )
    assert moved_assertions(diff) == {"tests/foo/test_x.py": 2}
    assert net_assertion_losses(diff) == {}


def test_a_created_module_that_drops_a_moved_assertion_leaves_the_loss():
    diff = _MOVE_SOURCE + _created(
        "tests/foo/helpers.py",
        "def check(result, order):",
        "    assert result == {'a': 1}",
    )
    assert net_assertion_losses(diff) == {
        "tests/foo/test_x.py": {"removed": 2, "moved": 1, "added": 0}
    }


def test_a_different_assertion_in_a_created_module_does_not_offset_a_removal():
    diff = _MOVE_SOURCE + _created(
        "tests/foo/helpers.py",
        "    assert result == {'a': 2}",
        "    assert order",
    )
    assert net_assertion_losses(diff) == {
        "tests/foo/test_x.py": {"removed": 2, "moved": 0, "added": 0}
    }


def test_one_created_assertion_credits_one_removal():
    diff = (
        _MOVE_SOURCE + "diff --git a/tests/foo/test_y.py b/tests/foo/test_y.py\n"
        "--- a/tests/foo/test_y.py\n"
        "+++ b/tests/foo/test_y.py\n"
        "-    assert result == {'a': 1}\n"
        + _created(
            "tests/foo/helpers.py",
            "    assert result == {'a': 1}",
            "    assert order == ['a', 'b']",
        )
    )
    assert net_assertion_losses(diff) == {
        "tests/foo/test_y.py": {"removed": 1, "moved": 0, "added": 0}
    }


def test_a_created_file_outside_tests_credits_nothing():
    diff = _MOVE_SOURCE + _created(
        "libs/core/helpers.py",
        "    assert result == {'a': 1}",
        "    assert order == ['a', 'b']",
    )
    assert net_assertion_losses(diff) == {
        "tests/foo/test_x.py": {"removed": 2, "moved": 0, "added": 0}
    }


def _waiver_repo(tmp_path: Path) -> tuple[Path, str]:
    """A branch that deletes ``Gone`` and the two assertions that tested it."""
    repo = tmp_path / "waivers"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    _run_git(["init", "-q", "-b", "main", str(repo)], tmp_path)
    _run_git(["config", "user.email", "guard@example.invalid"], repo)
    _run_git(["config", "user.name", "Guard"], repo)
    (repo / "src" / "lib.py").write_text(
        "class Gone:\n    pass\n\n\ndef kept():\n    return 1\n"
    )
    (repo / "tests" / "test_lib.py").write_text(
        "def test_lib():\n"
        "    assert kept() == 1\n"
        "    assert Gone().x == 1\n"
        "    assert Gone().y == 2\n"
    )
    (repo / "tests" / "test_other.py").write_text(
        "def test_other():\n    assert kept() == 1\n"
    )
    _run_git(["add", "."], repo)
    _run_git(["commit", "-q", "-m", "Seed"], repo)
    base = _run_git(["rev-parse", "HEAD"], repo).strip()
    (repo / "src" / "lib.py").write_text("def kept():\n    return 1\n")
    (repo / "tests" / "test_lib.py").write_text(
        "def test_lib():\n    assert kept() == 1\n"
    )
    _run_git(["commit", "-q", "-am", "Remove Gone"], repo)
    return repo, base


def _waived(repo: Path, base: str, *waivers: Waiver):
    losses = net_assertion_losses(_git_diff(base, repo))
    return apply_waivers(losses, list(waivers), base, repo)


def test_a_waiver_for_removed_code_accepts_the_loss(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    assert _waived(
        repo, base, Waiver("tests/test_lib.py", 2, ("src/lib.py:Gone",), "removed")
    ) == ({}, [])


def test_a_waived_symbol_that_still_exists_fails(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    remaining, errors = _waived(
        repo, base, Waiver("tests/test_lib.py", 2, ("src/lib.py:kept",), "removed")
    )

    assert remaining == {"tests/test_lib.py": {"removed": 2, "moved": 0, "added": 0}}
    assert errors == ["tests/test_lib.py: src/lib.py:kept still exists at HEAD"]


def test_a_waived_symbol_that_never_existed_fails(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    remaining, errors = _waived(
        repo,
        base,
        Waiver(
            "tests/test_lib.py",
            2,
            ("src/lib.py:Gone", "src/lib.py:Never"),
            "removed",
        ),
    )

    assert remaining == {"tests/test_lib.py": {"removed": 2, "moved": 0, "added": 0}}
    assert errors == ["tests/test_lib.py: src/lib.py:Never never existed"]


def test_a_waiver_settled_before_the_range_waives_nothing(tmp_path):
    repo, _ = _waiver_repo(tmp_path)
    (repo / "tests" / "test_other.py").write_text("def test_other():\n    pass\n")
    _run_git(["commit", "-q", "-am", "Weaken another file"], repo)
    base = _run_git(["rev-parse", "HEAD~1"], repo).strip()

    assert _waived(
        repo, base, Waiver("tests/test_other.py", 1, ("src/lib.py:Gone",), "removed")
    ) == ({"tests/test_other.py": {"removed": 1, "moved": 0, "added": 0}}, [])


def test_a_loss_over_the_waived_count_fails(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    remaining, errors = _waived(
        repo, base, Waiver("tests/test_lib.py", 1, ("src/lib.py:Gone",), "removed")
    )

    assert remaining == {"tests/test_lib.py": {"removed": 2, "moved": 0, "added": 0}}
    assert errors == ["tests/test_lib.py: net loss 2 exceeds the waived 1"]


def test_an_unused_waiver_fails(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    remaining, errors = _waived(
        repo,
        base,
        Waiver("tests/test_lib.py", 2, ("src/lib.py:Gone",), "removed"),
        Waiver("tests/test_other.py", 1, ("src/lib.py:Gone",), "removed"),
    )

    assert remaining == {}
    assert errors == ["tests/test_other.py: waiver is unused, the file lost nothing"]


def test_an_unwaived_loss_is_still_reported(tmp_path):
    repo, base = _waiver_repo(tmp_path)

    assert _waived(repo, base) == (
        {"tests/test_lib.py": {"removed": 2, "moved": 0, "added": 0}},
        [],
    )


def test_a_malformed_waiver_file_is_rejected(tmp_path):
    path = tmp_path / "waivers.toml"
    path.write_text('[[waiver]]\nfile = "tests/x.py"\nmax_net_loss = 1\n')

    with pytest.raises(ValueError, match="malformed assertion waiver"):
        load_waivers(path)
