"""Every path that installs uv must install the one version pyproject requires.

A floating uv (``version: "latest"``, the unversioned installer script, a bare
``pip install uv``, ``ghcr.io/astral-sh/uv:latest``) lets CI, the images and a
developer's machine run different uv releases against the same lock, and a
new release that reformats its output breaks tests on a commit that changed
nothing. The workspace pyproject's ``[tool.uv] required-version`` is the single
source; every install path must name exactly that version.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO / ".github" / "workflows"
# Trees holding Dockerfiles, deploy manifests and install scripts.
SCANNED_TREES = ("charts", "deploy", "libs", "scripts")
SCANNED_SUFFIXES = (".sh", ".yml", ".yaml", ".tpl")

_INSTALLER = re.compile(
    r"astral\.sh/uv/(?:(?P<version>[^/\s\"']+)/)?install\.(?:sh|ps1)"
)
_IMAGE = re.compile(r"ghcr\.io/astral-sh/uv(?::(?P<tag>[^\s@\"'/]+))?")
_PIP_INSTALL = re.compile(r"\bpip3?\s+install\s+(?P<args>[^\n;&|]*)")
_UV_REQUIREMENT = re.compile(
    r"[\"']?uv(?:\[[^\]]*\])?(?P<spec>[=<>~!][^\s\"']*)?[\"']?"
)


def required_uv_version(pyproject_text: str) -> str | None:
    """The exact version ``[tool.uv] required-version`` pins, or None."""
    required = tomllib.loads(pyproject_text).get("tool", {}).get("uv", {})
    required = required.get("required-version")
    if not isinstance(required, str):
        return None
    match = re.fullmatch(r"==\s*(\d+\.\d+\.\d+)", required.strip())
    return match.group(1) if match else None


def setup_uv_violations(name: str, workflow: dict, version: str) -> list[str]:
    """``astral-sh/setup-uv`` steps whose ``with.version`` is not ``version``."""
    findings = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            uses = step.get("uses", "") if isinstance(step, dict) else ""
            if not uses.startswith("astral-sh/setup-uv@"):
                continue
            pinned = (step.get("with") or {}).get("version")
            if str(pinned) != version:
                findings.append(
                    f"{name} job {job_name} step {index} ({uses}): "
                    f"version={pinned!r}, required {version}"
                )
    return findings


def text_violations(name: str, text: str, version: str) -> list[str]:
    """Installer scripts, pip installs and uv images not pinned to ``version``."""
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        where = f"{name}:{number}"
        for match in _INSTALLER.finditer(line):
            if match.group("version") != version:
                findings.append(f"{where}: installer {match.group(0)!r}")
        for match in _IMAGE.finditer(line):
            if match.group("tag") != version:
                findings.append(f"{where}: image {match.group(0)!r}")
        for match in _PIP_INSTALL.finditer(line):
            for token in match.group("args").split():
                requirement = _UV_REQUIREMENT.fullmatch(token)
                if requirement and requirement.group("spec") != f"=={version}":
                    findings.append(f"{where}: pip install {token!r}")
    return findings


def _shipped_files() -> dict[str, str]:
    paths = sorted(WORKFLOW_DIR.glob("*.y*ml"))
    for tree in SCANNED_TREES:
        paths.extend(
            path
            for path in sorted((REPO / tree).rglob("*"))
            if path.is_file()
            and (path.name.startswith("Dockerfile") or path.suffix in SCANNED_SUFFIXES)
        )
    paths.append(REPO / "Makefile")
    return {
        str(path.relative_to(REPO)): path.read_text(encoding="utf-8") for path in paths
    }


def _shipped_version() -> str | None:
    return required_uv_version((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_requires_one_exact_uv_version():
    assert _shipped_version() is not None, (
        'pyproject.toml needs [tool.uv] required-version = "==X.Y.Z"'
    )


def test_every_uv_install_path_uses_the_required_version():
    version = _shipped_version()
    assert version is not None, "pyproject.toml pins no exact uv version"
    findings = []
    for name, text in _shipped_files().items():
        if name.startswith(".github/workflows/"):
            findings += setup_uv_violations(name, yaml.safe_load(text), version)
        findings += text_violations(name, text, version)
    assert findings == [], "\n".join(findings)


# Synthetic input keeps the detectors honest once the tree has no offenders.


def test_required_version_accepts_only_an_exact_pin():
    assert required_uv_version('[tool.uv]\nrequired-version = "==1.2.3"\n') == "1.2.3"
    assert required_uv_version('[tool.uv]\nrequired-version = ">=1.2"\n') is None
    assert required_uv_version("[tool.uv]\nprerelease = 'allow'\n") is None


def test_setup_uv_detector_reports_latest_missing_and_other_versions():
    workflow = {
        "jobs": {
            "a": {
                "steps": [
                    {"uses": "astral-sh/setup-uv@v4", "with": {"version": "latest"}},
                    {"uses": "astral-sh/setup-uv@v5", "with": {"enable-cache": True}},
                    {"uses": "astral-sh/setup-uv@v4", "with": {"version": "1.2.4"}},
                    {"uses": "astral-sh/setup-uv@v4", "with": {"version": "1.2.3"}},
                    {"run": "uv sync"},
                ]
            }
        }
    }
    findings = setup_uv_violations("w.yml", workflow, "1.2.3")
    assert [finding.split(":")[0] for finding in findings] == [
        "w.yml job a step 0 (astral-sh/setup-uv@v4)",
        "w.yml job a step 1 (astral-sh/setup-uv@v5)",
        "w.yml job a step 2 (astral-sh/setup-uv@v4)",
    ]


def test_text_detector_reports_each_floating_install():
    text = "\n".join(
        [
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "curl -LsSf https://astral.sh/uv/1.2.4/install.sh | sh",
            "curl -LsSf https://astral.sh/uv/1.2.3/install.sh | sh",
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv",
            "COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv",
            "COPY --from=ghcr.io/astral-sh/uv:1.2.3 /uv /usr/local/bin/uv",
            "pip install uv",
            "python -m pip install --upgrade 'uv>=1.2'",
            "pip install uv==1.2.3 uvicorn",
            'log "Install with: pip install uv"',
        ]
    )
    findings = text_violations("f", text, "1.2.3")
    assert [finding.split(": ")[0] for finding in findings] == [
        "f:1",
        "f:2",
        "f:4",
        "f:5",
        "f:7",
        "f:8",
        "f:10",
    ]
