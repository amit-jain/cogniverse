"""Coverage for repo CLI scripts that previously had no tests.

``version_bump`` carries pure semantic-version logic exercised directly; the
Vespa/Phoenix management CLIs are service wrappers, so they get a ``--help``
smoke test that proves the entry point imports and its argparse is valid
without needing a live backend.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vb = _load("version_bump")


class TestVersionBump:
    def test_parse_valid(self):
        assert vb.parse_version("1.2.3") == (1, 2, 3, "")
        assert vb.parse_version("10.0.4-alpha.1") == (10, 0, 4, "-alpha.1")

    @pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "", "1.2.x"])
    def test_parse_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            vb.parse_version(bad)

    def test_format(self):
        assert vb.format_version(1, 2, 3) == "1.2.3"
        assert vb.format_version(1, 2, 3, "-rc.0") == "1.2.3-rc.0"

    def test_bump_major_minor_patch(self):
        assert vb.bump_version("1.2.3", "major") == "2.0.0"
        assert vb.bump_version("1.2.3", "minor") == "1.3.0"
        assert vb.bump_version("1.2.3", "patch") == "1.2.4"
        assert vb.bump_version("1.2.3-alpha.1", "patch") == "1.2.4"

    def test_bump_prerelease(self):
        assert vb.bump_version("1.2.3", "prerelease", "alpha") == "1.2.4-alpha.0"
        assert (
            vb.bump_version("1.2.4-alpha.0", "prerelease", "alpha") == "1.2.4-alpha.1"
        )
        assert vb.bump_version("1.2.4-alpha.1", "prerelease", "beta") == "1.2.4-beta.0"


_WRAPPER_CLIS = [
    "deploy_json_schema",
    "manage_datasets",
    "manage_golden_datasets",
    "manage_phoenix_data",
    "prune_config_metadata",
]


@pytest.mark.parametrize("script", _WRAPPER_CLIS)
def test_cli_help_loads(script):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / f"{script}.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script} --help failed: {proc.stderr}"
    assert "usage" in proc.stdout.lower()
