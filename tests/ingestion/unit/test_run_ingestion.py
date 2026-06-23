"""Coverage for the run_ingestion CLI script.

``scripts/run_ingestion.py`` is the documented primary ingestion CLI but had
zero tests. These exercise the content-type dispatch (``discover_content_files``)
and the early config guard that turns a missing ``BACKEND_URL`` into a clean
exit-2 message instead of a deep traceback from inside the bootstrap.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "run_ingestion.py"
_spec = importlib.util.spec_from_file_location("run_ingestion", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
discover_content_files = _mod.discover_content_files


def test_video_dispatch_returns_only_video_files(tmp_path):
    for name in ("a.mp4", "b.avi", "c.txt", "d.png"):
        (tmp_path / name).touch()
    assert discover_content_files(tmp_path, "video") == [
        tmp_path / "a.mp4",
        tmp_path / "b.avi",
    ]


@pytest.mark.parametrize("content_type", ["image", "audio", "document"])
def test_batch_dispatch_returns_the_directory(tmp_path, content_type):
    assert discover_content_files(tmp_path, content_type) == [tmp_path]


def test_cli_missing_backend_url_exits_cleanly(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BACKEND_URL"}
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--content-dir",
            str(tmp_path),
            "--content-type",
            "video",
            "--backend",
            "vespa",
            "--tenant-id",
            "t",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "Error: BACKEND_URL" in proc.stdout
    assert "Traceback" not in proc.stderr
