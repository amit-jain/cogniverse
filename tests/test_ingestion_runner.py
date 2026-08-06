"""Behavioral coverage for the ingestion test runner and its marker config."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_RUNNER = _REPO_ROOT / "scripts" / "test_ingestion.py"
_INGESTION_PYTEST_INI = _REPO_ROOT / "tests" / "ingestion" / "pytest.ini"


def _load_runner():
    spec = importlib.util.spec_from_file_location("test_ingestion_runner", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_command_builder_does_not_read_model_specific_marker_fields():
    runner = _load_runner()
    args = argparse.Namespace(
        unit=True,
        integration=False,
        ci_safe=False,
        local_only=False,
        requires_vespa=False,
        exclude_heavy=False,
        include_heavy=True,
        coverage_fail_under=80,
        verbose=False,
    )

    try:
        command = runner.build_pytest_command(args)
    except AttributeError as exc:
        pytest.fail(f"command builder read an obsolete marker field: {exc}")

    assert command == [
        "uv",
        "run",
        "python",
        "-m",
        "pytest",
        "tests/ingestion/unit/",
        "-m",
        "unit",
        "-v",
        "--tb=short",
        "--cov=src/app/ingestion/processors",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
    ]


def test_ingestion_pytest_config_registers_only_exact_inference_marker():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_INGESTION_PYTEST_INI),
            "--markers",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    inference_markers = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("@pytest.mark.requires_inference")
    ]
    assert inference_markers == [
        "@pytest.mark.requires_inference(service): require one exact named "
        "inference service"
    ]
