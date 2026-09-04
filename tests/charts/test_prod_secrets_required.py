"""Production installs must not inherit the repo's dev secret defaults.

The MinIO root password and the OpenShell HMAC handshake secret shipped as
committed dev defaults and values.prod.yaml did not override them. They are now
emptied in the prod values and guarded by Helm `required`, so a prod install
fails loud until the operator supplies a real value.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "charts" / "cogniverse"
PROD_VALUES = CHART / "values.prod.yaml"
K3S_VALUES = CHART / "values.k3s.yaml"
MINIO_TMPL = CHART / "templates" / "minio.yaml"


def _prod():
    return yaml.safe_load(PROD_VALUES.read_text())


@pytest.mark.unit
def test_prod_values_empty_the_dev_secrets():
    prod = _prod()
    assert prod["minio"]["rootPassword"] == "", "prod must not ship a minio password"
    assert prod["openshell"]["server"]["sshHandshakeSecret"] == "", (
        "prod must not ship the openshell handshake secret"
    )
    assert prod["phoenix"]["postgres"]["auth"]["password"] == "", (
        "prod must not ship the phoenix postgres password"
    )


@pytest.mark.unit
def test_minio_template_requires_password():
    assert "required" in MINIO_TMPL.read_text()
    assert ".Values.minio.rootPassword" in MINIO_TMPL.read_text()


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_prod_install_fails_loud_without_minio_password():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(PROD_VALUES),
            # Prod also requires redis and postgres passwords; give them so
            # this isolates the minio required check.
            "--set",
            "redis.auth.password=x",
            "--set",
            "phoenix.postgres.auth.password=x",
            "--show-only",
            "templates/minio.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "minio.rootPassword must be set" in result.stderr


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_dev_install_still_renders_minio_secret():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(K3S_VALUES),
            "--show-only",
            "templates/minio.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rootPassword:" in result.stdout


POSTGRES_TMPL = CHART / "templates" / "phoenix-postgres.yaml"


@pytest.mark.unit
def test_postgres_template_requires_password():
    assert "required" in POSTGRES_TMPL.read_text()
    assert ".Values.phoenix.postgres.auth.password" in POSTGRES_TMPL.read_text()


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_prod_install_fails_loud_without_postgres_password():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(PROD_VALUES),
            # Isolate the postgres required check from the other prod guards.
            "--set",
            "redis.auth.password=x",
            "--set",
            "minio.rootPassword=x",
            "--show-only",
            "templates/phoenix-postgres.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "phoenix.postgres.auth.password must be set" in result.stderr


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_dev_install_still_renders_postgres_secret():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(K3S_VALUES),
            "--show-only",
            "templates/phoenix-postgres.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "password:" in result.stdout
