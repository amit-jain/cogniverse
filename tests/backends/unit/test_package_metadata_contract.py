import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "libs" / "vespa"


def test_vespa_package_declares_every_direct_runtime_dependency():
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    dependencies = {
        dependency.split("==", 1)[0]
        for dependency in metadata["project"]["dependencies"]
    }

    assert dependencies == {
        "cogniverse-core",
        "cogniverse-foundation",
        "cogniverse-sdk",
        "numpy",
        "pydantic",
        "pyvespa",
        "requests",
    }
    assert metadata["tool"]["uv"]["sources"] == {
        "cogniverse-core": {"workspace": True},
        "cogniverse-foundation": {"workspace": True},
        "cogniverse-sdk": {"workspace": True},
    }
