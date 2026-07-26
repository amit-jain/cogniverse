"""Foundation package metadata covers its direct runtime imports."""

import tomllib
from pathlib import Path


def test_fastapi_is_declared_as_direct_dependency():
    project_file = Path(__file__).parents[3] / "libs/foundation/pyproject.toml"
    project = tomllib.loads(project_file.read_text())

    fastapi_requirements = [
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("fastapi")
    ]

    assert fastapi_requirements == ["fastapi==0.135.3"]
