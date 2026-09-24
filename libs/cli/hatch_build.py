"""Bundle the deployment assets the CLI resolves under ``cogniverse_cli/data/``.

From a checkout, the assets are the files git tracks under each configured
asset root of the repository. From an unpacked sdist, which carries no
repository, they are the files the sdist already holds under
``cogniverse_cli/data/``. Either way every required asset must be present, or
the build fails instead of producing a package that fails after installation.
Files are mapped straight from their source, so builds stage nothing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_DATA_DIR = "cogniverse_cli/data"


class DeploymentAssetsHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if version == "editable":
            return
        root = Path(self.root)
        asset_roots = self.config["asset-roots"]
        if (root / "PKG-INFO").is_file():
            sources = _sdist_assets(root / _DATA_DIR, asset_roots)
        else:
            sources = _tracked_assets(root, asset_roots)
        missing = sorted(
            {
                name
                for name in (*self.config["required-assets"], *sources)
                if name not in sources or not sources[name].is_file()
            }
        )
        if missing:
            raise RuntimeError(
                "cogniverse-cli is missing required deployment assets: "
                + ", ".join(missing)
            )
        build_data["force_include"].update(
            {str(path): f"{_DATA_DIR}/{name}" for name, path in sources.items()}
        )


def _tracked_assets(project_root: Path, asset_roots: list[str]) -> dict[str, Path]:
    repository = Path(
        _git(project_root, "rev-parse", "--show-toplevel").decode().strip()
    )
    listed = _git(repository, "ls-files", "-z", "--cached", "--", *asset_roots)
    return {name: repository / name for name in listed.decode().split("\0") if name}


def _sdist_assets(data_dir: Path, asset_roots: list[str]) -> dict[str, Path]:
    assets = {}
    for asset_root in asset_roots:
        for directory, _, files in os.walk(data_dir / asset_root):
            for filename in files:
                path = Path(directory, filename)
                assets[path.relative_to(data_dir).as_posix()] = path
    return assets


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, check=True
    ).stdout
