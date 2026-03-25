"""Path resolution for Cogniverse CLI.

Provides helpers that locate project assets (Helm charts, workflow
definitions, configuration files) in both *dev mode* (running from a
checkout of the monorepo) and *installed mode* (running from a pip/uv
installed wheel, where assets are bundled as package data).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def resolve_project_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a ``pyproject.toml`` that contains
    ``[tool.uv.workspace]``, indicating the monorepo root.

    Returns the directory containing that file, or ``None`` if the
    filesystem root is reached without finding one.
    """
    current = (start or Path.cwd()).resolve()
    while True:
        candidate = current / "pyproject.toml"
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            if "[tool.uv.workspace]" in text:
                return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _resolve_path(
    relative: str,
    package_data_subdir: str,
    project_root: Path | None = None,
) -> Path:
    """Shared resolver: prefer the monorepo checkout path, fall back to
    package data bundled inside the wheel.

    Raises ``FileNotFoundError`` when neither location exists.
    """
    root = project_root or resolve_project_root()
    if root is not None:
        dev_path = root / relative
        if dev_path.is_dir():
            return dev_path

    # Installed mode — look inside the package's ``data/`` tree.
    try:
        pkg_data = resources.files("cogniverse_cli") / "data" / package_data_subdir
        # resources.files returns a Traversable; materialise to a real Path.
        with resources.as_file(pkg_data) as p:
            if p.is_dir():
                return p
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    raise FileNotFoundError(
        f"Cannot locate '{relative}' in the project tree or in package data."
    )


def get_chart_path(project_root: Path | None = None) -> Path:
    """Return the path to the Helm chart directory.

    Dev mode: ``<root>/charts/cogniverse/``
    Installed mode: package data ``data/charts/cogniverse/``
    """
    return _resolve_path("charts/cogniverse", "charts/cogniverse", project_root)


def get_workflows_path(project_root: Path | None = None) -> Path:
    """Return the path to the Argo workflow definitions.

    Dev mode: ``<root>/workflows/``
    Installed mode: package data ``data/workflows/``
    """
    return _resolve_path("workflows", "workflows", project_root)


def get_configs_path(project_root: Path | None = None) -> Path:
    """Return the path to the shared configuration directory.

    Dev mode: ``<root>/configs/``
    Installed mode: package data ``data/configs/``
    """
    return _resolve_path("configs", "configs", project_root)


def get_values_file(
    project_root: Path | None = None,
    *,
    prod: bool = False,
) -> Path:
    """Return the path to the Helm values file.

    Returns ``values.prod.yaml`` when *prod* is ``True``, otherwise
    ``values.k3s.yaml``.  The file must exist inside the chart directory.
    """
    chart_dir = get_chart_path(project_root)
    filename = "values.prod.yaml" if prod else "values.k3s.yaml"
    values_file = chart_dir / filename
    if not values_file.is_file():
        raise FileNotFoundError(f"Values file not found: {values_file}")
    return values_file
