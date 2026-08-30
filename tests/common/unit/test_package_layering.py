"""Every package an image installs imports only packages that image installs.

A package that imports a sibling outside its image's install set runs green
in the monorepo venv and fails at import inside the image; the runtime image
is built from the packages its Dockerfile copies and never ships
cogniverse-cli. Imports wrapped in ``try/except ImportError`` or under ``if TYPE_CHECKING``
are not executed in the image and are not counted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBS = REPO_ROOT / "libs"
_OPTIONAL_HANDLERS = {"ImportError", "ModuleNotFoundError", "Exception"}
# Packages an image may lack by contract: the import site documents that it
# raises ImportError and every caller handles that as feature-absent.
OPTIONAL_PACKAGES = {
    # adapter_loader._resolve_active_adapter raises ImportError without
    # cogniverse_finetuning; adapter_lm_context and load_adapter_for_agent
    # catch it and serve the base model.
    "cogniverse_finetuning",
}


def image_packages(dockerfile: Path) -> list[str]:
    """Import names of the workspace packages a Dockerfile copies in."""
    return [
        f"cogniverse_{name.replace('-', '_')}"
        for name in re.findall(r"^COPY libs/([a-z-]+) ", dockerfile.read_text(), re.M)
    ]


def _guarded_by_import_error(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    current = node
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, ast.If) and "TYPE_CHECKING" in ast.unparse(current.test):
            return True
        if isinstance(current, ast.Try) and any(
            handler.type is None
            or any(
                name in _OPTIONAL_HANDLERS
                for name in re.findall(r"[A-Za-z_]+", ast.unparse(handler.type))
            )
            for handler in current.handlers
        ):
            return True
    return False


def required_cogniverse_imports(source: Path) -> set[str]:
    """Top-level cogniverse packages ``source`` imports unconditionally."""
    tree = ast.parse(source.read_text(), filename=str(source))
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        if _guarded_by_import_error(node, parents):
            continue
        found.update(n.split(".")[0] for n in names if n.startswith("cogniverse_"))
    return found


def image_import_violations(libs: Path, dockerfile: Path) -> list[tuple[str, str]]:
    """(file, imported_package) for every unconditional import the image lacks."""
    installed = image_packages(dockerfile)
    violations: list[tuple[str, str]] = []
    for package in installed:
        source_dir = next(libs.glob(f"*/{package}"))
        for py in sorted(source_dir.rglob("*.py")):
            for imported in sorted(required_cogniverse_imports(py)):
                if imported not in installed and imported not in OPTIONAL_PACKAGES:
                    violations.append((str(py.relative_to(libs)), imported))
    return violations


def test_detector_names_an_import_the_image_lacks(tmp_path: Path) -> None:
    libs = tmp_path / "libs"
    for name in ("alpha", "beta"):
        (libs / name / f"cogniverse_{name}").mkdir(parents=True)
        (libs / name / f"cogniverse_{name}" / "__init__.py").write_text("")
    (libs / "alpha" / "cogniverse_alpha" / "uses.py").write_text(
        "def f():\n    from cogniverse_beta.thing import y\n    return y\n"
    )
    (libs / "alpha" / "cogniverse_alpha" / "optional.py").write_text(
        "try:\n    import cogniverse_beta\nexcept ImportError:\n    cogniverse_beta = None\n"
    )
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM x\nCOPY libs/alpha ./libs/alpha\n")
    assert image_packages(dockerfile) == ["cogniverse_alpha"]
    assert image_import_violations(libs, dockerfile) == [
        ("alpha/cogniverse_alpha/uses.py", "cogniverse_beta")
    ]


def test_runtime_image_installs_every_package_it_imports() -> None:
    assert image_import_violations(LIBS, LIBS / "runtime" / "Dockerfile") == []
