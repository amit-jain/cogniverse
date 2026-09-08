"""Rendered manifests must not repeat a key inside one mapping.

YAML resolves a duplicate by keeping the last occurrence, so a template that
emits a block twice still applies -- until the two copies diverge, at which
point the earlier one is silently discarded. kubeconform --strict rejects
the document outright, which is how this surfaced.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest
import yaml

CHART_DIR = pathlib.Path(__file__).resolve().parents[2] / "charts" / "cogniverse"
VALUE_PROFILES = ("values.yaml", "values.k3s.yaml", "values.rocm.yaml")


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key."""


def _no_duplicates(loader: _DuplicateKeyLoader, node: yaml.MappingNode) -> dict:
    seen: set[str] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def duplicate_key_documents(rendered: str) -> dict[str, str]:
    """Map ``Kind/name`` -> the duplicate-key message, for every bad document."""
    offenders: dict[str, str] = {}
    for chunk in rendered.split("\n---\n"):
        if not chunk.strip():
            continue
        try:
            yaml.load(chunk, Loader=_DuplicateKeyLoader)
        except yaml.constructor.ConstructorError as exc:
            loose = yaml.safe_load(chunk) or {}
            kind = loose.get("kind", "?")
            name = (loose.get("metadata") or {}).get("name", "?")
            offenders[f"{kind}/{name}"] = str(exc.problem)
    return offenders


def _render(profile: str) -> str:
    values = CHART_DIR / profile
    helm = shutil.which("helm")
    assert helm is not None, (
        "helm is required to render the chart; this guard has no meaning without it"
    )
    command = [helm, "template", "cogniverse", str(CHART_DIR), "-n", "cogniverse"]
    if profile != "values.yaml":
        command += ["-f", str(values)]
    command += [
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set",
        "argo-workflows.crds.install=false",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("profile", VALUE_PROFILES)
def test_no_rendered_document_repeats_a_key(profile):
    assert duplicate_key_documents(_render(profile)) == {}


# The rendered-chart assertion above cannot protect its own detector: once the
# last duplicate is removed, gutting duplicate_key_documents leaves it green.

_CLEAN = "kind: Deployment\nmetadata:\n  name: a\nspec:\n  replicas: 1\n"
_DUPLICATE = (
    "kind: Deployment\nmetadata:\n  name: b\nspec:\n  replicas: 1\n  replicas: 2\n"
)
_NESTED_DUPLICATE = (
    "kind: Deployment\nmetadata:\n  name: c\nspec:\n"
    "  template:\n    resources: {}\n    resources: {}\n"
)


@pytest.mark.parametrize(
    ("rendered", "expected"),
    [
        (_CLEAN, {}),
        (_DUPLICATE, {"Deployment/b": "duplicate key 'replicas'"}),
        (_NESTED_DUPLICATE, {"Deployment/c": "duplicate key 'resources'"}),
        (f"{_CLEAN}\n---\n{_DUPLICATE}", {"Deployment/b": "duplicate key 'replicas'"}),
    ],
    ids=["clean", "top-level", "nested", "one-of-two-documents"],
)
def test_detector_names_exactly_the_duplicate_documents(rendered, expected):
    assert duplicate_key_documents(rendered) == expected
