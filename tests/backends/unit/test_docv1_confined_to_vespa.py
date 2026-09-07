"""Raw Vespa ``document/v1`` URL construction lives ONLY in the vespa package.

Three components (wiki_manager, graph_manager, the ingestion router) used to
hand-build ``/document/v1/...`` HTTP URLs, bypassing the backend abstraction's
session reuse, error contracts, and namespace handling. They now route through
the backend document API. This guard keeps it that way: a new raw URL anywhere
outside ``libs/vespa`` (the abstraction itself) fails here.

Prose mentions (``/document/v1 URL`` with a trailing space, in docstrings or
error text) are fine — only ``document/v1/<path>`` construction is flagged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# document/v1 followed by a path segment == URL construction. The descriptive
# mentions left in code read ``document/v1 URL`` (space), which never matches.
_CONSTRUCTION = re.compile(r"document/v1/")

# The vespa package IS the sanctioned Document v1 surface.
_ALLOWED_PREFIXES = ("libs/vespa/",)

# The startup readiness probe answers whether the data plane is up before any
# backend — and so any deployed schema — can exist, which is why it cannot go
# through the backend document API. It is allowed by the exact URL it builds
# rather than by the module it sits in, so a different raw URL in that same
# module still fails here.
_ALLOWED_CONSTRUCTIONS = ("document/v1/config_metadata/config_metadata/docid/probe",)

_PROBE_MODULE = "libs/runtime/cogniverse_runtime/backend_startup.py"


def _offenders(sources) -> list[str]:
    """Raw constructions in ``sources``, an iterable of ``(label, text)``."""
    offenders = []
    for label, text in sources:
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore inline comments
            if not _CONSTRUCTION.search(code):
                continue
            if any(allowed in code for allowed in _ALLOWED_CONSTRUCTIONS):
                continue
            offenders.append(f"{label}:{lineno}: {line.strip()}")
    return offenders


def _repo_sources():
    for root in (_REPO_ROOT / "libs", _REPO_ROOT / "scripts"):
        for py in root.rglob("*.py"):
            rel = py.relative_to(_REPO_ROOT).as_posix()
            if rel.startswith(_ALLOWED_PREFIXES):
                continue
            yield rel, py.read_text()


def test_no_raw_document_v1_construction_outside_vespa_package():
    assert _offenders(_repo_sources()) == [], (
        "Raw Vespa document/v1 URL construction must go through the backend "
        "document API (VespaBackend.put/get/update/delete_document_fields), not "
        "hand-built HTTP."
    )


def test_detector_flags_a_second_raw_url_inside_the_probe_module():
    """The allowance is the probe URL, not the module that holds it."""
    probe_line = f'    url = f"{{base}}/{_ALLOWED_CONSTRUCTIONS[0]}"'
    other_line = '    other = f"{base}/document/v1/tenant_metadata/x/docid/1"'
    source = "\n".join([probe_line, other_line])
    assert _offenders([(_PROBE_MODULE, source)]) == [
        f"{_PROBE_MODULE}:2: {other_line.strip()}"
    ]


def test_detector_ignores_prose_and_commented_mentions():
    source = "\n".join(
        [
            '    """Reads a document/v1 URL."""',
            '    # legacy: f"{base}/document/v1/wiki/wiki/docid/1"',
        ]
    )
    assert _offenders([("libs/runtime/x.py", source)]) == []


def test_every_allowed_construction_is_a_live_production_line():
    """A retired exemption must not linger: main.py's did, and hid this move."""
    probe = (_REPO_ROOT / _PROBE_MODULE).read_text()
    assert [c for c in _ALLOWED_CONSTRUCTIONS if c in probe] == list(
        _ALLOWED_CONSTRUCTIONS
    )
