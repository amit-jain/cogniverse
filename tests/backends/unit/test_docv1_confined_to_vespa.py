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

# The vespa package IS the sanctioned Document v1 surface; main.py's readiness
# probe is a deliberate low-level GET documented as the one allowed exception.
_ALLOWED_PREFIXES = ("libs/vespa/",)
_ALLOWED_FILES = {"libs/runtime/cogniverse_runtime/main.py"}


def test_no_raw_document_v1_construction_outside_vespa_package():
    offenders = []
    for py in (_REPO_ROOT / "libs").rglob("*.py"):
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith(_ALLOWED_PREFIXES) or rel in _ALLOWED_FILES:
            continue
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore inline comments
            if _CONSTRUCTION.search(code):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Raw Vespa document/v1 URL construction must go through the backend "
        "document API (VespaBackend.put/get/update/delete_document_fields), not "
        "hand-built HTTP. Offenders:\n" + "\n".join(offenders)
    )
