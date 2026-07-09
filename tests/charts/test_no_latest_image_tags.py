"""Shipped chart and workflow images must be pinned, not floating on :latest.

An upstream :latest push silently changes bucket-init / backup / provisioning
utility images on the next restart. This guards against :latest creeping back
into first-party manifests. The external OpenShell sandbox base image is the
only allowed exception (its tags aren't under this repo's control).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [REPO_ROOT / "charts", REPO_ROOT / "workflows"]
ALLOWED = {"ghcr.io/nvidia/openshell-community/sandboxes/base:latest"}

_IMAGE_LATEST = re.compile(r"image:\s*\"?([^\s\"]+:latest)\"?")
_SANDBOX_LATEST = re.compile(r":\s*\"?([\w./-]+/sandboxes/base:latest)\"?")


def _latest_hits():
    hits = []
    for base in SCAN_DIRS:
        for path in base.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            for m in _IMAGE_LATEST.finditer(text):
                if m.group(1) not in ALLOWED:
                    hits.append(f"{path.relative_to(REPO_ROOT)}: {m.group(1)}")
            for m in _SANDBOX_LATEST.finditer(text):
                if m.group(1) not in ALLOWED:
                    hits.append(f"{path.relative_to(REPO_ROOT)}: {m.group(1)}")
    return hits


@pytest.mark.unit
def test_no_floating_latest_image_tags():
    hits = _latest_hits()
    assert not hits, "unpinned :latest image tags:\n" + "\n".join(hits)
