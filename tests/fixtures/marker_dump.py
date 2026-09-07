"""pytest plugin: dump each collected item's resolved marker names as JSON.

Loaded with ``-p tests.fixtures.marker_dump`` and pointed at a file through
``COGNIVERSE_MARKER_DUMP``. The dump runs after every
``pytest_collection_modifyitems`` hook, so location-derived markers
(``tests/fixtures/markers.py``) are already applied — the names recorded are
the ones ``-m`` matches against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DUMP_PATH_ENV = "COGNIVERSE_MARKER_DUMP"


def pytest_collection_finish(session) -> None:
    destination = os.environ.get(DUMP_PATH_ENV)
    if not destination:
        return
    marker_map = {
        item.nodeid: sorted({mark.name for mark in item.iter_markers()})
        for item in session.items
    }
    Path(destination).write_text(json.dumps(marker_map, indent=0, sort_keys=True))
