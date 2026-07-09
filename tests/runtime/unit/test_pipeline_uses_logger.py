"""The ingestion pipeline must log through its logger, not print().

~24 print() calls (including failures) wrote to raw stdout, bypassing the
runtime's structured logging so a failed ingest was undiagnosable from logs.
This guards against print() creeping back into the module.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "libs/runtime/cogniverse_runtime/ingestion/pipeline.py"


def test_no_print_calls_in_pipeline():
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not prints, f"print() found at lines {prints}; use self.logger instead"
