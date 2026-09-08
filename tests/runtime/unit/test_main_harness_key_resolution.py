"""main.py resolves the harness bearer keys from the process env.

``configs/config.json`` ships ``harness.api_keys`` with a ``"$VAR"`` key so no
credential lives in the file; the entrypoint turns it into the literal
key -> tenant map the /v1 router authenticates against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cogniverse_runtime.main import resolve_harness_api_keys

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "config.json"


def _shipped_harness_api_keys() -> dict[str, str]:
    raw = SHIPPED_CONFIG.read_text(encoding="utf-8")
    return json.loads(re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw))[
        "harness"
    ]["api_keys"]


def test_env_reference_resolves_to_the_literal_key():
    resolved = resolve_harness_api_keys(
        {"$COGNIVERSE_HARNESS_API_KEY": "default"},
        {"COGNIVERSE_HARNESS_API_KEY": "cgv-live-secret"},
    )

    assert resolved == {"cgv-live-secret": "default"}


def test_unset_and_empty_variables_drop_their_entries():
    resolved = resolve_harness_api_keys(
        {"$COGNIVERSE_HARNESS_API_KEY": "default", "$BLANK": "other"},
        {"BLANK": ""},
    )

    assert resolved == {}


def test_literal_keys_pass_through_and_blank_keys_drop():
    resolved = resolve_harness_api_keys({"cgv-static": "acme:prod", "": "unnamed"}, {})

    assert resolved == {"cgv-static": "acme:prod"}


def test_absent_block_yields_an_empty_map():
    assert resolve_harness_api_keys(None, {"COGNIVERSE_HARNESS_API_KEY": "x"}) == {}
    assert resolve_harness_api_keys({}, {}) == {}


def test_the_shipped_block_resolves_through_its_own_variable():
    shipped = _shipped_harness_api_keys()
    referenced = [key for key in shipped if key.startswith("$")]

    assert referenced == ["$COGNIVERSE_HARNESS_API_KEY"]
    assert resolve_harness_api_keys(
        shipped, {name[1:]: "cgv-from-env" for name in referenced}
    ) == {"cgv-from-env": shipped["$COGNIVERSE_HARNESS_API_KEY"]}
    assert resolve_harness_api_keys(shipped, {}) == {}
