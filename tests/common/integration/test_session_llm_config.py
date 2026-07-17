"""The session LM config every integration test inherits must be fully live.

``cogniverse_test_config`` (tests/conftest.py) hands the whole suite a
materialised config clone via ``COGNIVERSE_CONFIG``. Real DSPy compiles read
BOTH ``llm_config.primary`` (student) and ``llm_config.teacher``
(``BootstrapFewShot(teacher_settings={"lm": ...})`` demo generation) from it,
so every endpoint present in that config must answer ``/v1/models`` and serve
its configured model. A half-live config — live student, dead teacher — fails
every real compile with a connection error deep inside litellm instead of a
pointed diagnostic.
"""

from __future__ import annotations

import httpx
import pytest

from tests.fixtures.llm import is_test_lm_available
from tests.utils.llm_config import _load_config

pytestmark = pytest.mark.integration

skip_if_no_lm = pytest.mark.skipif(
    not is_test_lm_available(),
    reason="No test LM provisioned for this session",
)


def _bare_model(model: str) -> str:
    return model[len("openai/") :] if model.startswith("openai/") else model


def _served_model_ids(api_base: str) -> set[str]:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        resp = httpx.get(f"{base}/v1/models", timeout=10.0)
    except httpx.HTTPError as exc:
        pytest.fail(f"LM endpoint {base} unreachable: {exc!r}")
    assert resp.status_code == 200, (
        f"GET {base}/v1/models returned HTTP {resp.status_code}"
    )
    return {row.get("id", "") for row in resp.json().get("data") or []}


@skip_if_no_lm
@pytest.mark.parametrize("role", ["primary", "teacher"])
def test_session_llm_endpoint_serves_its_configured_model(role):
    llm_config = _load_config().get("llm_config")
    assert llm_config, "materialised session config has no llm_config section"
    endpoint = llm_config.get(role)
    assert endpoint, f"llm_config.{role} missing from the session config"

    api_base = endpoint.get("api_base")
    model = endpoint.get("model")
    assert api_base, f"llm_config.{role}.api_base missing from the session config"
    assert model, f"llm_config.{role}.model missing from the session config"

    served = _served_model_ids(api_base)
    assert _bare_model(model) in served, (
        f"llm_config.{role} model {_bare_model(model)!r} not served at "
        f"{api_base} (available: {sorted(served)})"
    )
