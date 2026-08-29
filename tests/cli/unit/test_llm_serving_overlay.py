"""Where the chat LLMs are served is independent of the local GPU vendor."""

from __future__ import annotations

import pytest
from cogniverse_cli.config import (
    LLM_SERVING_LOCAL,
    LLM_SERVING_MODAL,
    get_llm_serving_values_file,
)


def test_local_serving_composes_no_extra_overlay():
    assert get_llm_serving_values_file(LLM_SERVING_LOCAL) is None


def test_modal_serving_composes_the_backend_agnostic_overlay():
    path = get_llm_serving_values_file(LLM_SERVING_MODAL)

    assert path is not None
    assert path.name == "values.modal-llm.yaml"
    assert path.is_file()


def test_unknown_serving_mode_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        get_llm_serving_values_file("h100")

    assert "h100" in str(excinfo.value)
    assert LLM_SERVING_MODAL in str(excinfo.value)


def test_the_overlay_is_not_a_device_overlay():
    """A device overlay is selected by torch backend; this one must not be,
    or a rocm host would silently inherit Modal serving."""
    from cogniverse_cli.config import get_device_values_file

    for backend in ("rocm", "cuda", "cpu", "mps"):
        device = get_device_values_file(backend)
        assert device is None or device.name != "values.modal-llm.yaml"
