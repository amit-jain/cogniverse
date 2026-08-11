"""Fault contract for inference services with an in-process fallback.

A service whose remote sidecar is unconfigured falls back to loading the
model in-process. The deployed runtime image ships no torch, so that
fallback cannot run there; it must fail naming the service rather than
surfacing a transformers ImportError from inside the model loader.
"""

import importlib
import sys

import pytest

from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
    require_in_process_backend,
)


class _BlockImport:
    """Meta-path finder that makes one module tree genuinely unimportable."""

    def __init__(self, blocked: str):
        self._blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._blocked or fullname.startswith(self._blocked + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def block_torch():
    finder = _BlockImport("torch")
    saved = {
        name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "torch"
    }
    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)
        importlib.invalidate_caches()


def test_missing_backend_raises_naming_service_and_module(block_torch):
    with pytest.raises(InferenceServiceUnavailableError) as excinfo:
        require_in_process_backend("clap_embed", module="torch")

    exc = excinfo.value
    assert exc.service == "clap_embed"
    assert exc.module == "torch"
    assert str(exc) == (
        "clap_embed inference service is not configured and its in-process "
        "backend is unavailable in this image (no module named 'torch'). "
        "Set INFERENCE_SERVICE_URLS['clap_embed'] to the clap_embed sidecar "
        "URL, or install 'torch' to run clap_embed in-process."
    )


def test_error_is_a_runtime_error_so_callers_do_not_special_case_it():
    assert issubclass(InferenceServiceUnavailableError, RuntimeError)


def test_present_backend_is_a_no_op():
    assert require_in_process_backend("clap_embed", module="importlib") is None


def test_face_embed_message_names_face_embed(block_torch):
    with pytest.raises(InferenceServiceUnavailableError) as excinfo:
        require_in_process_backend("face_embed", module="torch")

    assert excinfo.value.service == "face_embed"
    assert "face_embed inference service is not configured" in str(excinfo.value)
    assert "INFERENCE_SERVICE_URLS['face_embed']" in str(excinfo.value)
