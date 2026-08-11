"""Unconfigured clap_embed must fail naming the service, not ImportError.

With no ``clap_endpoint_url`` the generator falls back to loading CLAP
in-process. In the deployed runtime image torch is absent, so that path
raised ``ImportError: ClapModel requires the PyTorch library`` four frames
below the caller and the gateway returned an opaque 500.
"""

import importlib
import sys

import pytest

from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)


class _BlockImport:
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


def test_acoustic_text_embedding_without_sidecar_names_clap_embed(block_torch):
    generator = AudioEmbeddingGenerator()

    with pytest.raises(InferenceServiceUnavailableError) as excinfo:
        generator.generate_acoustic_text_embedding("podcasts about deep learning")

    exc = excinfo.value
    assert exc.service == "clap_embed"
    assert exc.module == "torch"
    assert "INFERENCE_SERVICE_URLS['clap_embed']" in str(exc)


def test_acoustic_audio_embedding_without_sidecar_names_clap_embed(
    block_torch, tmp_path
):
    generator = AudioEmbeddingGenerator()
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    with pytest.raises(InferenceServiceUnavailableError) as excinfo:
        generator.generate_acoustic_embedding(audio_path=clip)

    assert excinfo.value.service == "clap_embed"


def test_configured_sidecar_never_touches_the_in_process_backend(block_torch):
    """A configured endpoint must not consult torch at all: the remote path
    reaches HTTP and fails on the dead port, not on the missing backend."""
    generator = AudioEmbeddingGenerator(clap_endpoint_url="http://127.0.0.1:9")

    with pytest.raises(RuntimeError) as excinfo:
        generator.generate_acoustic_text_embedding("query")

    assert not isinstance(excinfo.value, InferenceServiceUnavailableError)
    assert "CLAP request to http://127.0.0.1:9/embed/text failed" in str(excinfo.value)
