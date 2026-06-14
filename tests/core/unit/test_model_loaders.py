"""Unit tests for ModelLoader implementations in cogniverse_core."""

import builtins

import pytest

from cogniverse_core.common.models.model_loaders import (
    ColBERTModelLoader,
    ModelLoaderFactory,
)


@pytest.mark.unit
class TestColBERTModelLoaderMissingPylate:
    """pylate is a [test]-only optional dependency. Production ColBERT is
    served via vLLM (RemoteColBERTLoader). A future local-colbert config
    would hit ColBERTModelLoader.load_model; if pylate is absent the user
    must get an actionable message, not a bare ModuleNotFoundError.
    """

    def test_loader_still_registered(self):
        # Never-delete rule: the loader stays in the factory registry.
        assert ModelLoaderFactory.LOADERS["colbert"] is ColBERTModelLoader

    def test_missing_pylate_raises_actionable_importerror(self, monkeypatch):
        # Make retries instant so the decorator's backoff doesn't slow the test.
        monkeypatch.setattr(
            "cogniverse_core.common.utils.retry.time.sleep", lambda *a, **k: None
        )

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pylate" or name.startswith("pylate."):
                raise ImportError("No module named 'pylate'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        loader = ColBERTModelLoader(
            model_name="lightonai/GTE-ModernColBERT-v1",
            config={"device": "cpu"},
        )

        with pytest.raises(ImportError) as excinfo:
            loader.load_model()

        msg = str(excinfo.value)
        assert "Local ColBERT loading requires the optional 'pylate'" in msg
        assert "inference_services.embedding" in msg
        assert "RemoteColBERTLoader" in msg
