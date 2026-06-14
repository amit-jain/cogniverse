"""Unit tests for ModelLoader implementations in cogniverse_core."""

import builtins

import pytest

from cogniverse_core.common.models.model_loaders import (
    ColBERTModelLoader,
    ColPaliModelLoader,
    ColQwenModelLoader,
    ModelLoaderFactory,
)

_REMOTE_ONLY_SUBSTRINGS = (
    "ColQwen3/Tomoro models are remote-only",
    "inference_service_url",
    "transformers>=4.57",
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


@pytest.mark.unit
class TestColQwen3RemoteOnlyGuard:
    """ColQwen3/Tomoro (model_type ``qwen3_vl``) has no local in-process
    loader: the pinned transformers (4.56.2, capped by pylate) lacks
    ``qwen3_vl`` support and colpali_engine mis-maps it to ``idefics3``,
    so a local load crashes with a bare ``KeyError: 'qwen3_vl_text'``.
    The loaders must turn that into a clear remote-only RuntimeError that
    tells the operator to serve via vLLM and set ``inference_service_url``.
    """

    @pytest.fixture(autouse=True)
    def _instant_retries(self, monkeypatch):
        monkeypatch.setattr(
            "cogniverse_core.common.utils.retry.time.sleep", lambda *a, **k: None
        )

    @pytest.mark.parametrize(
        "loader_cls,model_loader",
        [(ColPaliModelLoader, "colpali"), (ColQwenModelLoader, "colqwen")],
    )
    def test_proactive_name_match_raises_remote_only(self, loader_cls, model_loader):
        loader = loader_cls(
            model_name="TomoroAI/tomoro-colqwen3-embed-4b",
            config={"device": "cpu", "model_loader": model_loader},
        )
        with pytest.raises(RuntimeError) as excinfo:
            loader.load_model()

        msg = str(excinfo.value)
        for substr in _REMOTE_ONLY_SUBSTRINGS:
            assert substr in msg
        # The proactive guard must NOT surface the bare arch KeyError.
        assert "KeyError" not in msg

    def test_qwen3_vl_keyerror_is_wrapped(self, monkeypatch):
        """A non-name-detectable load that still fails with the qwen3_vl
        arch signature is re-raised as the clear remote-only error rather
        than the bare ``KeyError: 'qwen3_vl_text'``."""
        import colpali_engine.models as cem

        class _Boom:
            @staticmethod
            def from_pretrained(*a, **k):
                raise KeyError("qwen3_vl_text")

        monkeypatch.setattr(cem, "ColIdefics3", _Boom, raising=False)

        # A name colpali_engine accepts as ColPali but whose weights are
        # actually qwen3_vl — bypasses the proactive name guard, hits the load.
        loader = ColPaliModelLoader(
            model_name="vidore/colpali-v1.3",
            config={"device": "cpu", "model_loader": "colpali"},
        )
        with pytest.raises(RuntimeError) as excinfo:
            loader.load_model()

        msg = str(excinfo.value)
        assert "remote-only" in msg
        assert "qwen3_vl_text" not in msg

    def test_supported_colpali_load_failure_not_masked(self, monkeypatch):
        """A genuine (non-qwen3_vl) load failure for a supported model must
        propagate as-is, not be swallowed into the remote-only message."""
        import colpali_engine.models as cem

        class _Boom:
            @staticmethod
            def from_pretrained(*a, **k):
                raise OSError("connection reset while downloading weights")

        monkeypatch.setattr(cem, "ColIdefics3", _Boom, raising=False)

        loader = ColPaliModelLoader(
            model_name="vidore/colpali-v1.3",
            config={"device": "cpu", "model_loader": "colpali"},
        )
        with pytest.raises(OSError) as excinfo:
            loader.load_model()
        assert "connection reset" in str(excinfo.value)

    @pytest.mark.parametrize(
        "encoder_factory",
        ["ColPaliQueryEncoder", "ColQwenQueryEncoder"],
    )
    def test_local_query_encoder_for_tomoro_raises_remote_only(self, encoder_factory):
        """Constructing the local query encoder (no inference_service_url)
        for a Tomoro model surfaces the remote-only RuntimeError."""
        from cogniverse_core.query import encoders

        factory = getattr(encoders, encoder_factory)
        with pytest.raises(RuntimeError) as excinfo:
            factory("TomoroAI/tomoro-colqwen3-embed-4b")

        msg = str(excinfo.value)
        for substr in _REMOTE_ONLY_SUBSTRINGS:
            assert substr in msg
