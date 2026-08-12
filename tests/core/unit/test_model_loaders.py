"""Unit tests for ModelLoader implementations in cogniverse_core."""

import builtins

import numpy as np
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
@pytest.mark.ci_fast
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


@pytest.mark.unit
@pytest.mark.ci_fast
class TestProcessImagesVllmConcurrent:
    """The vLLM pooling client posts one request per image concurrently; the
    returned embeddings must line up with the input image order regardless of
    completion order."""

    def _client_with_recorded_posts(self, images):
        import base64
        import io
        import threading
        from unittest.mock import MagicMock

        from cogniverse_core.common.models.model_loaders import (
            RemoteInferenceClient,
        )

        client = RemoteInferenceClient(endpoint_url="http://unused:1")

        # Map each image's PNG b64 (exactly as the client encodes it) to its
        # input position so the fake response is request-derived — immune to
        # thread completion order.
        b64_to_index = {}
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_to_index[base64.b64encode(buf.getvalue()).decode("utf-8")] = i

        threads = set()
        # Block every request until all are in flight so the pool must use one
        # worker per image; without this a fast fake lets one thread serve the
        # whole batch and the fan-out count becomes racy.
        barrier = threading.Barrier(len(images))

        def fake_post(url, json=None, timeout=None):
            threads.add(threading.current_thread().name)
            barrier.wait(timeout=5)
            b64 = json["messages"][0]["content"][0]["image_url"]["url"].split(
                "base64,", 1
            )[1]
            idx = b64_to_index[b64]
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "data": [{"data": [[float(idx)] * 4]}],
                "model": "m",
                "usage": {},
            }
            return resp

        client.session = MagicMock()
        client.session.post.side_effect = fake_post
        return client, threads

    def test_batch_results_preserve_input_order(self):
        from PIL import Image as PILImage

        images = [PILImage.new("RGB", (2, 2), color=(i, 0, 0)) for i in range(6)]
        client, threads = self._client_with_recorded_posts(images)

        result = client.process_images_vllm(images, model_name="m")

        assert client.session.post.call_count == 6
        embeddings = result["embeddings"]
        assert len(embeddings) == 6
        for i in range(6):
            assert float(np.asarray(embeddings[i])[0][0]) == float(i)
        # Multi-image batches must fan out over worker threads.
        assert len(threads) > 1

    def test_single_image_returns_bare_array(self):
        from PIL import Image as PILImage

        images = [PILImage.new("RGB", (2, 2), color=(0, 0, 0))]
        client, _ = self._client_with_recorded_posts(images)

        result = client.process_images_vllm(images, model_name="m")

        assert client.session.post.call_count == 1
        arr = np.asarray(result["embeddings"])
        assert arr.shape == (1, 4)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestVideoPrismVespaFormat:
    """Multi-vector VideoPrism → Vespa conversion must emit the compact
    mixed-tensor blocks form (one dense row per patch), not a dict per
    tensor cell, with values identical to the source array."""

    def test_blocks_form_carries_exact_rows(self):
        from cogniverse_core.common.models.videoprism_loader import (
            VideoPrismLoader,
        )

        loader = object.__new__(VideoPrismLoader)
        rng = np.random.default_rng(3)
        embeddings = rng.standard_normal((5, 8)).astype(np.float32)

        float_dict, binary_dict = loader.embeddings_to_vespa_format(embeddings)

        assert set(float_dict.keys()) == {"blocks"}
        blocks = float_dict["blocks"]
        assert sorted(blocks.keys(), key=int) == ["0", "1", "2", "3", "4"]
        for idx in range(5):
            assert blocks[str(idx)] == embeddings[idx].tolist()

        # Binary side unchanged: one hex string per patch, dim/8 bytes each.
        assert sorted(binary_dict.keys()) == [f"patch{i}" for i in range(5)]
        for i in range(5):
            expected_bits = np.packbits(np.where(embeddings[i] > 0, 1, 0)).astype(
                np.int8
            )
            assert binary_dict[f"patch{i}"] == expected_bits.tobytes().hex()


class TestPerKeyModelLoadLocks:
    """A cold load of one model must not block cache hits for another —
    the single global lock previously serialized every lookup behind any
    in-flight (minutes-long) load."""

    def test_cache_hit_returns_while_other_model_loads(self, monkeypatch):
        import threading
        import time

        from cogniverse_core.common.models import model_loaders as ml

        monkeypatch.setattr(ml, "_model_cache", ml.OrderedDict(), raising=True)

        # Pre-warm model A in the cache (no parameters attr → fast path).
        ml._model_cache["model-a"] = ("model_a", "proc_a")

        slow_load_started = threading.Event()
        release_slow_load = threading.Event()

        class _SlowLoader:
            def load_model(self):
                slow_load_started.set()
                assert release_slow_load.wait(timeout=5), "test deadlock"
                return "model_b", "proc_b"

        monkeypatch.setattr(
            ml.ModelLoaderFactory,
            "create_loader",
            staticmethod(
                lambda name, config, logger=None, *, _resolved_headers=None: (
                    _SlowLoader()
                )
            ),
        )

        results = {}

        def load_b():
            results["b"] = ml.get_or_load_model("model-b", {}, None)

        loader_thread = threading.Thread(target=load_b)
        loader_thread.start()
        assert slow_load_started.wait(timeout=5)

        # While B is mid-load, a cache hit for A must return immediately.
        t0 = time.perf_counter()
        hit = ml.get_or_load_model("model-a", {}, None)
        elapsed = time.perf_counter() - t0
        assert hit == ("model_a", "proc_a")
        assert elapsed < 1.0, (
            f"cache hit blocked {elapsed:.2f}s behind an unrelated cold load"
        )

        release_slow_load.set()
        loader_thread.join(timeout=5)
        assert results["b"] == ("model_b", "proc_b")
        assert ml._model_cache["model-b"] == ("model_b", "proc_b")


@pytest.mark.unit
@pytest.mark.ci_fast
class TestQueryEncodeTimeout:
    """The per-query text-encode POST must fail fast on a dead endpoint —
    a 1800s timeout on the search hot path hangs every query for 30 min."""

    def test_query_encode_uses_bounded_timeout(self):
        import socket
        import threading
        import time

        import requests

        from cogniverse_core.common.models.model_loaders import RemoteInferenceClient
        from cogniverse_core.common.utils.circuit_breaker import CircuitBreaker

        # A real TCP endpoint that accepts the connection but never replies.
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        held = []

        def accept_and_hang():
            try:
                conn, _ = server.accept()
                held.append(conn)  # hold the connection open, send nothing
            except OSError:
                pass

        threading.Thread(target=accept_and_hang, daemon=True).start()

        CircuitBreaker.reset_registry()
        client = RemoteInferenceClient(endpoint_url=f"http://127.0.0.1:{port}")
        client.query_encode_timeout_s = 1.0  # inject a tiny budget

        captured = {}

        def worker():
            start = time.monotonic()
            try:
                client.process_queries_vllm(["cats"], model_name="m")
            except BaseException as exc:  # noqa: BLE001
                captured["exc"] = exc
            finally:
                captured["elapsed"] = time.monotonic() - start

        w = threading.Thread(target=worker, daemon=True)
        w.start()
        w.join(timeout=5.0)

        try:
            assert not w.is_alive(), (
                "query encode did not return within 5s — the timeout is not "
                "bounded (still the 1800s literal)"
            )
            assert isinstance(captured.get("exc"), requests.exceptions.Timeout)
            assert captured["elapsed"] < 3.0
        finally:
            server.close()
            for c in held:
                try:
                    c.close()
                except OSError:
                    pass


class TestGlinerDevice:
    """get_or_load_gliner honors the requested torch device for local models."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from cogniverse_core.common.models import model_loaders

        model_loaders._gliner_cache.clear()
        yield
        model_loaders._gliner_cache.clear()

    def test_non_cpu_device_moves_local_model_and_keys_cache(self, monkeypatch):
        import gliner

        from cogniverse_core.common.models.model_loaders import get_or_load_gliner

        moved = {"to": None}

        class _FakeModel:
            def to(self, device):
                moved["to"] = device
                return self

        monkeypatch.setattr(
            gliner.GLiNER, "from_pretrained", lambda *a, **k: _FakeModel()
        )

        m_cuda = get_or_load_gliner("fake/gliner", device="cuda")
        assert isinstance(m_cuda, _FakeModel)
        assert moved["to"] == "cuda"

        # cpu leaves from_pretrained's placement untouched, and device is part
        # of the cache key so it is a distinct instance from the cuda one.
        moved["to"] = None
        m_cpu = get_or_load_gliner("fake/gliner", device="cpu")
        assert moved["to"] is None
        assert m_cpu is not m_cuda

    def test_local_load_failure_raises_with_model_context(self, monkeypatch):
        import gliner

        from cogniverse_core.common.models import model_loaders

        failure = OSError("controlled weight read failure")

        def fail_load(*args, **kwargs):
            raise failure

        monkeypatch.setattr(gliner.GLiNER, "from_pretrained", fail_load)

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load local GLiNER model 'fake/broken-gliner': "
                "controlled weight read failure"
            ),
        ) as excinfo:
            model_loaders.get_or_load_gliner("fake/broken-gliner")

        assert excinfo.value.__cause__ is failure
        assert model_loaders._gliner_cache == model_loaders.OrderedDict()

    def test_local_loader_rejects_missing_model_instance(self, monkeypatch):
        import gliner

        from cogniverse_core.common.models import model_loaders

        monkeypatch.setattr(
            gliner.GLiNER,
            "from_pretrained",
            lambda *args, **kwargs: None,
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load local GLiNER model 'fake/empty-gliner': "
                "GLiNER.from_pretrained returned no model"
            ),
        ):
            model_loaders.get_or_load_gliner("fake/empty-gliner")

        assert model_loaders._gliner_cache == model_loaders.OrderedDict()

    def test_device_move_failure_raises_without_caching_wrong_device(self, monkeypatch):
        import gliner

        from cogniverse_core.common.models import model_loaders

        failure = RuntimeError("CUDA driver unavailable")

        class _WrongDeviceModel:
            def to(self, device):
                raise failure

        monkeypatch.setattr(
            gliner.GLiNER,
            "from_pretrained",
            lambda *args, **kwargs: _WrongDeviceModel(),
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to move local GLiNER model 'fake/gliner' to device "
                "'cuda': CUDA driver unavailable"
            ),
        ) as excinfo:
            model_loaders.get_or_load_gliner("fake/gliner", device="cuda")

        assert excinfo.value.__cause__ is failure
        assert model_loaders._gliner_cache == model_loaders.OrderedDict()

    def test_device_move_rejects_missing_model_instance(self, monkeypatch):
        import gliner

        from cogniverse_core.common.models import model_loaders

        class _MissingMovedModel:
            def to(self, device):
                return None

        monkeypatch.setattr(
            gliner.GLiNER,
            "from_pretrained",
            lambda *args, **kwargs: _MissingMovedModel(),
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to move local GLiNER model 'fake/gliner' to device "
                "'cuda': model.to returned no model"
            ),
        ):
            model_loaders.get_or_load_gliner("fake/gliner", device="cuda")

        assert model_loaders._gliner_cache == model_loaders.OrderedDict()


@pytest.mark.unit
class TestRemoteGlinerClientHTTPContract:
    """Exercise RemoteGlinerClient against a real sidecar-shaped HTTP server."""

    def _serve(self, handler_fn):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                status, payload = handler_fn(
                    self.path,
                    body,
                    self.headers.get("Authorization"),
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def test_entities_round_trip_with_exact_request_payload(self):
        import json

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        seen = {}

        def handler(path, body, authorization):
            seen["path"] = path
            seen["payload"] = json.loads(body)
            seen["authorization"] = authorization
            return 200, json.dumps(
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": 0.93,
                            "start": 0,
                            "end": 11,
                        },
                        {
                            "text": "radium",
                            "label": "chemical",
                            "score": 0.88,
                            "start": 24,
                            "end": 30,
                        },
                    ],
                    "model": "gliner_large-v2.1",
                }
            ).encode()

        server, url = self._serve(handler)
        try:
            client = RemoteGlinerClient(
                url,
                "gliner_large-v2.1",
                api_key="gliner-modal-secret",
            )
            entities = client.predict_entities(
                "Marie Curie discovered radium",
                ["person", "chemical"],
                threshold=0.5,
            )
        finally:
            server.shutdown()

        assert seen["path"] == "/predict_entities"
        assert seen["authorization"] == "Bearer gliner-modal-secret"
        assert seen["payload"] == {
            "text": "Marie Curie discovered radium",
            "labels": ["person", "chemical"],
            "threshold": 0.5,
            "model": "gliner_large-v2.1",
        }
        assert entities == [
            {
                "text": "Marie Curie",
                "label": "person",
                "score": 0.93,
                "start": 0,
                "end": 11,
            },
            {
                "text": "radium",
                "label": "chemical",
                "score": 0.88,
                "start": 24,
                "end": 30,
            },
        ]

    def test_concurrent_requests_preserve_each_response(self):
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        scores = (0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)

        def handler(path, body, authorization):
            request = json.loads(body)
            index = int(request["text"].removeprefix("entity-"))
            return 200, json.dumps(
                {
                    "entities": [
                        {
                            "text": request["text"],
                            "label": "concept",
                            "score": scores[index],
                            "start": index,
                            "end": index + 8,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                }
            ).encode()

        server, url = self._serve(handler)
        client = RemoteGlinerClient(url, "gliner_large-v2.1")
        barrier = threading.Barrier(8)

        def predict(index):
            barrier.wait(timeout=5)
            return client.predict_entities(f"entity-{index}", ["concept"])

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(predict, range(8)))
        finally:
            server.shutdown()

        assert results == [
            [
                {
                    "text": "entity-0",
                    "label": "concept",
                    "score": 0.9,
                    "start": 0,
                    "end": 8,
                }
            ],
            [
                {
                    "text": "entity-1",
                    "label": "concept",
                    "score": 0.91,
                    "start": 1,
                    "end": 9,
                }
            ],
            [
                {
                    "text": "entity-2",
                    "label": "concept",
                    "score": 0.92,
                    "start": 2,
                    "end": 10,
                }
            ],
            [
                {
                    "text": "entity-3",
                    "label": "concept",
                    "score": 0.93,
                    "start": 3,
                    "end": 11,
                }
            ],
            [
                {
                    "text": "entity-4",
                    "label": "concept",
                    "score": 0.94,
                    "start": 4,
                    "end": 12,
                }
            ],
            [
                {
                    "text": "entity-5",
                    "label": "concept",
                    "score": 0.95,
                    "start": 5,
                    "end": 13,
                }
            ],
            [
                {
                    "text": "entity-6",
                    "label": "concept",
                    "score": 0.96,
                    "start": 6,
                    "end": 14,
                }
            ],
            [
                {
                    "text": "entity-7",
                    "label": "concept",
                    "score": 0.97,
                    "start": 7,
                    "end": 15,
                }
            ],
        ]

    def test_explicit_empty_entities_list_is_valid(self):
        import json

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        server, url = self._serve(
            lambda p, b, auth: (
                200,
                json.dumps({"entities": [], "model": "gliner_large-v2.1"}).encode(),
            )
        )
        try:
            client = RemoteGlinerClient(url, "gliner_large-v2.1")
            assert client.predict_entities("text", ["person"]) == []
        finally:
            server.shutdown()

    @pytest.mark.parametrize(
        "payload,reason",
        [
            ([], "must be a JSON object"),
            ({"model": "gliner_large-v2.1"}, "must contain 'entities'"),
            (
                {"entities": None, "model": "gliner_large-v2.1"},
                "'entities' must be a list",
            ),
            (
                {"entities": {}, "model": "gliner_large-v2.1"},
                "'entities' must be a list",
            ),
            (
                {"entities": ["Marie Curie"], "model": "gliner_large-v2.1"},
                "entity at index 0 must be an object",
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": 0.93,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                (
                    "entity at index 0 must have exactly fields "
                    "['text', 'label', 'score', 'start', 'end']"
                ),
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": 0.93,
                            "start": 0,
                            "end": 11,
                            "confidence": 0.93,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                (
                    "entity at index 0 must have exactly fields "
                    "['text', 'label', 'score', 'start', 'end']"
                ),
            ),
            (
                {
                    "entities": [
                        {
                            "text": None,
                            "label": "person",
                            "score": 0.93,
                            "start": 0,
                            "end": 11,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                "entity at index 0 field 'text' must be a string",
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": 17,
                            "score": 0.93,
                            "start": 0,
                            "end": 11,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                "entity at index 0 field 'label' must be a string",
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": "0.93",
                            "start": 0,
                            "end": 11,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                "entity at index 0 field 'score' must be a number",
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": 0.93,
                            "start": False,
                            "end": 11,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                "entity at index 0 field 'start' must be an integer or null",
            ),
            (
                {
                    "entities": [
                        {
                            "text": "Marie Curie",
                            "label": "person",
                            "score": 0.93,
                            "start": 0,
                            "end": 11.5,
                        }
                    ],
                    "model": "gliner_large-v2.1",
                },
                "entity at index 0 field 'end' must be an integer or null",
            ),
        ],
    )
    def test_malformed_success_response_raises(self, payload, reason):
        import json

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        server, url = self._serve(
            lambda p, b, auth: (200, json.dumps(payload).encode())
        )
        endpoint = f"{url}/predict_entities"
        try:
            client = RemoteGlinerClient(url, "gliner_large-v2.1")
            with pytest.raises(ValueError) as excinfo:
                client.predict_entities("Marie Curie", ["person"])
        finally:
            server.shutdown()

        assert str(excinfo.value) == f"Remote GLiNER response from {endpoint} {reason}"

    def test_server_error_raises_not_empty_list(self):
        """A 5xx (sidecar outage) must RAISE — swallowing it to [] made the
        gateway's entity_extraction_failed degrade branch unreachable on the
        remote path, so an outage read as a genuine no-entities routing signal.
        Only an explicit empty entities list is a valid [] response."""
        import requests

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        server, url = self._serve(lambda p, b, auth: (500, b"{}"))
        try:
            client = RemoteGlinerClient(url, "gliner_large-v2.1")
            with pytest.raises(requests.exceptions.HTTPError):
                client.predict_entities("text", ["person"])
        finally:
            server.shutdown()

    def test_connection_refused_raises(self):
        """A dead sidecar (connection refused) is an outage, not empty."""
        import requests

        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        # Nothing listening on this port.
        client = RemoteGlinerClient("http://127.0.0.1:1", "gliner_large-v2.1")
        with pytest.raises(requests.exceptions.RequestException):
            client.predict_entities("text", ["person"])

    def test_get_or_load_gliner_returns_remote_client_for_url(self):
        from cogniverse_core.common.models.model_loaders import (
            RemoteGlinerClient,
            get_or_load_gliner,
        )

        loaded = get_or_load_gliner(
            "gliner_large-v2.1", inference_url="http://gliner:8080"
        )
        assert isinstance(loaded, RemoteGlinerClient)
        assert loaded._url == "http://gliner:8080"

    def test_modal_remote_cache_uses_environment_credential_concurrently(
        self, monkeypatch
    ):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from cogniverse_core.common.models import model_loaders

        token = "shared-production-key"
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)
        model_loaders._gliner_cache.clear()
        barrier = threading.Barrier(8)

        def resolve(_: int):
            barrier.wait(timeout=5)
            return model_loaders.get_or_load_gliner(
                "urchade/gliner_medium-v2.1",
                inference_url="https://gliner.modal.run",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            clients = list(executor.map(resolve, range(8)))

        assert len({id(client) for client in clients}) == 1
        assert clients[0]._session.headers["Authorization"] == f"Bearer {token}"
        cache_keys = repr(tuple(model_loaders._gliner_cache))
        assert token not in cache_keys

    def test_modal_remote_requires_environment_credential(self, monkeypatch):
        from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            RemoteGlinerClient(
                "https://gliner.modal.run",
                "urchade/gliner_large-v2.1",
            )


def test_remote_inference_client_uses_modal_environment_credential(monkeypatch):
    from cogniverse_core.common.models.model_loaders import RemoteInferenceClient

    token = "shared-production-key"
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)

    client = RemoteInferenceClient("https://colpali.modal.run")

    assert client.session.headers["Authorization"] == f"Bearer {token}"


def test_remote_inference_client_rejects_modal_caller_key(monkeypatch):
    from cogniverse_core.common.models.model_loaders import RemoteInferenceClient

    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")
    with pytest.raises(ValueError, match="api_key.*Modal"):
        RemoteInferenceClient(
            "https://colpali.modal.run",
            api_key="caller-specific-key",
        )


def test_remote_videoprism_wrapper_forwards_exact_model_name():
    import base64
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    from cogniverse_core.common.models.model_loaders import RemoteVideoPrismLoader

    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.update(json.loads(self.rfile.read(length)))
            payload = json.dumps(
                {
                    "embeddings": [[0.5, -0.25]],
                    "processing_time": 0.125,
                    "model": "videoprism_public_v1_base_hf",
                    "frames_processed": 16,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source_video = (
        Path(__file__).resolve().parents[2]
        / "system/resources/videos/v_-D1gdv_gQyw.mp4"
    )
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        loader = RemoteVideoPrismLoader(
            "videoprism_public_v1_base_hf",
            {"remote_inference_url": base_url},
        )
        wrapper, processor = loader.load_model()
        result = wrapper.process_video_segment(source_video, 0.0, 0.25)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert processor is None
    assert received["start_time"] == 0.0
    assert received["end_time"] == 0.25
    assert received["model"] == "videoprism_public_v1_base_hf"
    assert base64.b64decode(received["video"], validate=True)[4:8] == b"ftyp"
    np.testing.assert_array_equal(
        result["embeddings_np"],
        np.array([[0.5, -0.25]], dtype=np.float32),
    )
    assert result["processing_time"] == 0.125


@pytest.mark.parametrize(
    "loader_name,model_name",
    [
        ("colbert", "lightonai/LateOn"),
        ("whisper", "openai/whisper-large-v3-turbo"),
    ],
)
def test_specialized_modal_loaders_use_environment_credential(
    monkeypatch, loader_name, model_name
):
    from cogniverse_core.common.models.model_loaders import ModelLoaderFactory

    token = "shared-production-key"
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)
    loader = ModelLoaderFactory.create_loader(
        model_name,
        {
            "model_loader": loader_name,
            "remote_inference_url": f"https://{loader_name}.modal.run",
        },
    )

    model, _ = loader.load_model()

    assert model.session.headers["Authorization"] == f"Bearer {token}"


def test_remote_model_cache_isolates_custom_credentials():
    from cogniverse_core.common.models import model_loaders

    model_loaders._model_cache.clear()
    base_config = {
        "model_loader": "colbert",
        "remote_inference_url": "http://colbert.internal:8000",
    }
    first, _ = model_loaders.get_or_load_model(
        "lightonai/LateOn",
        {**base_config, "remote_inference_api_key": "first-key"},
    )
    second, _ = model_loaders.get_or_load_model(
        "lightonai/LateOn",
        {**base_config, "remote_inference_api_key": "second-key"},
    )

    assert first is not second
    assert first.session.headers["Authorization"] == "Bearer first-key"
    assert second.session.headers["Authorization"] == "Bearer second-key"
    cache_keys = " ".join(model_loaders._model_cache)
    assert "first-key" not in cache_keys
    assert "second-key" not in cache_keys


def test_remote_model_cache_is_bounded_lru_and_closes_evicted_client(monkeypatch):
    from cogniverse_core.common.models import model_loaders

    class Client:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.close_calls = 0

        def _close(self):
            self.close_calls += 1

    class Loader:
        def __init__(self, client):
            self.client = client

        def load_model(self):
            return self.client, None

    clients = {}

    def create_loader(name, config, logger=None, *, _resolved_headers=None):
        client = Client(config["remote_inference_url"])
        clients[client.endpoint] = client
        return Loader(client)

    monkeypatch.setattr(model_loaders, "_MODEL_CACHE_CAPACITY", 2)
    monkeypatch.setattr(
        model_loaders.ModelLoaderFactory,
        "create_loader",
        staticmethod(create_loader),
    )
    model_loaders._model_cache.clear()

    configs = [
        {
            "model_loader": "colbert",
            "remote_inference_url": f"http://embed-{index}:8000",
            "remote_inference_api_key": f"key-{index}",
        }
        for index in range(3)
    ]
    first, _ = model_loaders.get_or_load_model("lightonai/LateOn", configs[0])
    second, _ = model_loaders.get_or_load_model("lightonai/LateOn", configs[1])
    assert model_loaders.get_or_load_model("lightonai/LateOn", configs[0])[0] is first
    third, _ = model_loaders.get_or_load_model("lightonai/LateOn", configs[2])

    assert len(model_loaders._model_cache) == 2
    assert list(model_loaders._model_cache.values()) == [(first, None), (third, None)]
    assert first.close_calls == 0
    assert second.close_calls == 1
    assert third.close_calls == 0


def test_remote_model_cache_restores_old_entry_when_eviction_close_fails(
    monkeypatch,
):
    from cogniverse_core.common.models import model_loaders

    class Client:
        def __init__(self, endpoint, *, fail_close=False):
            self.endpoint = endpoint
            self.fail_close = fail_close
            self.close_calls = 0

        def _close(self):
            self.close_calls += 1
            if self.fail_close:
                raise OSError("controlled close failure")

    class Loader:
        def __init__(self, client):
            self.client = client

        def load_model(self):
            return self.client, None

    created = []

    def create_loader(name, config, logger=None, *, _resolved_headers=None):
        client = Client(
            config["remote_inference_url"],
            fail_close=not created,
        )
        created.append(client)
        return Loader(client)

    monkeypatch.setattr(model_loaders, "_MODEL_CACHE_CAPACITY", 1)
    monkeypatch.setattr(
        model_loaders.ModelLoaderFactory,
        "create_loader",
        staticmethod(create_loader),
    )
    model_loaders._model_cache.clear()
    base = {"model_loader": "colbert"}
    first, _ = model_loaders.get_or_load_model(
        "lightonai/LateOn",
        {**base, "remote_inference_url": "http://embed-first:8000"},
    )

    with pytest.raises(
        RuntimeError,
        match="model cache eviction failed.*controlled close failure",
    ):
        model_loaders.get_or_load_model(
            "lightonai/LateOn",
            {**base, "remote_inference_url": "http://embed-second:8000"},
        )

    assert list(model_loaders._model_cache.values()) == [(first, None)]
    assert created[0].close_calls == 1
    assert created[1].close_calls == 1


def test_gliner_cache_is_bounded_lru_and_closes_evicted_client(monkeypatch):
    from cogniverse_core.common.models import model_loaders

    class Client:
        def __init__(self, url, model_name, **kwargs):
            self.url = url
            self.close_calls = 0

        def _close(self):
            self.close_calls += 1

    monkeypatch.setattr(model_loaders, "_GLINER_CACHE_CAPACITY", 2)
    monkeypatch.setattr(model_loaders, "RemoteGlinerClient", Client)
    model_loaders._gliner_cache.clear()

    first = model_loaders.get_or_load_gliner(
        "urchade/gliner_large-v2.1", inference_url="http://gliner-first:8080"
    )
    second = model_loaders.get_or_load_gliner(
        "urchade/gliner_large-v2.1", inference_url="http://gliner-second:8080"
    )
    assert (
        model_loaders.get_or_load_gliner(
            "urchade/gliner_large-v2.1", inference_url="http://gliner-first:8080"
        )
        is first
    )
    third = model_loaders.get_or_load_gliner(
        "urchade/gliner_large-v2.1", inference_url="http://gliner-third:8080"
    )

    assert len(model_loaders._gliner_cache) == 2
    assert list(model_loaders._gliner_cache.values()) == [first, third]
    assert first.close_calls == 0
    assert second.close_calls == 1
    assert third.close_calls == 0


def test_gliner_cache_and_client_share_one_credential_snapshot(monkeypatch):
    from cogniverse_core.common.models import model_loaders

    resolved = iter(
        [
            {"Authorization": "Bearer first-key"},
            {"Authorization": "Bearer second-key"},
        ]
    )
    monkeypatch.setattr(
        model_loaders,
        "_resolved_inference_headers",
        lambda endpoint_url, api_key: next(resolved),
    )
    model_loaders._gliner_cache.clear()

    client = model_loaders.get_or_load_gliner(
        "urchade/gliner_large-v2.1",
        inference_url="https://gliner.modal.run",
    )

    assert client._session.headers["Authorization"] == "Bearer first-key"
    assert next(resolved) == {"Authorization": "Bearer second-key"}


class _PoolingResponse:
    def __init__(self, payload, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


def _remote_lateon():
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    wrapper, processor = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": "http://lateon.test:8000"},
    ).load_model()
    assert processor is None
    return wrapper


@pytest.mark.unit
def test_remote_lateon_queries_send_raw_text_with_is_query():
    wrapper = _remote_lateon()
    requests_seen = []

    class Session:
        def post(self, url, *, json, timeout):
            requests_seen.append((url, json, timeout))
            return _PoolingResponse(
                {
                    "object": "list",
                    "model": "lightonai/LateOn",
                    "data": [
                        {
                            "object": "pooling",
                            "index": index,
                            "data": [[float(index)] * 128] * 32,
                        }
                        for index, _ in enumerate(json["input"])
                    ],
                }
            )

    wrapper.session = Session()
    result = wrapper.encode(["first query", "second query"], is_query=True)

    assert requests_seen == [
        (
            "http://lateon.test:8000/pooling",
            {
                "input": ["first query", "second query"],
                "model": "lightonai/LateOn",
                "is_query": True,
            },
            120,
        )
    ]
    assert result == [[[0.0] * 128] * 32, [[1.0] * 128] * 32]


@pytest.mark.unit
def test_remote_lateon_documents_send_unprefixed_text_and_keep_all_rows():
    """The PyLate service applies the document marker and punctuation
    skiplist itself; the client must send the raw text (no ``[D] `` prefix,
    ``is_query`` false) and return the matrix without dropping rows."""
    wrapper = _remote_lateon()
    requests_seen = []

    class Session:
        def post(self, url, *, json, timeout):
            requests_seen.append(json)
            return _PoolingResponse(
                {
                    "object": "list",
                    "model": "lightonai/LateOn",
                    "data": [
                        {"object": "pooling", "index": 0, "data": [[0.25] * 128] * 7}
                    ],
                }
            )

    wrapper.session = Session()
    result = wrapper.encode(["Vespa stores token embeddings."], is_query=False)

    assert requests_seen == [
        {
            "input": ["Vespa stores token embeddings."],
            "model": "lightonai/LateOn",
            "is_query": False,
        }
    ]
    assert result == [[[0.25] * 128] * 7]


@pytest.mark.unit
def test_remote_lateon_concurrent_queries_keep_their_batches_isolated():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    wrapper = _remote_lateon()
    barrier = Barrier(2)
    payloads = []
    payload_lock = Lock()

    class Session:
        def post(self, url, *, json, timeout):
            barrier.wait(timeout=3)
            with payload_lock:
                payloads.append(json)
            value = float(len(json["input"][0]))
            return _PoolingResponse(
                {
                    "object": "list",
                    "model": "lightonai/LateOn",
                    "data": [
                        {"object": "pooling", "index": 0, "data": [[value] * 4] * 2}
                    ],
                }
            )

    wrapper.session = Session()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(
            pool.map(
                lambda text: wrapper.encode([text], is_query=True)[0],
                ("first query", "second query"),
            )
        )

    assert first == [[float(len("first query"))] * 4] * 2
    assert second == [[float(len("second query"))] * 4] * 2
    assert sorted(payloads, key=lambda payload: payload["input"][0]) == [
        {
            "input": ["first query"],
            "model": "lightonai/LateOn",
            "is_query": True,
        },
        {
            "input": ["second query"],
            "model": "lightonai/LateOn",
            "is_query": True,
        },
    ]


@pytest.mark.unit
def test_remote_lateon_pooling_failure_has_model_and_endpoint_context():
    import requests

    wrapper = _remote_lateon()

    class Session:
        def post(self, url, *, json, timeout):
            return _PoolingResponse(
                {},
                requests.HTTPError("503 Service Unavailable"),
            )

    wrapper.session = Session()

    with pytest.raises(
        RuntimeError,
        match=(
            "remote ColBERT pooling failed for model 'lightonai/LateOn' "
            "at http://lateon.test:8000"
        ),
    ):
        wrapper.encode(["first query"], is_query=True)


@pytest.mark.unit
def test_remote_lateon_embedding_count_mismatch_names_counts():
    wrapper = _remote_lateon()

    class Session:
        def post(self, url, *, json, timeout):
            return _PoolingResponse(
                {
                    "object": "list",
                    "model": "lightonai/LateOn",
                    "data": [
                        {"object": "pooling", "index": 0, "data": [[0.5] * 4]},
                        {"object": "pooling", "index": 1, "data": [[0.5] * 4]},
                    ],
                }
            )

    wrapper.session = Session()

    with pytest.raises(
        RuntimeError,
        match=(
            "remote ColBERT pooling returned 2 embeddings for 1 inputs "
            "from model 'lightonai/LateOn' at http://lateon.test:8000"
        ),
    ):
        wrapper.encode(["only one text"], is_query=False)


@pytest.mark.unit
def test_remote_lateon_non_list_embedding_is_rejected():
    wrapper = _remote_lateon()

    class Session:
        def post(self, url, *, json, timeout):
            return _PoolingResponse(
                {
                    "object": "list",
                    "model": "lightonai/LateOn",
                    "data": [{"object": "pooling", "index": 0, "data": "corrupt"}],
                }
            )

    wrapper.session = Session()

    with pytest.raises(
        RuntimeError,
        match=(
            "remote ColBERT pooling returned a non-list embedding for "
            "model 'lightonai/LateOn' at http://lateon.test:8000"
        ),
    ):
        wrapper.encode(["a document"], is_query=False)


def _dead_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _remote_lateon_at(url: str):
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    wrapper, processor = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": url},
    ).load_model()
    assert processor is None
    return wrapper


@pytest.mark.unit
def test_remote_lateon_unreachable_sidecar_raises_service_unavailable():
    import requests

    from cogniverse_foundation.config.inference_service import (
        InferenceServiceUnavailableError,
    )

    port = _dead_port()
    wrapper = _remote_lateon_at(f"http://127.0.0.1:{port}")

    with pytest.raises(InferenceServiceUnavailableError) as excinfo:
        wrapper.encode(["first query"], is_query=True)

    exc = excinfo.value
    assert exc.service == "colbert_pooling"
    assert str(exc) == (
        "remote ColBERT pooling sidecar unreachable for model "
        f"'lightonai/LateOn' at http://127.0.0.1:{port}"
    )
    assert isinstance(exc.__cause__, requests.ConnectionError)


@pytest.mark.unit
def test_remote_lateon_http_error_is_not_service_unavailable():
    import requests

    from cogniverse_foundation.config.inference_service import (
        InferenceServiceUnavailableError,
    )

    wrapper = _remote_lateon()

    class Session:
        def post(self, url, *, json, timeout):
            return _PoolingResponse(
                {},
                requests.HTTPError("400 Bad Request"),
            )

    wrapper.session = Session()

    with pytest.raises(RuntimeError) as excinfo:
        wrapper.encode(["first query"], is_query=True)

    assert not isinstance(excinfo.value, InferenceServiceUnavailableError)
    assert str(excinfo.value) == (
        "remote ColBERT pooling failed for model 'lightonai/LateOn' "
        "at http://lateon.test:8000"
    )


@pytest.mark.unit
def test_remote_lateon_unreachable_sidecar_concurrent_encodes_all_raise():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cogniverse_foundation.config.inference_service import (
        InferenceServiceUnavailableError,
    )

    port = _dead_port()
    wrapper = _remote_lateon_at(f"http://127.0.0.1:{port}")
    barrier = threading.Barrier(4)

    def encode_after_barrier(text: str) -> Exception:
        barrier.wait(timeout=5)
        try:
            wrapper.encode([text], is_query=True)
        except Exception as exc:
            return exc
        raise AssertionError("encode against a dead port must raise")

    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = list(pool.map(encode_after_barrier, ["a", "b", "c", "d"]))

    assert [type(e) for e in errors] == [InferenceServiceUnavailableError] * 4
