from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import socket
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
import uvicorn
from cogniverse_cli.modal_inference import face as face_deploy
from cogniverse_cli.modal_inference.clap import app as clap_modal_app
from cogniverse_cli.modal_inference.face import app as face_modal_app
from cogniverse_cli.modal_inference.servers import clap as clap_server
from cogniverse_cli.modal_inference.servers import face as face_server
from PIL import Image

from cogniverse_agents.graph.face_extractor import extract_faces_per_keyframe
from cogniverse_foundation.inference_specs import get_inference_service_spec
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)

API_KEY = "audio-face-test-key"
VIDEO_A = Path("tests/system/resources/videos/v_-D1gdv_gQyw.mp4")
VIDEO_B = Path("tests/system/resources/videos/v_-6dz6tBH77I.mp4")
FACE_MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
FACE_MODEL_REVISION = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
FACE_MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


class _ArrayTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)

    def to(self, device: str) -> _ArrayTensor:
        return self

    def squeeze(self) -> _ArrayTensor:
        return _ArrayTensor(np.squeeze(self.values))

    def cpu(self) -> _ArrayTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _AudioProcessor:
    def __call__(self, *, audios, sampling_rate, return_tensors):
        assert sampling_rate == 48000
        assert return_tensors == "pt"
        return {"samples": _ArrayTensor(np.asarray(audios, dtype=np.float32))}


class _SpectralClapModel:
    def get_audio_features(self, *, samples: _ArrayTensor) -> _ArrayTensor:
        spectrum = np.abs(np.fft.rfft(samples.values))
        dominant_bin = int(np.argmax(spectrum[1:513]) + 1)
        vector = np.zeros((1, 512), dtype=np.float32)
        vector[0, dominant_bin] = 0.8
        vector[0, (dominant_bin + 13) % 512] = 0.6
        return _ArrayTensor(vector)


class _ImageSignatureFaceModel:
    def get(self, image_bgr: np.ndarray):
        height, width = image_bgr.shape[:2]
        signature = int(round(float(image_bgr.mean()))) % 512
        vector = np.zeros(512, dtype=np.float32)
        vector[signature] = 0.8
        vector[(signature + 13) % 512] = 0.6
        return [
            SimpleNamespace(
                bbox=np.asarray(
                    [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
                    dtype=np.float32,
                ),
                normed_embedding=vector,
                det_score=0.99,
            )
        ]


@pytest.fixture(autouse=True)
def _reset_sidecar_models():
    clap_server._MODEL = None
    clap_server._PROCESSOR = None
    face_server._MODEL = None
    yield
    clap_server._MODEL = None
    clap_server._PROCESSOR = None
    face_server._MODEL = None


@contextmanager
def _live_server(app):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            ws="none",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("audio/face inference server did not start")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("audio/face inference server did not stop")


def _modal_asgi_app(modal_app):
    return modal_app.registered_functions["Inference"].get_raw_f()()


def _authorization() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _write_tone(path: Path, frequency_hz: float) -> None:
    import soundfile as sf

    sample_rate = 16000
    duration_s = 0.5
    times = np.linspace(
        0,
        duration_s,
        int(sample_rate * duration_s),
        endpoint=False,
    )
    samples = (0.3 * np.sin(2 * np.pi * frequency_hz * times)).astype(np.float32)
    sf.write(path, samples, sample_rate)


def _video_frame_b64(path: Path, seconds: float = 1.0) -> str:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ok, frame = capture.read()
    capture.release()
    assert ok
    encoded, buffer = cv2.imencode(".jpg", frame)
    assert encoded
    return base64.b64encode(buffer.tobytes()).decode("ascii")


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://inference.test",
        ) as client:
            return await client.request(method, path, **kwargs)


def _face_artifact(root: Path) -> Path:
    model_dir = root / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    for filename in FACE_MODEL_FILES:
        (model_dir / filename).write_bytes(filename.encode())
    return model_dir


def _face_artifact_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename in FACE_MODEL_FILES:
            archive.writestr(filename, filename.encode())
    return buffer.getvalue()


def test_face_artifact_contract_pins_the_official_release():
    spec = get_inference_service_spec("face_embed")

    assert (
        face_deploy._FACE_MODEL_URL,
        face_deploy._FACE_MODEL_SHA256,
        face_deploy._FACE_MODEL_ROOT,
        spec.model_id,
        spec.model_revision,
    ) == (
        FACE_MODEL_URL,
        FACE_MODEL_REVISION,
        "/opt/insightface",
        "buffalo_l",
        FACE_MODEL_REVISION,
    )


def test_face_artifact_installer_verifies_digest_before_unpack(monkeypatch, tmp_path):
    payload = _face_artifact_zip()
    requested_urls = []

    def open_archive(url):
        requested_urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(face_deploy, "urlopen", open_archive, raising=False)

    with pytest.raises(RuntimeError, match="buffalo_l.zip SHA256 mismatch"):
        face_deploy._install_face_artifact(
            url=FACE_MODEL_URL,
            expected_sha256=FACE_MODEL_REVISION,
            model_root=str(tmp_path),
            required_files=FACE_MODEL_FILES,
        )

    assert requested_urls == [FACE_MODEL_URL]
    assert not (tmp_path / "models" / "buffalo_l").exists()


def test_face_artifact_installer_unpacks_the_complete_model(monkeypatch, tmp_path):
    payload = _face_artifact_zip()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        face_deploy,
        "urlopen",
        lambda url: io.BytesIO(payload) if url == FACE_MODEL_URL else None,
        raising=False,
    )

    face_deploy._install_face_artifact(
        url=FACE_MODEL_URL,
        expected_sha256=digest,
        model_root=str(tmp_path),
        required_files=FACE_MODEL_FILES,
    )

    model_dir = tmp_path / "models" / "buffalo_l"
    assert {path.name: path.read_bytes() for path in sorted(model_dir.iterdir())} == {
        filename: filename.encode() for filename in FACE_MODEL_FILES
    }


def test_face_artifact_installer_preserves_copy_error_when_cleanup_fails(
    monkeypatch, tmp_path
):
    payload = _face_artifact_zip()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(face_deploy, "urlopen", lambda url: io.BytesIO(payload))
    monkeypatch.setattr(
        face_deploy.shutil,
        "copyfileobj",
        lambda source, destination: (_ for _ in ()).throw(
            OSError("artifact stream closed")
        ),
    )
    real_rmtree = face_deploy.shutil.rmtree
    cleanup_calls: list[tuple[Path, bool]] = []

    def fail_cleanup(path, *, ignore_errors=False):
        cleanup_calls.append((Path(path), ignore_errors))
        if not ignore_errors:
            raise PermissionError("staging cleanup denied")

    monkeypatch.setattr(face_deploy.shutil, "rmtree", fail_cleanup)
    models_dir = tmp_path / "models"
    try:
        with pytest.raises(OSError) as exc_info:
            face_deploy._install_face_artifact(
                url=FACE_MODEL_URL,
                expected_sha256=digest,
                model_root=str(tmp_path),
                required_files=FACE_MODEL_FILES,
            )

        staging_dirs = list(models_dir.glob(".buffalo_l-*"))
        assert len(staging_dirs) == 1
        assert cleanup_calls == [(staging_dirs[0], False)]
        assert str(exc_info.value) == "artifact stream closed"
        assert exc_info.value.__notes__ == [
            f"failed to remove face model staging directory {staging_dirs[0]} "
            "(PermissionError): staging cleanup denied"
        ]
    finally:
        for staging_dir in models_dir.glob(".buffalo_l-*"):
            real_rmtree(staging_dir)


def test_face_image_runs_the_verified_installer_during_build(monkeypatch):
    events = []

    class Image:
        def apt_install(self, *packages):
            events.append(("apt_install", packages))
            return self

        def pip_install(self, *packages):
            events.append(("pip_install", packages))
            return self

        def env(self, values):
            events.append(("env", values))
            return self

        def run_function(self, function, **kwargs):
            events.append(("run_function", function, kwargs))
            return self

        def add_local_python_source(self, *modules, copy):
            events.append(("add_local_python_source", modules, copy))
            return self

    image = Image()
    monkeypatch.setattr(
        face_deploy.modal.Image,
        "debian_slim",
        lambda *, python_version: image,
    )

    assert face_deploy._build_image() is image
    install_index = next(
        index for index, event in enumerate(events) if event[0] == "run_function"
    )
    assert events[install_index] == (
        "run_function",
        face_deploy._install_face_artifact,
        {
            "kwargs": {
                "url": FACE_MODEL_URL,
                "expected_sha256": FACE_MODEL_REVISION,
                "model_root": "/opt/insightface",
                "required_files": FACE_MODEL_FILES,
            }
        },
    )
    assert events[install_index + 1] == (
        "add_local_python_source",
        (
            "cogniverse_cli.modal_inference",
            "cogniverse_foundation.inference_specs",
        ),
        True,
    )


def test_modal_apps_pin_gpu_cache_auth_and_scale_to_zero():
    expected = {
        "clap_embed": (
            clap_modal_app,
            "/root/.cache/huggingface",
            "modal.Volume.from_name('cogniverse-huggingface-cache')",
        ),
        "face_embed": (
            face_modal_app,
            "/root/.insightface",
            "modal.Volume.from_name('cogniverse-insightface-cache')",
        ),
    }

    for service, (modal_app, cache_path, cache_repr) in expected.items():
        spec = get_inference_service_spec(service)
        function = modal_app.registered_functions["Inference"]

        assert modal_app.name == spec.modal_app
        assert modal_app.registered_web_endpoints == ["Inference"]
        assert function.tag == spec.modal_object
        assert function.spec.gpus == list(spec.gpu_candidates)
        assert spec.min_containers == 0
        assert spec.scaledown_window == 300
        assert list(function.spec.volumes) == [cache_path]
        assert repr(function.spec.volumes[cache_path]) == cache_repr
        assert [repr(secret) for secret in function.spec.secrets] == [
            "modal.Secret.from_name('cogniverse-inference-api-key')"
        ]


def test_audio_generator_preserves_fixed_audio_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    clap_server._MODEL = _SpectralClapModel()
    clap_server._PROCESSOR = _AudioProcessor()
    app = _modal_asgi_app(clap_modal_app)

    tone = tmp_path / "tone-440.wav"
    unrelated = tmp_path / "tone-880.wav"
    _write_tone(tone, 440.0)
    _write_tone(unrelated, 880.0)

    with _live_server(app) as endpoint:
        identity = httpx.get(
            f"{endpoint}/v1/models",
            headers=_authorization(),
        )
        generator = AudioEmbeddingGenerator(
            clap_endpoint_url=endpoint,
            clap_headers=_authorization(),
        )
        try:
            first = generator.generate_acoustic_embedding(audio_path=tone)
            second = generator.generate_acoustic_embedding(audio_path=tone)
            other = generator.generate_acoustic_embedding(audio_path=unrelated)
        finally:
            generator.close()

    assert identity.json() == {
        "data": [
            {
                "created": 0,
                "id": "laion/clap-htsat-unfused",
                "object": "model",
                "owned_by": "cogniverse",
                "revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
            }
        ],
        "object": "list",
    }
    assert first.shape == (512,)
    assert np.flatnonzero(first).tolist() == [220, 233]
    assert first[[220, 233]].tolist() == pytest.approx([0.8, 0.6], abs=1e-7)
    assert 1.39999 < float(np.abs(first).sum()) < 1.40001
    assert np.array_equal(first, second)
    assert np.flatnonzero(other).tolist() == [440, 453]
    assert float(np.dot(first, second)) > 0.99999
    assert float(np.dot(first, other)) < 0.00001


def test_face_extractor_preserves_fixed_image_embeddings(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    face_server._MODEL = _ImageSignatureFaceModel()
    app = _modal_asgi_app(face_modal_app)
    frame = _video_frame_b64(VIDEO_A)
    unrelated = _video_frame_b64(VIDEO_B)
    processing_results = {
        "keyframes": {
            "items": [
                {"segment_id": "same-a", "ts_start": 1.0, "image_b64": frame},
                {"segment_id": "same-b", "ts_start": 2.0, "image_b64": frame},
                {
                    "segment_id": "unrelated",
                    "ts_start": 3.0,
                    "image_b64": unrelated,
                },
            ]
        }
    }

    with _live_server(app) as endpoint:
        identity = httpx.get(
            f"{endpoint}/v1/models",
            headers=_authorization(),
        )
        records = extract_faces_per_keyframe(
            processing_results,
            "fixed-videos",
            endpoint,
            headers=_authorization(),
        )

    assert identity.json() == {
        "data": [
            {
                "created": 0,
                "id": "buffalo_l",
                "object": "model",
                "owned_by": "cogniverse",
                "revision": FACE_MODEL_REVISION,
            }
        ],
        "object": "list",
    }
    assert [record.segment_id for record in records] == [
        "same-a",
        "same-b",
        "unrelated",
    ]
    assert [record.bbox for record in records] == [
        (320, 180, 960, 540),
        (320, 180, 960, 540),
        (160, 120, 480, 360),
    ]
    vectors = [np.asarray(record.vec, dtype=np.float32) for record in records]
    assert [vector.shape for vector in vectors] == [(512,), (512,), (512,)]
    assert [np.flatnonzero(vector).tolist() for vector in vectors] == [
        [117, 130],
        [117, 130],
        [135, 148],
    ]
    assert all(1.39999 < float(np.abs(vector).sum()) < 1.40001 for vector in vectors)
    assert float(np.dot(vectors[0], vectors[1])) > 0.99999
    assert float(np.dot(vectors[0], vectors[2])) < 0.00001


def test_clap_concurrent_cold_health_requests_build_one_pinned_model(monkeypatch):
    events: list[tuple] = []
    loaded_model = SimpleNamespace(
        to=lambda device: events.append(("device", device)),
        eval=lambda: events.append(("eval",)),
    )

    class _Processor:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str):
            events.append(("processor", model_id, revision))
            time.sleep(0.05)
            return object()

    class _Model:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str):
            events.append(("model", model_id, revision))
            return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(ClapModel=_Model, ClapProcessor=_Processor),
    )
    config = clap_server.ClapEmbedConfig(device="cuda")
    app = clap_server.build_app(config)

    with _live_server(app) as endpoint:
        with ThreadPoolExecutor(max_workers=12) as executor:
            responses = list(
                executor.map(
                    lambda _: httpx.get(f"{endpoint}/health", timeout=5),
                    range(12),
                )
            )

    expected_revision = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    assert events == [
        ("processor", "laion/clap-htsat-unfused", expected_revision),
        ("model", "laion/clap-htsat-unfused", expected_revision),
        ("device", "cuda"),
        ("eval",),
    ]
    assert [response.status_code for response in responses] == [200] * 12
    assert [response.json() for response in responses] == [
        {
            "status": "ready",
            "model": "laion/clap-htsat-unfused",
            "model_revision": expected_revision,
        }
    ] * 12


def test_face_concurrent_cold_health_requests_build_one_gpu_model(
    monkeypatch, tmp_path
):
    events: list[tuple] = []
    _face_artifact(tmp_path)

    class _FaceAnalysis:
        def __new__(cls, *, name: str, root: str, providers: list[str]):
            events.append(("construct", name, root, tuple(providers)))
            time.sleep(0.05)
            return SimpleNamespace(
                prepare=lambda *, ctx_id, det_size: events.append(
                    ("prepare", ctx_id, det_size)
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "insightface.app",
        SimpleNamespace(FaceAnalysis=_FaceAnalysis),
    )
    config = face_server.FaceEmbedConfig(ctx_id=0, model_root=str(tmp_path))
    app = face_server.build_app(config)

    with _live_server(app) as endpoint:
        with ThreadPoolExecutor(max_workers=12) as executor:
            responses = list(
                executor.map(
                    lambda _: httpx.get(f"{endpoint}/health", timeout=5),
                    range(12),
                )
            )

    assert events == [
        ("construct", "buffalo_l", str(tmp_path), ("CUDAExecutionProvider",)),
        ("prepare", 0, (640, 640)),
    ]
    assert [response.status_code for response in responses] == [200] * 12
    assert [response.json() for response in responses] == [
        {
            "status": "ready",
            "model": "buffalo_l",
            "model_revision": FACE_MODEL_REVISION,
        }
    ] * 12


def test_face_health_reports_exact_artifact_identity():
    face_server._MODEL = _ImageSignatureFaceModel()
    response = asyncio.run(_request(face_server.app, "GET", "/health"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "buffalo_l",
        "model_revision": FACE_MODEL_REVISION,
    }


def test_clap_health_load_failure_is_not_ready_and_next_request_retries(monkeypatch):
    attempts = 0

    class _Processor:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str):
            nonlocal attempts
            attempts += 1
            assert (model_id, revision) == (
                "laion/clap-htsat-unfused",
                "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
            )
            if attempts == 1:
                raise OSError("processor files are unreadable")
            return object()

    class _Model:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str):
            return SimpleNamespace(to=lambda device: None, eval=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(ClapModel=_Model, ClapProcessor=_Processor),
    )

    failed = asyncio.run(_request(clap_server.app, "GET", "/health"))
    recovered = asyncio.run(_request(clap_server.app, "GET", "/health"))

    assert failed.status_code == 503
    assert failed.json() == {
        "detail": (
            "clap_embed: model laion/clap-htsat-unfused load failed (OSError): "
            "processor files are unreadable"
        )
    }
    assert recovered.status_code == 200
    assert recovered.json() == {
        "status": "ready",
        "model": "laion/clap-htsat-unfused",
        "model_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
    }
    assert attempts == 2


def test_face_health_load_failure_is_not_ready_and_next_request_retries(
    monkeypatch, tmp_path
):
    _face_artifact(tmp_path)
    attempts = 0

    class _FaceAnalysis:
        def __init__(self, *, name: str, root: str):
            nonlocal attempts
            attempts += 1
            assert (name, root) == ("buffalo_l", str(tmp_path))
            if attempts == 1:
                raise OSError("recognizer graph is unreadable")

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            assert (ctx_id, det_size) == (-1, (640, 640))

    monkeypatch.setitem(
        sys.modules,
        "insightface.app",
        SimpleNamespace(FaceAnalysis=_FaceAnalysis),
    )
    app = face_server.build_app(face_server.FaceEmbedConfig(model_root=str(tmp_path)))

    failed = asyncio.run(_request(app, "GET", "/health"))
    recovered = asyncio.run(_request(app, "GET", "/health"))

    assert failed.status_code == 503
    assert failed.json() == {
        "detail": (
            "face_embed: model buffalo_l load failed (OSError): "
            "recognizer graph is unreadable"
        )
    }
    assert recovered.status_code == 200
    assert recovered.json() == {
        "status": "ready",
        "model": "buffalo_l",
        "model_revision": FACE_MODEL_REVISION,
    }
    assert attempts == 2


def test_face_missing_artifact_fails_before_insightface_download(monkeypatch, tmp_path):
    construction = []

    class _FaceAnalysis:
        def __init__(self, **kwargs):
            construction.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "insightface.app",
        SimpleNamespace(FaceAnalysis=_FaceAnalysis),
    )
    app = face_server.build_app(face_server.FaceEmbedConfig(model_root=str(tmp_path)))
    image = Image.new("RGB", (8, 8), (100, 120, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/embed",
            json={"image_b64": base64.b64encode(buffer.getvalue()).decode("ascii")},
        )
    )

    missing = ", ".join(
        str(tmp_path / "models" / "buffalo_l" / filename)
        for filename in FACE_MODEL_FILES
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "face_embed: model buffalo_l load failed (FileNotFoundError): "
            f"face model artifact is incomplete; missing: {missing}"
        )
    }
    assert construction == []


@pytest.mark.parametrize(
    ("server", "path", "payload", "service", "model_id"),
    [
        (
            clap_server,
            "/embed/text",
            {"text": "rain on a roof"},
            "clap_embed",
            "laion/clap-htsat-unfused",
        ),
        (
            face_server,
            "/embed",
            {"image_b64": base64.b64encode(io.BytesIO().getvalue()).decode("ascii")},
            "face_embed",
            "buffalo_l",
        ),
    ],
)
def test_model_load_failure_has_service_model_and_cause(
    monkeypatch,
    server,
    path,
    payload,
    service,
    model_id,
    tmp_path,
):
    app = server.app
    if server is clap_server:

        class _Broken:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise OSError("checkpoint is truncated")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(ClapModel=_Broken, ClapProcessor=_Broken),
        )
    else:
        _face_artifact(tmp_path)

        class _BrokenFaceAnalysis:
            def __init__(self, **kwargs):
                raise OSError("model pack is truncated")

        monkeypatch.setitem(
            sys.modules,
            "insightface.app",
            SimpleNamespace(FaceAnalysis=_BrokenFaceAnalysis),
        )
        image = Image.new("RGB", (8, 8), (100, 120, 140))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = {"image_b64": base64.b64encode(buffer.getvalue()).decode("ascii")}
        app = face_server.build_app(
            face_server.FaceEmbedConfig(model_root=str(tmp_path))
        )

    response = asyncio.run(_request(app, "POST", path, json=payload))

    assert response.status_code == 503
    assert response.json()["detail"].startswith(
        f"{service}: model {model_id} load failed (OSError): "
    )
    assert response.json()["detail"].endswith("is truncated")


def test_clap_inference_failure_has_service_model_and_cause():
    class _BrokenModel:
        def get_text_features(self, **inputs):
            raise RuntimeError("device execution failed")

    clap_server._MODEL = _BrokenModel()
    clap_server._PROCESSOR = lambda **kwargs: {}

    response = asyncio.run(
        _request(clap_server.app, "POST", "/embed/text", json={"text": "rain"})
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "clap_embed: model laion/clap-htsat-unfused inference failed "
            "(RuntimeError): device execution failed"
        )
    }


def test_face_inference_failure_has_service_model_and_cause():
    class _BrokenModel:
        def get(self, image):
            raise RuntimeError("onnx execution failed")

    face_server._MODEL = _BrokenModel()
    image = Image.new("RGB", (8, 8), (100, 120, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = asyncio.run(
        _request(
            face_server.app,
            "POST",
            "/embed",
            json={"image_b64": base64.b64encode(buffer.getvalue()).decode("ascii")},
        )
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "face_embed: model buffalo_l inference failed "
            "(RuntimeError): onnx execution failed"
        )
    }
