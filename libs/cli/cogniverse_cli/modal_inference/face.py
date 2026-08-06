"""Authenticated scale-to-zero Modal App for pinned face embeddings."""

import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, ZipInfo

import modal

from cogniverse_cli.modal_inference_config import get_inference_service_spec

_SPEC = get_inference_service_spec("face_embed")
_CACHE_PATH = "/root/.insightface"
_FACE_MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
_FACE_MODEL_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
_FACE_MODEL_ROOT = "/opt/insightface"
_FACE_MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


def _install_face_artifact(
    *,
    url: str,
    expected_sha256: str,
    model_root: str,
    required_files: tuple[str, ...],
) -> None:
    """Download, verify, and atomically install one InsightFace model pack."""

    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".zip") as archive_file:
        with urlopen(url) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                archive_file.write(chunk)
        archive_file.flush()
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "buffalo_l.zip SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        with ZipFile(archive_file.name) as archive:
            members: dict[str, ZipInfo] = {}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                filename = Path(member.filename).name
                if filename in members:
                    raise RuntimeError(
                        f"buffalo_l.zip contains duplicate file {filename!r}"
                    )
                members[filename] = member
            missing = [
                filename for filename in required_files if filename not in members
            ]
            if missing:
                raise RuntimeError(
                    "buffalo_l.zip is missing required files: " + ", ".join(missing)
                )

            models_dir = Path(model_root) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_dir = models_dir / "buffalo_l"
            if model_dir.exists():
                raise FileExistsError(
                    f"face model artifact already exists: {model_dir}"
                )
            staging_dir = Path(
                tempfile.mkdtemp(prefix=".buffalo_l-", dir=str(models_dir))
            )
            try:
                for filename in required_files:
                    with (
                        archive.open(members[filename]) as source,
                        (staging_dir / filename).open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
                staging_dir.rename(model_dir)
            except BaseException as install_error:
                try:
                    shutil.rmtree(staging_dir)
                except BaseException as cleanup_error:
                    install_error.add_note(
                        f"failed to remove face model staging directory {staging_dir} "
                        f"({type(cleanup_error).__name__}): {cleanup_error}"
                    )
                raise


def _build_image():
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "ca-certificates",
            "g++",
            "libgomp1",
            "libjpeg62-turbo",
            "libpng16-16",
        )
        .pip_install(
            "fastapi==0.135.3",
            "httpx==0.28.1",
            "insightface==0.7.3",
            "numpy==1.26.4",
            "onnxruntime-gpu==1.20.1",
            "pillow==11.0.0",
            "pydantic==2.13.0",
        )
        .env(
            {
                "FACE_EMBED_CTX_ID": "0",
                "FACE_EMBED_MODEL": _SPEC.model_id,
                "FACE_EMBED_MODEL_REVISION": _SPEC.model_revision,
                "FACE_EMBED_MODEL_ROOT": _FACE_MODEL_ROOT,
                "INSIGHTFACE_HOME": _FACE_MODEL_ROOT,
            }
        )
        .run_function(
            _install_face_artifact,
            kwargs={
                "url": _FACE_MODEL_URL,
                "expected_sha256": _FACE_MODEL_SHA256,
                "model_root": _FACE_MODEL_ROOT,
                "required_files": _FACE_MODEL_FILES,
            },
        )
        .add_local_python_source(
            "cogniverse_cli.modal_inference",
            "cogniverse_cli.modal_inference_config",
            copy=True,
        )
    )


_image = _build_image()
_volume = modal.Volume.from_name(
    "cogniverse-insightface-cache",
    create_if_missing=True,
)
_secrets = [
    modal.Secret.from_name(
        "cogniverse-inference-api-key",
        required_keys=["COGNIVERSE_INFERENCE_API_KEY"],
    )
]

app = modal.App(_SPEC.modal_app)


@app.function(
    image=_image,
    gpu=list(_SPEC.gpu_candidates),
    volumes={_CACHE_PATH: _volume},
    secrets=_secrets,
    min_containers=_SPEC.min_containers,
    scaledown_window=_SPEC.scaledown_window,
    timeout=900,
    startup_timeout=900,
    serialized=True,
    name=_SPEC.modal_object,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def inference():
    from cogniverse_cli.modal_inference.servers.face import FaceEmbedConfig, build_app
    from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app

    production_app = build_app(
        FaceEmbedConfig(
            model_name=_SPEC.model_id,
            model_revision=_SPEC.model_revision,
            model_root=_FACE_MODEL_ROOT,
            ctx_id=0,
        )
    )
    return build_authenticated_asgi_app(
        production_app,
        model_id=_SPEC.model_id,
        model_revision=_SPEC.model_revision,
    )
