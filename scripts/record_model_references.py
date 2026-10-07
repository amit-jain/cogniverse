#!/usr/bin/env python3
"""Record the reference model outputs the parity tests compare served models to.

Tests never load a model in-process (tests/fixtures/no_local_models.py refuses
it). The parity tests instead compare what the cluster serves against outputs
recorded here, once, from each model's own reference library at the pinned
revision, on CPU:

- ``lateon.json``: per-token matrices from ``pylate.models.ColBERT`` for
  ``lightonai/LateOn`` (``colbert_pylate``), query and document side;
- ``denseon.json``: the normalized ``sentence_transformers`` vector for
  ``lightonai/DenseOn`` (``denseon``) of a ``document: `` prompted text;
- ``face_embed.json``: InsightFace ``FaceAnalysis`` faces (box, normalized
  ArcFace vector, detection score) for the frames in
  ``tests/fixtures/model_references/face_embed/``, from the full Buffalo_L
  pack with ONNX Runtime's default sessions. InsightFace is not a project
  dependency; record it inside the face-embed image:

    docker run --rm -u "$(id -u)" -v "$PWD:/repo" -w /repo \
        cogniverse/face-embed:<tag> python scripts/record_model_references.py face_embed

Each file records the model, revision, library versions, device and inputs
beside the outputs. Regenerate after a pinned revision changes:

    uv run python scripts/record_model_references.py lateon denseon
"""

from __future__ import annotations

import os

# Hide every accelerator before torch is imported, so the models load on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "model_references"

LATEON_CASES = (
    ("what is a vector database", True),
    ("Vespa stores token embeddings as tensor<bfloat16>(token{}, v[128]).", False),
)
DENSEON_TEXTS = ("Vespa is a vector database for low-latency retrieval.",)
DENSEON_PROMPT = "document: "


def _spec(service: str):
    sys.path.insert(0, str(REPO_ROOT / "libs" / "foundation"))
    from cogniverse_foundation.inference_specs import get_inference_service_spec

    return get_inference_service_spec(service)


def _require_cpu(model) -> str:
    import torch

    if torch.cuda.is_available():
        raise SystemExit("an accelerator is visible; references are recorded on CPU")
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise SystemExit(f"model parameters are on {sorted(devices)}, not cpu")
    return "cpu"


def _write(name: str, payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def record_lateon() -> None:
    import numpy as np
    import pylate.models as pylate_models

    spec = _spec("colbert_pylate")
    model = pylate_models.ColBERT(
        spec.model_id, device="cpu", revision=spec.model_revision
    )
    device = _require_cpu(model)
    cases = []
    for text, is_query in LATEON_CASES:
        matrix = np.asarray(
            model.encode([text], is_query=is_query)[0], dtype=np.float32
        )
        cases.append({"text": text, "is_query": is_query, "embedding": matrix.tolist()})
    _write(
        "lateon.json",
        {
            "service": spec.name,
            "model": spec.model_id,
            "revision": spec.model_revision,
            "library": {"pylate": version("pylate"), "torch": version("torch")},
            "device": device,
            "recorded_by": "scripts/record_model_references.py",
            "cases": cases,
        },
    )


def record_denseon() -> None:
    import numpy as np
    import sentence_transformers

    spec = _spec("denseon")
    model = sentence_transformers.SentenceTransformer(
        spec.model_id, device="cpu", revision=spec.model_revision
    )
    device = _require_cpu(model)
    cases = []
    for text in DENSEON_TEXTS:
        vector = np.asarray(
            model.encode([f"{DENSEON_PROMPT}{text}"], normalize_embeddings=True)[0],
            dtype=np.float32,
        )
        cases.append({"text": text, "is_query": False, "embedding": vector.tolist()})
    _write(
        "denseon.json",
        {
            "service": spec.name,
            "model": spec.model_id,
            "revision": spec.model_revision,
            "library": {
                "sentence_transformers": version("sentence-transformers"),
                "torch": version("torch"),
            },
            "device": device,
            "prompt": DENSEON_PROMPT,
            "normalized": True,
            "recorded_by": "scripts/record_model_references.py",
            "cases": cases,
        },
    )


FACE_FRAMES_DIR = OUTPUT_DIR / "face_embed"


def record_face_embed() -> None:
    import cv2
    import insightface
    import onnxruntime
    from insightface.app import FaceAnalysis

    sys.path.insert(
        0, str(REPO_ROOT / "libs/cli/cogniverse_cli/modal_inference/servers")
    )
    from face import FACE_MODEL_NAME, FACE_MODEL_REVISION  # noqa: PLC0415

    model_root = os.environ.get("FACE_EMBED_MODEL_ROOT", "/opt/insightface")
    app = FaceAnalysis(name=FACE_MODEL_NAME, root=model_root)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    cases = []
    for frame in sorted(FACE_FRAMES_DIR.glob("*.jpg")):
        faces = sorted(app.get(cv2.imread(str(frame))), key=lambda f: tuple(f.bbox))
        cases.append(
            {
                "frame": frame.name,
                "faces": [
                    {
                        "bbox": [int(c) for c in face.bbox.astype(int).tolist()],
                        "vec": [float(v) for v in face.normed_embedding.tolist()],
                        "det_score": float(face.det_score),
                    }
                    for face in faces
                ],
            }
        )
    _write(
        "face_embed.json",
        {
            "service": "face_embed",
            "model": FACE_MODEL_NAME,
            "revision": FACE_MODEL_REVISION,
            "library": {
                "insightface": insightface.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
            "device": "cpu",
            "recorded_by": "scripts/record_model_references.py",
            "cases": cases,
        },
    )


RECORDERS = {
    "lateon": record_lateon,
    "denseon": record_denseon,
    "face_embed": record_face_embed,
}


def main() -> None:
    for name in sys.argv[1:] or ("lateon", "denseon"):
        RECORDERS[name]()


if __name__ == "__main__":
    main()
