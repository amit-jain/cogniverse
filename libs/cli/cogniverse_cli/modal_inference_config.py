"""Canonical production model contracts for Modal inference services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class EndpointAuth(StrEnum):
    """Authentication scheme required by an inference endpoint."""

    BEARER = "bearer"
    MODAL_PROXY = "modal_proxy"


@dataclass(frozen=True, slots=True)
class InferenceServiceSpec:
    """Immutable deployment and response contract for one inference service."""

    name: str
    model_id: str
    model_revision: str
    output_dimension: int | None
    gpu_candidates: tuple[str, ...]
    requires_hf_token: bool = False
    source_revision: str | None = None
    auth: EndpointAuth = EndpointAuth.BEARER
    modal_object: str = "Inference"
    health_path: str = "/health"
    models_path: str = "/v1/models"
    scaledown_window: int = 300
    min_containers: int = 0

    def __post_init__(self) -> None:
        if not self.model_revision or self.model_revision in {
            "main",
            "master",
            "latest",
        }:
            raise ValueError(
                f"{self.name}: model_revision must identify an immutable artifact"
            )
        if not self.gpu_candidates:
            raise ValueError(f"{self.name}: at least one GPU candidate is required")

    @property
    def modal_app(self) -> str:
        """Stable Modal deployment name for this service."""

        return f"cogniverse-{self.name.replace('_', '-')}"


def _spec(
    name: str,
    model_id: str,
    model_revision: str,
    output_dimension: int | None,
    *gpu_candidates: str,
    requires_hf_token: bool = False,
    source_revision: str | None = None,
) -> InferenceServiceSpec:
    return InferenceServiceSpec(
        name=name,
        model_id=model_id,
        model_revision=model_revision,
        output_dimension=output_dimension,
        gpu_candidates=gpu_candidates,
        requires_hf_token=requires_hf_token,
        source_revision=source_revision,
    )


INFERENCE_SERVICE_SPECS: Mapping[str, InferenceServiceSpec] = MappingProxyType(
    {
        "vllm_colpali": _spec(
            "vllm_colpali",
            "TomoroAI/tomoro-colqwen3-embed-4b",
            "bf790bd8780b098b86453444632a184bb770be1a",
            320,
            "L4",
            "A10",
            "L40S",
        ),
        "colbert_pylate": _spec(
            "colbert_pylate",
            "lightonai/LateOn",
            "c01907b70557ee5c7753680d4819a5cce1674b83",
            128,
            "T4",
            "L4",
        ),
        "code_colbert_pylate": _spec(
            "code_colbert_pylate",
            "lightonai/LateOn-Code-edge",
            "07ef20f406c86badca122464808f4cac2f6e4b25",
            48,
            "T4",
            "L4",
        ),
        "denseon": _spec(
            "denseon",
            "lightonai/DenseOn",
            "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
            768,
            "T4",
            "L4",
        ),
        "gliner": _spec(
            "gliner",
            "urchade/gliner_large-v2.1",
            "abd49a1f1ebc12af1be84d06f6848221cf96dcad",
            None,
            "T4",
            "L4",
        ),
        "videoprism_jax": _spec(
            "videoprism_jax",
            "videoprism_public_v1_base_hf",
            "be719a406d563b66f0ac969e7c94bab8e997c81a",
            768,
            "T4",
            "L4",
            source_revision="d481d91b9bf8c9d330d1e526e511a359c799bbe1",
        ),
        "vllm_llm_student": _spec(
            "vllm_llm_student",
            "google/gemma-4-e4b-it",
            "ee0ef6023621cff504d758262d4e04895a5af4a2",
            None,
            "L4",
            "A10",
            "L40S",
            requires_hf_token=True,
        ),
        "vllm_llm_teacher": _spec(
            "vllm_llm_teacher",
            "Qwen/Qwen3-14B-AWQ",
            "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
            None,
            "L4",
            "A10",
            "L40S",
            requires_hf_token=True,
        ),
        "vllm_asr": _spec(
            "vllm_asr",
            "openai/whisper-large-v3-turbo",
            "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
            None,
            "T4",
            "L4",
        ),
        "clap_embed": _spec(
            "clap_embed",
            "laion/clap-htsat-unfused",
            "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
            512,
            "T4",
            "L4",
        ),
        "face_embed": _spec(
            "face_embed",
            "buffalo_l",
            "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
            512,
            "T4",
            "L4",
        ),
    }
)


def get_inference_service_spec(name: str) -> InferenceServiceSpec:
    """Return the canonical contract for ``name`` or reject unknown services."""

    try:
        return INFERENCE_SERVICE_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown inference service {name!r}") from None
