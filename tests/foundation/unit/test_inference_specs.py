import pytest

from cogniverse_foundation.inference_specs import (
    INFERENCE_SERVICE_SPECS,
    EndpointAuth,
    InferenceServiceSpec,
    get_inference_service_spec,
)

EXPECTED_MODELS = {
    "vllm_colpali": (
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "bf790bd8780b098b86453444632a184bb770be1a",
        320,
    ),
    "colbert_pylate": (
        "lightonai/LateOn",
        "c01907b70557ee5c7753680d4819a5cce1674b83",
        128,
    ),
    "code_colbert_pylate": (
        "lightonai/LateOn-Code-edge",
        "07ef20f406c86badca122464808f4cac2f6e4b25",
        48,
    ),
    "denseon": (
        "lightonai/DenseOn",
        "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
        768,
    ),
    "gliner": (
        "urchade/gliner_large-v2.1",
        "abd49a1f1ebc12af1be84d06f6848221cf96dcad",
        None,
    ),
    "videoprism_jax": (
        "videoprism_public_v1_base_hf",
        "be719a406d563b66f0ac969e7c94bab8e997c81a",
        768,
    ),
    "vllm_llm_student": (
        "google/gemma-4-e4b-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        None,
    ),
    "vllm_llm_teacher": (
        "Qwen/Qwen3-14B-AWQ",
        "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        None,
    ),
    "vllm_asr": (
        "openai/whisper-large-v3-turbo",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        None,
    ),
    "clap_embed": (
        "laion/clap-htsat-unfused",
        "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        512,
    ),
    "face_embed": (
        "buffalo_l",
        "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        512,
    ),
}

EXPECTED_HF_TOKEN_SERVICES = {"vllm_llm_student", "vllm_llm_teacher"}


def test_definitions_pin_every_production_model_contract():
    observed = {
        name: (spec.model_id, spec.model_revision, spec.output_dimension)
        for name, spec in INFERENCE_SERVICE_SPECS.items()
    }

    assert observed == EXPECTED_MODELS


def test_tomoro_definition_retains_the_production_encoder_dimension():
    spec = get_inference_service_spec("vllm_colpali")

    assert spec.model_id == "TomoroAI/tomoro-colqwen3-embed-4b"
    assert spec.output_dimension == 320
    assert spec.auth is EndpointAuth.BEARER
    assert spec.min_containers == 0


def test_teacher_definition_pins_the_production_chat_contract():
    spec = get_inference_service_spec("vllm_llm_teacher")

    assert spec == InferenceServiceSpec(
        name="vllm_llm_teacher",
        model_id="Qwen/Qwen3-14B-AWQ",
        model_revision="31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        output_dimension=None,
        gpu_candidates=("H100", "A100-80GB", "L40S"),
        requires_hf_token=True,
    )
    assert spec.boot_deadline_seconds == 600.0


def test_exact_hf_token_services_are_the_chat_models():
    observed = {
        name for name, spec in INFERENCE_SERVICE_SPECS.items() if spec.requires_hf_token
    }

    assert observed == EXPECTED_HF_TOKEN_SERVICES


def test_videoprism_definition_pins_source_and_checkpoint_independently():
    spec = get_inference_service_spec("videoprism_jax")

    assert spec.source_revision == "d481d91b9bf8c9d330d1e526e511a359c799bbe1"
    assert spec.model_revision == "be719a406d563b66f0ac969e7c94bab8e997c81a"


def test_each_service_has_an_independent_scale_to_zero_app():
    specs = tuple(INFERENCE_SERVICE_SPECS.values())

    assert len(specs) == 11
    assert len({spec.modal_app for spec in specs}) == 11
    assert all(
        spec.modal_app == f"cogniverse-{spec.name.replace('_', '-')}" for spec in specs
    )
    assert all(spec.min_containers == 0 for spec in specs)
    assert all(spec.scaledown_window == 300 for spec in specs)
    assert all(spec.boot_deadline_seconds == 600.0 for spec in specs)


def test_mutable_or_missing_model_revisions_are_rejected():
    for spec in INFERENCE_SERVICE_SPECS.values():
        assert spec.model_revision
        assert spec.model_revision not in {"main", "master", "latest"}


def test_unknown_service_is_an_error():
    with pytest.raises(KeyError, match="unknown inference service 'missing'"):
        get_inference_service_spec("missing")


# Decode is memory-bandwidth bound, so the chat models' throughput tracks GPU
# bandwidth almost linearly. Measured on a deployed student: L4 (300 GB/s) served
# 21-24 tok/s, which is no better than the local APU. H100 (3350 GB/s) and
# A100-80GB (2039 GB/s) are the tiers that make offloading the compiles worthwhile.
_HIGH_BANDWIDTH_GPUS = ("H100", "A100-80GB")


def test_chat_models_prefer_a_high_bandwidth_gpu():
    chat_services = sorted(
        name for name, spec in INFERENCE_SERVICE_SPECS.items() if spec.requires_hf_token
    )

    assert chat_services == ["vllm_llm_student", "vllm_llm_teacher"]
    for name in chat_services:
        candidates = get_inference_service_spec(name).gpu_candidates
        assert candidates[0] in _HIGH_BANDWIDTH_GPUS, name
        assert candidates == ("H100", "A100-80GB", "L40S"), name
