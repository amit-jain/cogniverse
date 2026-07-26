import pytest
from cogniverse_cli.modal_inference_config import (
    INFERENCE_SERVICE_SPECS,
    EndpointAuth,
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
        128,
    ),
    "denseon": (
        "lightonai/DenseOn",
        "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
        768,
    ),
    "gliner": (
        "urchade/gliner_medium-v2.1",
        "40ec419335d09393f298636f471328b722c6da9e",
        None,
    ),
    "videoprism_jax": (
        "videoprism_public_v1_base_hf",
        "d481d91b9bf8c9d330d1e526e511a359c799bbe1",
        768,
    ),
    "vllm_llm_student": (
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
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
        "insightface==0.7.3:buffalo_l",
        512,
    ),
}


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


def test_each_service_has_an_independent_scale_to_zero_app():
    specs = tuple(INFERENCE_SERVICE_SPECS.values())

    assert len(specs) == 10
    assert len({spec.modal_app for spec in specs}) == 10
    assert all(
        spec.modal_app == f"cogniverse-{spec.name.replace('_', '-')}" for spec in specs
    )
    assert all(spec.min_containers == 0 for spec in specs)
    assert all(spec.scaledown_window == 300 for spec in specs)


def test_mutable_or_missing_model_revisions_are_rejected():
    for spec in INFERENCE_SERVICE_SPECS.values():
        assert spec.model_revision
        assert spec.model_revision not in {"main", "master", "latest"}


def test_unknown_service_is_an_error():
    with pytest.raises(KeyError, match="unknown inference service 'missing'"):
        get_inference_service_spec("missing")
