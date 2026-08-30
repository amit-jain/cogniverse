from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.inference_specs import get_inference_service_spec


def _gemma_config(endpoint) -> LLMEndpointConfig:
    headers = dict(endpoint.headers)
    assert set(headers) == {"Authorization"}
    authorization = headers["Authorization"]
    scheme, separator, token = authorization.partition(" ")
    assert (scheme, separator, bool(token), token == token.strip()) == (
        "Bearer",
        " ",
        True,
        True,
    )
    return LLMEndpointConfig(
        model=f"openai/{endpoint.model_id}",
        api_base=f"{endpoint.base_url}/v1",
        api_key=token,
        temperature=0.0,
        max_tokens=20,
        seed=0,
        request_timeout=60,
        num_retries=0,
    )


@pytest.mark.e2e
@pytest.mark.requires_inference("vllm_llm_student")
@pytest.mark.requires_modal_inference("vllm_llm_student")
def test_resolved_gemma_runs_exact_concurrent_production_requests(
    resolved_inference_endpoints,
):
    spec = get_inference_service_spec("vllm_llm_student")
    endpoint = resolved_inference_endpoints[spec.name]
    config = _gemma_config(endpoint)
    secret = config.api_key

    assert (
        endpoint.service,
        endpoint.provider,
        endpoint.model_id,
        endpoint.model_revision,
    ) == (
        "vllm_llm_student",
        "modal",
        "google/gemma-4-e4b-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
    )
    assert secret
    assert secret not in repr(endpoint)
    assert config.to_dict()["api_key"] == "***"

    lm = create_dspy_lm(config)

    def answer(index: int) -> str:
        expected = f"radium=Ra;request={index}"
        prompt = (
            "The chemical symbol for radium is Ra. Reply with exactly "
            f"{expected} and no other characters."
        )
        return lm(prompt, cache=False)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = tuple(pool.map(answer, range(4)))

    assert answers == (
        "radium=Ra;request=0",
        "radium=Ra;request=1",
        "radium=Ra;request=2",
        "radium=Ra;request=3",
    )
