"""Real ``vllm_omni_embed`` (EmbeddingGemma 2 on vLLM): the serving contract.

Resolves the cluster's service and pins what the production client relies on:

- the server samples video at 1 fps: a 6 s chunk costs exactly twice a 3 s
  one;
- every content part the client sends is accepted and lands in one space with
  the query text: a chunk of a snow-shoveling news clip is nearer to a query
  about snow shoveling than to an unrelated one, for each modality;
- production query vectors, and the bare audio window and video chunk
  documents ingestion sends, match the Sentence-Transformers reference
  recorded on CPU by ``scripts/record_model_references.py`` (cosine >= 0.99);
- a request past the 8192-token context surfaces the server's 400.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.models.model_loaders import (
    ModelLoaderFactory,
    RemoteInferenceClient,
    _extract_segment_b64,
)
from cogniverse_core.query.encoders import EmbeddingGemma2QueryEncoder
from tests.utils.model_references import (
    load_reference,
    reference_embedding,
    reference_media_embedding,
)

pytestmark = [
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not installed",
    ),
]

MODEL = "google/embeddinggemma-2"
SNOW_CLIP = Path("data/testset/evaluation/sample_videos/v_-IMXSEIabMM.mp4")
NEAR = "people shoveling snow after a snowstorm"
FAR = "a dog catching a frisbee on a sunny beach"


@pytest.fixture(scope="module")
def omni_url(remote_inference):
    return remote_inference.resolve("vllm_omni_embed").base_url


def _wrapper(url: str, modalities: list[str]):
    loader = ModelLoaderFactory.create_loader(
        MODEL,
        {
            "model_loader": "embeddinggemma2",
            "remote_inference_url": url,
            "modalities": modalities,
        },
    )
    return loader.load_model()[0]


def _prompt_tokens(url: str, content: list[dict]) -> int:
    response = requests.post(
        f"{url}/v1/embeddings",
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
        timeout=600,
    )
    assert response.status_code == 200, response.text[:500]
    return response.json()["usage"]["prompt_tokens"]


def test_the_service_serves_the_pinned_model(omni_url):
    response = requests.get(f"{omni_url}/v1/models", timeout=30)

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == [MODEL]


def test_video_is_sampled_at_one_frame_per_second(omni_url):
    prompt = [{"type": "text", "text": "snow"}]

    def video_tokens(seconds: float) -> int:
        video = {
            "type": "video_url",
            "video_url": {
                "url": "data:video/mp4;base64,"
                + _extract_segment_b64(SNOW_CLIP, 0.0, seconds)
            },
        }
        return _prompt_tokens(omni_url, [*prompt, video]) - _prompt_tokens(
            omni_url, prompt
        )

    # A fixed frame count would cost the same for both spans; one frame per
    # second makes the 6 s chunk exactly twice the 3 s one. Each frame of
    # this 320x240 clip is 120 soft tokens plus its begin and end tokens.
    assert (video_tokens(3.0), video_tokens(6.0)) == (3 * 122, 6 * 122)


@pytest.mark.parametrize(
    "modalities", (["video", "audio"], ["video"], ["audio"]), ids="+".join
)
def test_each_modality_lands_in_the_query_space(omni_url, modalities):
    wrapper = _wrapper(omni_url, modalities)
    if modalities == ["audio"]:
        chunk = wrapper.embed_audio_window(SNOW_CLIP, 0.0, 10.0)
    else:
        chunk = wrapper.process_video_segment(SNOW_CLIP, 0.0, 6.0)["embeddings_np"]
    near, far = wrapper.embed_text(NEAR), wrapper.embed_text(FAR)

    assert chunk.shape == (768,)
    assert float(np.linalg.norm(chunk)) == pytest.approx(1.0, abs=1e-5)
    assert float(chunk @ near) > float(chunk @ far)


def test_query_vectors_match_the_sentence_transformers_reference(omni_url):
    reference = load_reference("embeddinggemma2")
    assert (reference["model"], reference["prompt"], reference["device"]) == (
        MODEL,
        "task: search result | query: ",
        "cpu",
    )
    encoder = EmbeddingGemma2QueryEncoder(omni_url)

    def cosine(text: str) -> float:
        # The bf16 reference is normalized to within 0.4 %, so renormalize.
        expected = reference_embedding(reference, text, is_query=True)
        return round(
            float(encoder.encode(text) @ (expected / np.linalg.norm(expected))), 4
        )

    cosines = {case["text"]: cosine(case["text"]) for case in reference["cases"]}

    assert len(cosines) == 6
    assert {text: cosine >= 0.99 for text, cosine in cosines.items()} == {
        text: True for text in cosines
    }, cosines


@pytest.mark.parametrize("media", ("audio", "video"))
def test_media_documents_match_the_sentence_transformers_reference(omni_url, media):
    reference = load_reference("embeddinggemma2")
    case, expected = reference_media_embedding(reference, media)
    source = Path(case["source"])
    assert source == SNOW_CLIP
    wrapper = _wrapper(omni_url, [media])
    if media == "audio":
        served = wrapper.embed_audio_window(source, case["start"], case["end"])
    else:
        served = wrapper.process_video_segment(source, case["start"], case["end"])[
            "embeddings_np"
        ]

    # A task prefix on the media moves the vector to cosine 0.93-0.99.
    assert round(float(served @ (expected / np.linalg.norm(expected))), 4) >= 0.999


def test_a_request_past_the_context_is_refused_with_the_server_reason(omni_url):
    client = RemoteInferenceClient(omni_url)

    with pytest.raises(ValueError) as caught:
        client.embed_messages([{"type": "text", "text": "snow " * 9000}], timeout=120.0)

    assert str(caught.value).startswith(
        f"vllm_omni_embed at {omni_url} rejected the request (400): "
    )
