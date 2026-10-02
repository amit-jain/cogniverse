"""ROCm inference startup budgets rendered by the Helm chart."""

import bisect
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _rocm_inference_containers() -> dict[str, dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "-f",
            str(CHART_PATH / "values.rocm.yaml"),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )

    containers = {}
    for document in yaml.safe_load_all(result.stdout):
        if document is None or document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if component.startswith("inference-"):
            containers[component.removeprefix("inference-")] = document["spec"][
                "template"
            ]["spec"]["containers"][0]
    return containers


def test_tomoro_rocm_serve_args():
    container = _rocm_inference_containers()["vllm_colpali"]

    assert container["args"] == [
        "serve",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--revision",
        "bf790bd8780b098b86453444632a184bb770be1a",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "1536",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
        "--kv-cache-memory-bytes",
        "1G",
        "--mm-processor-kwargs",
        '{"max_pixels":1048576}',
        "--gpu-memory-utilization",
        "0.18",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "1536",
        "--max-cudagraph-capture-size",
        "512",
    ]


# The Qwen3-VL vision tower merges 2 x 2 patches of 16 px, so one image
# embedding token covers a 32 x 32 pixel square.
MERGED_PATCH_EDGE_PX = 32
# Chat-template tokens around one image placeholder run: the deployed encoder
# reports prompt_tokens == 1031 for a 1024 x 1024 image (1024 image tokens).
IMAGE_TEMPLATE_TOKENS = 7
# Longest golden query is 13 prompt tokens; this leaves room for rewritten
# queries more than twice that long.
QUERY_TOKEN_CEILING = 32


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _vision_items_per_step(
    *,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    image_tokens: int,
    images_per_prompt: int,
) -> int:
    """Maximum-size images vLLM encodes in one step and profiles at startup.

    vLLM 0.23 ``MultiModalBudget._get_max_items`` with chunked prefill off,
    which it is for this CLS-pooled model: the encoder budget is the step's
    token budget floored at one image, and the decoder side admits at most
    ``max_num_seqs`` requests and only as many images as fit the token budget.
    """
    encoder_items = max(max_num_batched_tokens, image_tokens) // image_tokens
    decoder_items = (
        min(max_num_seqs, max_num_batched_tokens // image_tokens) * images_per_prompt
    )
    return max(1, min(encoder_items, decoder_items))


def _cudagraph_capture_sizes(max_size: int) -> list[int]:
    """vLLM 0.23 ``VllmConfig._set_cudagraph_sizes`` default size list."""
    sizes = [size for size in (1, 2, 4) if size <= max_size]
    sizes += list(range(8, min(max_size + 1, 256), 8))
    if max_size >= 256:
        sizes += list(range(256, max_size + 1, 16))
    return sorted(set(sizes))


def test_vision_items_per_step_counts_the_images_vllm_batches():
    """The counter must see what each budget actually admits.

    One image per sequence and 1024 tokens per image throughout; only the
    step budget and the sequence cap vary.
    """

    def items(budget: int, seqs: int) -> int:
        return _vision_items_per_step(
            max_num_batched_tokens=budget,
            max_num_seqs=seqs,
            image_tokens=1024,
            images_per_prompt=1,
        )

    # vLLM's default 8192-token budget on this host, one sequence per step.
    assert items(8192, 1) == 1
    # The same budget with batching turned on encodes eight images at once.
    assert items(8192, 16) == 8
    assert items(4096, 16) == 4
    assert items(2048, 16) == 2
    assert items(2047, 16) == 1
    assert items(1536, 16) == 1


def test_tomoro_step_batches_queries_but_encodes_one_image():
    """Batching text queries must not multiply the vision workspace.

    A step's token budget sets how many images the vision tower encodes at
    once, at startup profiling and at run time alike. The budget stays under
    two maximum-size images, so concurrent ingestion still encodes one image
    per step while sixteen queries share a step.
    """
    args = _rocm_inference_containers()["vllm_colpali"]["args"]

    max_pixels = json.loads(_flag(args, "--mm-processor-kwargs"))["max_pixels"]
    image_tokens = max_pixels // (MERGED_PATCH_EDGE_PX * MERGED_PATCH_EDGE_PX)
    images_per_prompt = json.loads(_flag(args, "--limit-mm-per-prompt"))["image"]
    max_num_seqs = int(_flag(args, "--max-num-seqs"))
    max_num_batched_tokens = int(_flag(args, "--max-num-batched-tokens"))
    max_model_len = int(_flag(args, "--max-model-len"))

    assert image_tokens == 1024
    assert images_per_prompt == 1
    assert max_num_seqs == 16
    assert (
        _vision_items_per_step(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            image_tokens=image_tokens,
            images_per_prompt=images_per_prompt,
        )
        == 1
    )
    # Without chunked prefill vLLM refuses a step budget below the context
    # length, so the two are set together.
    assert max_model_len == max_num_batched_tokens == 1536
    # The largest ingestion request still fits one step.
    assert image_tokens + IMAGE_TEMPLATE_TOKENS == 1031
    assert image_tokens + IMAGE_TEMPLATE_TOKENS <= max_model_len


def test_tomoro_query_steps_run_at_a_fixed_set_of_gemm_sizes():
    """Every query step pads to a captured size, so its GEMM shapes repeat.

    vLLM pads a step of at most ``--max-cudagraph-capture-size`` tokens up to
    the next captured size. A full step of sixteen queries fits under that
    ceiling, so query traffic only ever runs the captured sizes.
    """
    args = _rocm_inference_containers()["vllm_colpali"]["args"]
    max_capture = int(_flag(args, "--max-cudagraph-capture-size"))
    max_num_seqs = int(_flag(args, "--max-num-seqs"))
    sizes = _cudagraph_capture_sizes(max_capture)

    assert max_capture == max_num_seqs * QUERY_TOKEN_CEILING == 512
    # 1, 2, 4; every 8 from 8 to 248; every 16 from 256 to 512.
    assert len(sizes) == 3 + 31 + 17
    assert sizes[:5] == [1, 2, 4, 8, 16]
    assert sizes[-3:] == [480, 496, 512]

    def padded(tokens: int) -> int | None:
        index = bisect.bisect_left(sizes, tokens)
        return sizes[index] if index < len(sizes) else None

    # 512 possible step lengths collapse onto the 51 captured sizes.
    query_steps = range(1, max_num_seqs * QUERY_TOKEN_CEILING + 1)
    assert len({padded(tokens) for tokens in query_steps}) == 51
    assert padded(13) == 16
    assert padded(3 * 13) == 40
    assert padded(16 * 13) == 208
    # A longer step runs at its own length.
    assert padded(max_capture + 1) is None


def test_whisper_rocm_startup_caps_sequences_and_batched_tokens():
    container = _rocm_inference_containers()["vllm_asr"]
    command = container["args"][0]
    serve_command = command[command.index("exec ") + len("exec ") :].replace(
        "\\\n", " "
    )

    assert shlex.split(serve_command) == [
        "vllm",
        "serve",
        "openai/whisper-large-v3-turbo",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--revision",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "--runner",
        "generate",
        "--max-model-len",
        "448",
        "--gpu-memory-utilization",
        "0.04",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
    ]
