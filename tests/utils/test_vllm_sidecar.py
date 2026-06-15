"""Unit tests for the vLLM sidecar serving-arg defaults.

``_merge_serve_args`` is what keeps the test sidecars serving the SAME
config the deploy chart applies — in particular the qwen3_vl
``--limit-mm-per-prompt`` guard, without which vLLM's startup profiler
allocates a worst-case video attention buffer and OOMs.
"""

from tests.utils.vllm_sidecar import _merge_serve_args

TOMORO = "TomoroAI/tomoro-colqwen3-embed-4b"
LATEON = "lightonai/LateOn"


def test_tomoro_gets_gpu_mem_and_mm_limit_defaults():
    assert _merge_serve_args(TOMORO, ["--runner", "pooling"]) == [
        "--runner",
        "pooling",
        "--gpu-memory-utilization",
        "0.10",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ]


def test_explicit_mm_limit_is_not_duplicated():
    out = _merge_serve_args(TOMORO, ["--limit-mm-per-prompt", '{"video":0,"image":2}'])
    assert out.count("--limit-mm-per-prompt") == 1
    assert out == [
        "--limit-mm-per-prompt",
        '{"video":0,"image":2}',
        "--gpu-memory-utilization",
        "0.10",
    ]


def test_explicit_gpu_mem_kept_and_mm_limit_still_injected():
    assert _merge_serve_args(TOMORO, ["--gpu-memory-utilization", "0.20"]) == [
        "--gpu-memory-utilization",
        "0.20",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ]


def test_non_qwen3_model_gets_no_mm_limit():
    out = _merge_serve_args(LATEON, ["--runner", "pooling"])
    assert "--limit-mm-per-prompt" not in out
    assert out == ["--runner", "pooling", "--gpu-memory-utilization", "0.10"]


def test_model_name_match_is_case_insensitive():
    out = _merge_serve_args("TomoroAI/Tomoro-ColQwen3-Embed-4B", [])
    assert out[out.index("--limit-mm-per-prompt") + 1] == '{"video":0,"image":1}'
