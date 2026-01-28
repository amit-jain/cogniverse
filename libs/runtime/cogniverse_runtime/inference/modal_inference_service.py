#!/usr/bin/env python3
"""
Modal Inference Service - Fixed Version

A simple, general-purpose inference service that can be deployed on Modal.
This service loads models dynamically and provides a clean API for text generation.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import modal

# Define the Modal app
app = modal.App("general-inference-service")

# Shared volume for model cache
volume = modal.Volume.from_name("model-cache", create_if_missing=True)


def download_model():
    """Download the model during image build for faster startup."""
    import os

    from huggingface_hub import snapshot_download

    model_id = os.getenv("DEFAULT_MODEL", "google/gemma-3-1b-it")
    hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN", ""))
    print(f"📥 Downloading model: {model_id}")

    try:
        snapshot_download(
            model_id,
            local_dir=f"/model-cache/{model_id}",
            ignore_patterns=["*.bin", "*.pt"],  # vLLM uses safetensors
            token=hf_token if hf_token else None,
        )
        print(f"✅ Model {model_id} downloaded successfully")
    except Exception as e:
        print(f"⚠️ Model download failed (will download on first use): {e}")


# Base image with ML dependencies
# Using CUDA base image for better compatibility
cuda_version = "12.4.0"  # Compatible with most GPUs
flavor = "runtime"  # Lighter than devel
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("build-essential", "gcc", "g++", "cmake")  # Install C compiler
    .pip_install(
        [
            "vllm==0.9.2",
            "transformers",
            "accelerate",
            "pydantic>=2.0",
            "einops",
            "sentencepiece",
            "protobuf",
            "hf-transfer",
            "huggingface_hub",
            "numpy<2",
            "fastapi",
            "uvicorn",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable faster downloads
            "HF_HOME": "/model-cache",  # Set HF cache directory
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN", ""),
        }
    )
    # Pre-download model during image build
    .run_function(download_model, volumes={"/model-cache": volume})
)

# ==================== Configuration ====================

# Default configuration (can be overridden via environment variables)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "HuggingFaceTB/SmolLM3-3B")
DEFAULT_GPU = os.getenv("DEFAULT_GPU", "A100-80GB")
DEFAULT_MEMORY = int(os.getenv("DEFAULT_MEMORY", "64000"))  # Increased for A100-80GB
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "600"))  # Increased for startup


# ==================== Shared Model Logic ====================

# Global model cache
_model_cache = {}


def _generate_text_internal(
    prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    stop: List[str],
    system_prompt: str = "",
) -> Dict[str, Any]:
    """Internal text generation function that can be called by multiple endpoints."""
    from vllm import LLM, SamplingParams

    start_time = time.time()

    if not prompt:
        return {"error": "prompt is required", "status": "error"}

    # Load model if not cached
    if model_id not in _model_cache:
        print(f"🔄 Loading model: {model_id}")
        try:
            _model_cache[model_id] = LLM(
                model=model_id,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_num_seqs=32,
                dtype="float16",
                download_dir="/model-cache",
            )
            print(f"✅ Model {model_id} loaded successfully")
        except Exception as e:
            return {
                "error": f"Failed to load model {model_id}: {str(e)}",
                "status": "error",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

    model = _model_cache[model_id]

    # Prepare prompt with system prompt if provided
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    try:
        # Generate with VLLM
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, stop=stop or None
        )

        outputs = model.generate([full_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Get token counts
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        return {
            "text": generated_text,
            "model": model_id,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "status": "success",
        }

    except Exception as e:
        return {
            "error": f"Generation failed: {str(e)}",
            "status": "error",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }


# ==================== Model Inference ====================


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    memory=DEFAULT_MEMORY,
    timeout=DEFAULT_TIMEOUT,
    keep_warm=1,  # Keep warm for fast response
    container_idle_timeout=300,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """
    Spawns a vLLM instance that serves an OpenAI-compatible API.
    """
    import subprocess

    # Build vLLM command following Modal example pattern
    cmd = [
        "vllm",
        "serve",
        DEFAULT_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "float16",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "1.0",
        "--download-dir",
        "/model-cache",
    ]

    print(f"🚀 Starting vLLM server with model: {DEFAULT_MODEL}")
    cmd_str = " ".join(cmd)
    print(f"📍 Command: {cmd_str}")

    # Start the vLLM server process with HF token and compatibility env vars
    env = os.environ.copy()
    env["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    env["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    # vLLM 0.9.2 compatibility environment variables
    env["VLLM_USE_SPAWN"] = "1"
    env["PYTHONMULTIPROCESSING_START_METHOD"] = "spawn"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["VLLM_DISABLE_TRITON_AUTOTUNE"] = "1"  # Disable Triton autotune for stability
    env["VLLM_USE_V1"] = "0"  # Force disable v1 engine, use legacy engine
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # Use spawn for multiprocessing
    env["CC"] = "gcc"  # Specify C compiler for Triton
    env["CXX"] = "g++"  # Specify C++ compiler for Triton
    subprocess.Popen(cmd_str, shell=True, env=env)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    memory=DEFAULT_MEMORY,
    timeout=DEFAULT_TIMEOUT,
    keep_warm=1,
    allow_concurrent_inputs=10,
    container_idle_timeout=300,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.web_endpoint(method="POST", label="generate")
def generate_text(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    General text generation endpoint.

    Request format:
    {
        "prompt": "text prompt",
        "model": "model-id (optional, defaults to configured model)",
        "temperature": 0.7,
        "max_tokens": 100,
        "stop": ["\\n"],
        "system_prompt": "optional system prompt",
        "stream": false
    }

    Response format:
    {
        "text": "generated text",
        "model": "model-id",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        },
        "latency_ms": 234.5,
        "status": "success"
    }
    """
    # Extract parameters
    prompt = request.get("prompt", "")
    model_id = request.get("model", DEFAULT_MODEL)
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 100)
    stop = request.get("stop", [])
    system_prompt = request.get("system_prompt", "")

    # Use internal generation function
    return _generate_text_internal(
        prompt, model_id, temperature, max_tokens, stop, system_prompt
    )


# ==================== OpenAI Compatible Endpoints ====================


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    memory=DEFAULT_MEMORY,
    timeout=DEFAULT_TIMEOUT,
    keep_warm=1,
    allow_concurrent_inputs=10,
    container_idle_timeout=300,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.web_endpoint(method="POST", label="chat-completions")
def chat_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI-compatible chat completions endpoint.

    This allows the service to be used as a drop-in replacement
    for OpenAI API in many applications.
    """
    # Convert chat format to simple prompt
    messages = request.get("messages", [])
    if not messages:
        return {"error": "messages array is required", "status": "error"}

    # Build prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            prompt_parts.insert(0, content)  # System prompt goes first
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")  # Prompt for response
    full_prompt = "\n\n".join(prompt_parts)

    # Use internal generation function
    result = _generate_text_internal(
        prompt=full_prompt,
        model_id=request.get("model", DEFAULT_MODEL),
        temperature=request.get("temperature", 0.7),
        max_tokens=request.get("max_tokens", 100),
        stop=request.get("stop", []),
        system_prompt="",
    )

    # Convert to OpenAI format
    if result.get("status") == "success":
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": result.get("usage", {}),
        }
    else:
        return result  # Return error as-is


# ==================== Service Management ====================


@app.function(
    image=image,
    keep_warm=1,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.web_endpoint(method="GET", label="health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "general-inference-service",
        "default_model": DEFAULT_MODEL,
        "default_gpu": DEFAULT_GPU,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "generate": "/generate",
            "chat": "/chat-completions",
            "health": "/health",
            "models": "/models",
        },
    }


@app.function(
    image=image,
    keep_warm=1,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.web_endpoint(method="GET", label="models")
def list_models() -> Dict[str, Any]:
    """List available and loaded models."""
    loaded_models = list(_model_cache.keys()) if "_model_cache" in globals() else []

    return {
        "default_model": DEFAULT_MODEL,
        "loaded_models": loaded_models,
        "recommended_models": [
            {
                "id": "google/gemma-3-1b-it",
                "description": "Small, fast model for routing tasks",
                "size": "1B",
            },
            {
                "id": "google/gemma-2-2b-it",
                "description": "Balanced model for general tasks",
                "size": "2B",
            },
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "description": "High quality small model",
                "size": "3B",
            },
        ],
    }


# ==================== CLI Interface ====================


@app.function(image=image)
def test_generation(
    prompt: str = "Hello, how are you?",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 100,
):
    """Test function for local runs."""
    result = _generate_text_internal(
        prompt=prompt,
        model_id=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=[],
        system_prompt="",
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    # Local testing
    test_generation.local()
