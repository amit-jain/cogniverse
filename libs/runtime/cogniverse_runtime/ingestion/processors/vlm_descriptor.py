#!/usr/bin/env python3
"""
VLM Description Generation Step

Generates visual descriptions for keyframes via an OpenAI-compatible
``/v1`` vision chat endpoint (e.g. an in-cluster vLLM vision model).
"""

import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

_VLM_DESCRIBE_PROMPT = (
    "Describe this video frame in detail: the objects, people, actions, "
    "scene setting, and any visible text. Be concise and factual."
)

# Persist progress every N batches (plus once after the loop). Each flush
# re-serializes the ENTIRE growing descriptions dict, so a per-batch flush
# was quadratic write amplification over a long video.
_PROGRESS_FLUSH_EVERY = 10


class VLMDescriptor:
    """Handles VLM description generation for keyframes"""

    def __init__(
        self,
        vlm_endpoint: str,
        batch_size: int = 500,
        timeout: int = 10800,
        vlm_concurrency: int = 8,
    ):
        if "/v1" not in vlm_endpoint:
            raise ValueError(
                "VLM endpoint must be an OpenAI-compatible /v1 URL "
                f"(e.g. http://host:8000/v1), got {vlm_endpoint!r}"
            )
        self.vlm_endpoint = vlm_endpoint
        self.batch_size = batch_size
        self.timeout = timeout  # 3 hours default
        # How many keyframe describe-requests to keep in flight. Concurrent
        # requests are what feed vLLM's continuous batching (there is no
        # single-request multi-image describe API), so this is the throughput
        # lever: raise it on a GPU that can serve a bigger batch, keep it low
        # on a small one.
        self.vlm_concurrency = max(1, vlm_concurrency)
        self._openai_model: str | None = None

        self.logger = logging.getLogger("VLMDescriptor")
        self.logger.info(f"Initialized VLMDescriptor with endpoint: {vlm_endpoint}")
        self.logger.info(f"Batch size: {batch_size}, Timeout: {timeout}s")

    def generate_descriptions(
        self, keyframes_metadata: dict[str, Any], output_dir: Path = None
    ) -> dict[str, Any]:
        """Generate VLM descriptions for keyframes."""
        # Check if keyframes_metadata is empty or doesn't have required data
        if not keyframes_metadata or "video_id" not in keyframes_metadata:
            self.logger.info("No keyframes to generate descriptions for")
            return {"descriptions": {}}

        video_id = keyframes_metadata["video_id"]
        self.logger.info(f"Starting VLM description generation for video: {video_id}")

        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from cogniverse_core.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            descriptions_file = (
                output_manager.get_processing_dir("descriptions") / f"{video_id}.json"
            )
        else:
            # For testing - should migrate tests to use OutputManager
            descriptions_file = output_dir / "descriptions" / f"{video_id}.json"

        keyframes = keyframes_metadata["keyframes"]
        if not keyframes:
            self.logger.warning(f"No keyframes found for video: {video_id}")
            return {}

        self.logger.info(f"Processing {len(keyframes)} keyframes for video: {video_id}")

        # Process in batches
        descriptions = {}
        total_batches = (len(keyframes) + self.batch_size - 1) // self.batch_size
        descriptions_file.parent.mkdir(parents=True, exist_ok=True)

        def _flush_progress():
            with open(descriptions_file, "w") as f:
                json.dump(descriptions, f, indent=2)

        for i in range(0, len(keyframes), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = keyframes[i : i + self.batch_size]

            self.logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} frames)"
            )
            batch_descriptions = self._process_vlm_batch(batch)
            descriptions.update(batch_descriptions)

            if batch_num % _PROGRESS_FLUSH_EVERY == 0 and batch_num < total_batches:
                _flush_progress()

        _flush_progress()

        self.logger.info(
            f"Successfully generated {len(descriptions)} descriptions for video: {video_id}"
        )

        # Return in the expected format for the pipeline
        return {
            "video_id": video_id,
            "descriptions": descriptions,
            "total_descriptions": len(descriptions),
            "created_at": time.time(),
        }

    def _openai_base(self) -> str:
        """The ``/v1`` root of the configured endpoint (accepts either a bare
        ``.../v1`` or a full ``.../v1/chat/completions`` URL)."""
        return self.vlm_endpoint.split("/v1")[0] + "/v1"

    def _resolve_openai_model(self) -> str:
        """Discover the served model id from the vLLM ``/v1/models`` list."""
        if self._openai_model:
            return self._openai_model
        resp = requests.get(f"{self._openai_base()}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError(f"VLM endpoint {self._openai_base()} served no models")
        self._openai_model = data[0]["id"]
        return self._openai_model

    def _describe_one_openai(
        self, keyframe: dict, model: str, chat_url: str
    ) -> tuple[str, str] | None:
        """Describe one keyframe; return (frame_ref, description) or None when
        the frame file is missing."""
        frame_path = Path(keyframe["path"])
        if not frame_path.exists():
            return None
        frame_ref = keyframe.get("frame_id")
        if frame_ref is None:
            frame_ref = keyframe.get("frame_number")
        if frame_ref is None:
            frame_ref = frame_path.stem
        suffix = frame_path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix == "jpg" else suffix
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VLM_DESCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.2,
        }
        resp = requests.post(chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return str(frame_ref), (content or "").strip()

    def _process_vlm_batch(self, keyframes: list[dict]) -> dict[str, str]:
        """Describe each keyframe via an OpenAI-compatible vision chat model.

        Frames are described concurrently (up to ``vlm_concurrency`` in flight)
        so vLLM's continuous batching is fed; a strictly-sequential loop starved
        it. Concurrent requests are the only way to feed the server-side batcher
        — the chat API returns one completion per request, so there is no
        single-request multi-image describe.
        """
        model = self._resolve_openai_model()
        chat_url = f"{self._openai_base()}/chat/completions"
        descriptions: dict[str, str] = {}
        if not keyframes:
            return descriptions

        def _describe(kf: dict):
            # Best-effort per frame: a transient error on one keyframe must not
            # abort the whole video's description stage. Carry the outcome so the
            # caller can keep successes and report the failures by name.
            try:
                return ("ok", self._describe_one_openai(kf, model, chat_url))
            except Exception as exc:  # noqa: BLE001 - recorded, re-raised if total
                return ("err", kf.get("path", "?"), exc)

        if len(keyframes) <= 1:
            results = [_describe(kf) for kf in keyframes]
        else:
            workers = min(self.vlm_concurrency, len(keyframes))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_describe, keyframes))

        failures: list[tuple[str, Exception]] = []
        for r in results:
            if r[0] == "ok":
                pair = r[1]
                if pair is not None:  # None = missing frame file, skip silently
                    descriptions[pair[0]] = pair[1]
            else:
                failures.append((r[1], r[2]))

        if failures:
            self.logger.warning(
                "VLM description failed for %d/%d keyframes: %s",
                len(failures),
                len(keyframes),
                "; ".join(f"{path}: {exc}" for path, exc in failures[:5]),
            )
        # A total outage (errors on every keyframe, nothing salvaged) must not
        # return an empty map the pipeline reads as "no descriptions" success —
        # raise so the description stage reports the failure.
        if failures and not descriptions:
            raise RuntimeError(
                f"VLM description failed for all {len(keyframes)} keyframes; "
                f"first error: {failures[0][1]}"
            )
        self.logger.info(
            f"VLM (vision chat) described {len(descriptions)}/{len(keyframes)} frames"
        )
        return descriptions
