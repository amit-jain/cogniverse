"""
LLM-as-Judge Evaluators for Video Retrieval

Provides three types of LLM-based evaluation:
1. Reference-free: Evaluates query-result relevance without ground truth
2. Reference-based: Compares results against ground truth from database
3. Hybrid: Combines both approaches for comprehensive evaluation
"""

import base64
import logging
from pathlib import Path

import httpx

from cogniverse_foundation.config.llm_factory import resolve_inference_api_key

logger = logging.getLogger(__name__)


class LLMJudgeCore:
    """Base class for LLM judge evaluators.

    Posts to any OAI-compatible ``/v1/chat/completions`` endpoint via
    ``httpx`` — provider-agnostic. Multimodal images are encoded as
    OpenAI-format ``image_url`` parts with ``data:image/...;base64,`` URIs,
    which all major OAI-compat servers (vLLM, LM Studio, llama.cpp's
    server, etc.) accept.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
    ):
        """
        Initialize LLM judge.

        Args:
            model_name: Model id to send in the chat-completions request.
                Must come from config — callers must not rely on a
                hardcoded default, which would silently mask
                ``evaluators.llm_judge.model`` being out of sync.
            base_url: Base URL for the OAI-compat LM endpoint. The
                ``/v1/chat/completions`` path is appended automatically;
                a trailing ``/v1`` on the base is stripped first so callers
                can pass either form.
            api_key: Explicit bearer for ``base_url``. None resolves the
                shared inference bearer exactly as ``create_dspy_lm`` does.
        """
        self.model_name = model_name
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
        self.base_url = normalized
        self.api_key = resolve_inference_api_key(self.base_url, api_key)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None,
        encoded_images: list[str],
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if encoded_images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img_b64 in encoded_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list | None = None,
    ) -> str:
        """Call the configured LM via OAI-compat chat-completions.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            images: List of image file paths to load and encode for
                multimodal evaluation. Files that don't exist are
                logged and skipped.

        Returns:
            LLM response text (empty string on transport failure — the
            error is logged but doesn't propagate so a single judge
            failure doesn't abort the whole evaluation batch).
        """
        encoded_images: list[str] = []
        if images:
            for img_path in images:
                if isinstance(img_path, str) and Path(img_path).exists():
                    with open(img_path, "rb") as img_file:
                        encoded_images.append(
                            base64.b64encode(img_file.read()).decode("utf-8")
                        )
                else:
                    logger.warning(f"Image not found or invalid: {img_path}")

        messages = self._build_messages(prompt, system_prompt, encoded_images)
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": self.model_name, "messages": messages}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            logger.error(f"LLM call failed: {e}")
            return f"Evaluation failed: {str(e)}"

    def _extract_score_from_response(self, response: str) -> tuple[float | None, str]:
        """
        Extract numerical score and explanation from LLM response

        Args:
            response: LLM response text

        Returns:
            Tuple of (score, explanation). ``score`` is ``None`` when the
            response carries no parseable score (an LM transport failure or a
            reply with no rating) — distinct from a real ``0.5`` so callers can
            skip a non-judgement instead of treating it as neutral.
        """
        import re

        # Try to extract score from response. Each pattern carries its
        # denominator: /100 must not be read as /10 (85/100 is 0.85, not 1.0),
        # and a leading minus is captured so "-3/10" clamps to 0.0 rather than
        # being read as a positive 0.3. ``None`` uses the 0-10-vs-0-1 heuristic.
        score_patterns = [
            (r"(-?[0-9.]+)\s*/\s*100\b", 100.0),
            (r"(-?[0-9.]+)\s*/\s*10\b", 10.0),
            (r"(-?[0-9.]+)\s+out of\s+10", 10.0),
            (r"score[:\s]+(-?[0-9.]+)", None),
            (r"rating[:\s]+(-?[0-9.]+)", None),
        ]

        score: float | None = None
        for pattern, denom in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    raw_score = float(match.group(1))
                except ValueError:
                    continue
                if denom is not None:
                    value = raw_score / denom
                elif raw_score > 1:
                    value = raw_score / 10
                else:
                    value = raw_score
                # Anything outside [0, 1] would skew persisted quality means
                # and the 0.8/0.5 example-classification gates.
                score = max(0.0, min(1.0, value))
                break

        # Extract explanation (first sentence or full response)
        explanation = response.split("\n")[0] if "\n" in response else response
        if len(explanation) > 200:
            explanation = explanation[:200] + "..."

        return score, explanation
