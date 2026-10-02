"""The document tokenizer the e2e cluster serves, loaded from the pinned cache."""

from __future__ import annotations

from pathlib import Path

from cogniverse_foundation.inference_specs import get_inference_service_spec

E2E_HF_HUB_CACHE = Path.home() / ".cache/cogniverse-tests/huggingface/hub"


def _served_document_tokens(
    text: str, *, cache_dir: Path = E2E_HF_HUB_CACHE
) -> list[int]:
    """``text`` tokenized by the model the document profile is served from.

    The tokenizer loads from the local cache only: a revision missing from
    ``cache_dir`` raises at once instead of waiting on the Hub.
    """
    from transformers import AutoTokenizer

    spec = get_inference_service_spec("colbert_pylate")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]
