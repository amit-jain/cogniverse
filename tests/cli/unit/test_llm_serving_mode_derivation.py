"""The serving mode of an already-deployed release is derived, never assumed."""

from __future__ import annotations

import yaml
from cogniverse_cli.config import (
    LLM_SERVING_LOCAL,
    LLM_SERVING_MODAL,
    get_llm_serving_values_file,
    llm_serving_mode_from_values,
)


def _modal_overlay() -> dict:
    return yaml.safe_load(get_llm_serving_values_file(LLM_SERVING_MODAL).read_text())


class TestModeIsDerivedFromTheDeployedValues:
    def test_values_carrying_the_modal_overlay_derive_modal(self) -> None:
        assert llm_serving_mode_from_values(_modal_overlay()) == LLM_SERVING_MODAL

    def test_values_without_it_derive_local(self) -> None:
        assert llm_serving_mode_from_values({"runtime": {}}) == LLM_SERVING_LOCAL

    def test_empty_values_derive_local(self) -> None:
        assert llm_serving_mode_from_values({}) == LLM_SERVING_LOCAL

    def test_no_release_at_all_derives_nothing(self) -> None:
        assert llm_serving_mode_from_values(None) is None

    def test_the_marker_is_the_overlays_own_api_base_not_a_literal(self) -> None:
        overlay = _modal_overlay()
        expected = overlay["runtime"]["primaryLLM"]["apiBase"]
        assert (
            llm_serving_mode_from_values(
                {"runtime": {"primaryLLM": {"apiBase": expected}}}
            )
            == LLM_SERVING_MODAL
        )

    def test_a_different_api_base_is_not_mistaken_for_modal(self) -> None:
        assert (
            llm_serving_mode_from_values(
                {
                    "runtime": {
                        "primaryLLM": {"apiBase": "http://cogniverse-vllm:8000/v1"}
                    }
                }
            )
            == LLM_SERVING_LOCAL
        )
