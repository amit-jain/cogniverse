"""ClaimExtractor must bind the per-tenant LM, not the ambient one."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy

from cogniverse_agents.graph.claim_extractor import ClaimExtractor


class _CapturingModule:
    """Records the active ``dspy.settings.lm`` at invocation time."""

    def __init__(self) -> None:
        self.captured_lm: object = None
        self.call_count: int = 0

    def __call__(self, **_):
        self.captured_lm = dspy.settings.lm
        self.call_count += 1
        return MagicMock(claims=[])


def test_per_tenant_lm_wraps_module_call() -> None:
    sentinel = MagicMock(name="per_tenant_lm")
    ambient = MagicMock(name="ambient_global_lm")
    extractor = ClaimExtractor(llm_config=MagicMock())
    extractor._cot_module = _CapturingModule()

    with dspy.context(lm=ambient):
        with patch(
            "cogniverse_foundation.config.llm_factory.create_dspy_lm",
            return_value=sentinel,
        ):
            extractor._invoke(
                text="some short text",
                entity_hints=["Alice"],
                modality_hint="text",
                tenant_id="acme",
            )

    assert extractor._cot_module.call_count == 1
    assert extractor._cot_module.captured_lm is sentinel, (
        f"Expected per-tenant LM; got {extractor._cot_module.captured_lm!r}"
    )


def test_no_llm_config_falls_through_to_ambient() -> None:
    ambient = MagicMock(name="ambient_global_lm")
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _CapturingModule()

    with dspy.context(lm=ambient):
        extractor._invoke(
            text="hi",
            entity_hints=[],
            modality_hint="text",
            tenant_id="acme",
        )

    assert extractor._cot_module.captured_lm is ambient
