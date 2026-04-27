"""Wiring tests for ``StrategyFactory`` profile-level inference_services map.

The factory reads each profile's ``inference_services[<strategy_type>]``
and injects it as the ``inference_service`` param on the matching
strategy class — so JSON configs no longer need to repeat the service
name inside each strategy's params block.

These tests pin the contract: a ``transcription`` entry in the map flows
into ``AudioTranscriptionStrategy.inference_service``; a value already
present in strategy params wins (backwards compatible); strategies whose
type isn't keyed in the map come through with no injection.
"""

from __future__ import annotations

from cogniverse_runtime.ingestion.strategies import AudioTranscriptionStrategy
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory


class TestInferenceServicesInjection:
    def test_transcription_service_lifted_from_profile_map(self):
        """``inference_services.transcription`` lands on the strategy."""
        profile_config = {
            "inference_services": {"transcription": "vllm_asr"},
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base"},
                }
            },
        }

        strategy_set = StrategyFactory.create_from_profile_config(profile_config)

        strategy = strategy_set.transcription
        assert isinstance(strategy, AudioTranscriptionStrategy)
        assert strategy.inference_service == "vllm_asr", (
            "factory must inject inference_service from profile-level map "
            "into the strategy"
        )

    def test_strategy_params_win_over_profile_map(self):
        """Backwards compat: an explicit param in ``params`` is not overwritten.

        Tests and tooling that build strategies directly with the param
        keep working unchanged after the refactor.
        """
        profile_config = {
            "inference_services": {"transcription": "ignored_by_explicit_param"},
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base", "inference_service": "explicit"},
                }
            },
        }

        strategy_set = StrategyFactory.create_from_profile_config(profile_config)

        assert strategy_set.transcription.inference_service == "explicit"

    def test_unmatched_strategy_type_gets_no_injection(self):
        """``inference_services.embedding`` is consumed by EmbeddingGeneratorFactory,
        not by StrategyFactory — embedding strategies must not gain a
        spurious ``inference_service`` param at construction time.
        """
        profile_config = {
            "inference_services": {"embedding": "vllm_colpali"},
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base"},
                }
            },
        }

        strategy_set = StrategyFactory.create_from_profile_config(profile_config)

        # No transcription key in the map → no injection on the
        # transcription strategy.
        assert strategy_set.transcription.inference_service is None

    def test_no_inference_services_block_keeps_strategies_local(self):
        """Profiles that don't set ``inference_services`` at all run
        strategies in local mode."""
        profile_config = {
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base"},
                }
            },
        }

        strategy_set = StrategyFactory.create_from_profile_config(profile_config)
        assert strategy_set.transcription.inference_service is None
