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

import pytest

from cogniverse_runtime.ingestion.strategies import (
    AudioTranscriptionStrategy,
    ImageSegmentationStrategy,
    MultiVectorEmbeddingStrategy,
    SingleVectorEmbeddingStrategy,
)
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
        """A strategy whose type is not keyed in ``inference_services``
        gets no injection — the ``embedding`` entry only reaches an
        ``embedding`` strategy, never the transcription one.
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

    def test_strategy_without_inference_service_param_is_not_injected(self):
        """A strategy whose ``__init__`` does not declare
        ``inference_service`` must not receive the injection.

        Pre-refactor the factory broadcast the kwarg to every strategy
        and used a whole-signature filter to silently drop unaccepted
        params. That hid typos in profile JSON. The opt-in design only
        injects when the constructor explicitly declares the parameter,
        and any other unknown kwarg propagates to a loud TypeError
        rather than being silently swallowed.
        """
        # ``FrameSegmentationStrategy.__init__`` does not declare
        # inference_service. The factory must not inject and must not
        # raise; constructing it without the kwarg is the desired path.
        profile_config = {
            "inference_services": {"segmentation": "vllm_colpali"},
            "strategies": {
                "segmentation": {
                    "class": "FrameSegmentationStrategy",
                    "params": {"fps": 1.0},
                }
            },
        }
        strategy_set = StrategyFactory.create_from_profile_config(profile_config)
        assert strategy_set.segmentation is not None
        assert not hasattr(strategy_set.segmentation, "inference_service"), (
            "FrameSegmentationStrategy gained an inference_service attr "
            "that its __init__ should never have set"
        )

    def test_embedding_service_round_trips_through_factory(self):
        """``inference_services.embedding`` reaches the embedding strategy's
        processor requirements so ProcessorManager can resolve the URL."""
        profile_config = {
            "inference_services": {"embedding": "svc_x"},
            "strategies": {
                "embedding": {
                    "class": "MultiVectorEmbeddingStrategy",
                    "params": {"model_name": "TomoroAI/tomoro-colqwen3-embed-4b"},
                }
            },
        }

        strategy_set = StrategyFactory.create_from_profile_config(profile_config)

        strategy = strategy_set.embedding
        assert isinstance(strategy, MultiVectorEmbeddingStrategy)
        assert strategy.inference_service == "svc_x"
        assert strategy.get_required_processors() == {
            "embedding": {
                "type": "multi_vector",
                "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
                "inference_service": "svc_x",
            }
        }

    def test_unknown_param_in_profile_surfaces_as_typeerror(self):
        """A typo in profile params raises TypeError at construction so the
        misconfiguration is loud, not a silently dropped strategy."""
        profile_config = {
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base", "tpyo_field": "value"},
                }
            },
        }
        with pytest.raises(TypeError):
            StrategyFactory.create_from_profile_config(profile_config)


class TestInferenceServiceInProcessorRequirements:
    """Strategies that accept ``inference_service`` must surface it in
    ``get_required_processors()`` — that config key is what ProcessorManager
    pops and resolves to a concrete endpoint URL. A stored-but-unemitted
    value would leave the pipeline on the local model path.
    """

    def test_multi_vector_embedding_carries_service(self):
        strategy = MultiVectorEmbeddingStrategy(inference_service="vllm_colpali")
        assert strategy.get_required_processors() == {
            "embedding": {
                "type": "multi_vector",
                "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
                "inference_service": "vllm_colpali",
            }
        }

    def test_multi_vector_embedding_local_by_default(self):
        strategy = MultiVectorEmbeddingStrategy()
        assert strategy.get_required_processors() == {
            "embedding": {
                "type": "multi_vector",
                "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
            }
        }

    def test_single_vector_embedding_carries_service(self):
        strategy = SingleVectorEmbeddingStrategy(inference_service="vllm_colpali")
        assert strategy.get_required_processors() == {
            "embedding": {
                "type": "single_vector",
                "model_name": "google/videoprism-base",
                "inference_service": "vllm_colpali",
            }
        }

    def test_single_vector_embedding_local_by_default(self):
        strategy = SingleVectorEmbeddingStrategy()
        assert strategy.get_required_processors() == {
            "embedding": {
                "type": "single_vector",
                "model_name": "google/videoprism-base",
            }
        }

    def test_image_segmentation_carries_service(self):
        strategy = ImageSegmentationStrategy(inference_service="vllm_colpali")
        assert strategy.get_required_processors() == {
            "image": {
                "max_images": 10000,
                "inference_service": "vllm_colpali",
            }
        }

    def test_image_segmentation_local_by_default(self):
        strategy = ImageSegmentationStrategy()
        assert strategy.get_required_processors() == {"image": {"max_images": 10000}}


class TestKwargsDoesNotCountAsOptIn:
    """A bare **kwargs constructor must NOT be treated as accepting the
    injection — that shape absorbed and silently discarded the kwarg while
    the factory believed it was delivered."""

    def test_var_keyword_only_strategy_is_not_injected(self, monkeypatch):
        from cogniverse_runtime.ingestion import strategies as strategies_mod
        from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory

        class KwargsOnlyStrategy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(
            strategies_mod, "KwargsOnlyStrategy", KwargsOnlyStrategy, raising=False
        )
        assert (
            StrategyFactory._strategy_accepts_inference_service("KwargsOnlyStrategy")
            is False
        )

    def test_explicit_parameter_still_opts_in(self):
        from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory

        assert (
            StrategyFactory._strategy_accepts_inference_service(
                "MultiVectorEmbeddingStrategy"
            )
            is True
        )
