"""
Unit tests for fine-tuning orchestrator.

Tests validation functions and orchestration flows for SFT, DPO, and embedding datasets.
"""

import asyncio
import hashlib
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewItem,
)
from cogniverse_finetuning.dataset.output_projection import training_example_identity
from cogniverse_finetuning.orchestrator import (
    FinetuningOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    finetune,
    get_experiment_details,
    list_experiments,
    validate_dpo_dataset,
    validate_embedding_dataset,
    validate_sft_dataset,
)
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError


class _NoTracerProvider:
    """Stand-in for the real TelemetryProvider, which exposes NO ``tracer``
    attribute. The orchestrator's Phoenix logging used ``self.provider.tracer``
    — with this provider the old code AttributeErrors; the fix emits spans via
    the per-tenant TelemetryManager instead, so it must not touch the provider.
    """


class _RecordingSpan:
    def set_status(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingTelemetryManager:
    def __init__(self):
        self.calls = []

    def span(
        self,
        name,
        *,
        tenant_id=None,
        project_name=None,
        attributes=None,
        require_export=False,
    ):
        self.calls.append(
            {
                "name": name,
                "tenant_id": tenant_id,
                "project_name": project_name,
                "attributes": attributes or {},
                "require_export": require_export,
            }
        )
        return _RecordingSpan()


class _FailingTelemetryManager:
    def span(self, *args, **kwargs):
        raise RuntimeError("Required telemetry export is unavailable")


class _EvaluationFailingTelemetryManager(_RecordingTelemetryManager):
    def span(self, name, **kwargs):
        if name.startswith("evaluation."):
            raise RuntimeError("Required evaluation export is unavailable")
        return super().span(name, **kwargs)


@pytest.mark.unit
class TestValidationFunctions:
    """Test dataset validation functions"""

    def test_validate_sft_dataset_valid(self):
        """Test SFT validation with valid dataset"""
        dataset = [
            {"text": "Example 1"},
            {"text": "Example 2"},
        ]
        # Should not raise
        validate_sft_dataset(dataset)

    def test_validate_sft_dataset_empty(self):
        """Test SFT validation fails with empty dataset"""
        dataset = []
        with pytest.raises(ValueError, match="Cannot train with empty dataset"):
            validate_sft_dataset(dataset)

    def test_validate_sft_dataset_missing_fields(self):
        """Test SFT validation fails with missing required fields"""
        dataset = [
            {"wrong_field": "Example 1"},
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_sft_dataset(dataset)

    def test_validate_sft_dataset_partial_missing_fields(self):
        """Test SFT validation fails when some items miss fields"""
        dataset = [
            {"text": "Example 1"},
            {"wrong_field": "Example 2"},  # Missing 'text'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_sft_dataset(dataset)

    def test_validate_dpo_dataset_valid(self):
        """Test DPO validation with valid dataset"""
        dataset = [
            {"prompt": "Q1", "chosen": "Good answer", "rejected": "Bad answer"},
            {"prompt": "Q2", "chosen": "Good answer", "rejected": "Bad answer"},
        ]
        # Should not raise
        validate_dpo_dataset(dataset)

    def test_validate_dpo_dataset_empty(self):
        """Test DPO validation fails with empty dataset"""
        dataset = []
        with pytest.raises(ValueError, match="Cannot train with empty dataset"):
            validate_dpo_dataset(dataset)

    def test_validate_dpo_dataset_missing_prompt(self):
        """Test DPO validation fails with missing prompt"""
        dataset = [
            {"chosen": "Good", "rejected": "Bad"},  # Missing 'prompt'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_dpo_dataset(dataset)

    def test_validate_dpo_dataset_missing_chosen(self):
        """Test DPO validation fails with missing chosen"""
        dataset = [
            {"prompt": "Q1", "rejected": "Bad"},  # Missing 'chosen'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_dpo_dataset(dataset)

    def test_validate_dpo_dataset_missing_rejected(self):
        """Test DPO validation fails with missing rejected"""
        dataset = [
            {"prompt": "Q1", "chosen": "Good"},  # Missing 'rejected'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_dpo_dataset(dataset)

    def test_validate_embedding_dataset_valid(self):
        """Test embedding validation with valid dataset"""
        dataset = [
            {"anchor": "A1", "positive": "P1", "negative": "N1"},
            {"anchor": "A2", "positive": "P2", "negative": "N2"},
        ]
        # Should not raise
        validate_embedding_dataset(dataset)

    def test_validate_embedding_dataset_empty(self):
        """Test embedding validation fails with empty dataset"""
        dataset = []
        with pytest.raises(ValueError, match="Cannot train with empty dataset"):
            validate_embedding_dataset(dataset)

    def test_validate_embedding_dataset_missing_anchor(self):
        """Test embedding validation fails with missing anchor"""
        dataset = [
            {"positive": "P1", "negative": "N1"},  # Missing 'anchor'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_embedding_dataset(dataset)

    def test_validate_embedding_dataset_missing_positive(self):
        """Test embedding validation fails with missing positive"""
        dataset = [
            {"anchor": "A1", "negative": "N1"},  # Missing 'positive'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_embedding_dataset(dataset)

    def test_validate_embedding_dataset_missing_negative(self):
        """Test embedding validation fails with missing negative"""
        dataset = [
            {"anchor": "A1", "positive": "P1"},  # Missing 'negative'
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_embedding_dataset(dataset)


@pytest.mark.unit
class TestApprovedSyntheticLoader:
    @staticmethod
    def _config(tenant_id="acme:production"):
        return OrchestrationConfig(
            tenant_id=tenant_id,
            project="cogniverse-acme:production-finetuning",
            model_type="llm",
            agent_type="entity_extraction",
        )

    @staticmethod
    def _orchestrator(outcome):
        class DatasetStore:
            def __init__(self):
                self.names = []

            async def get_dataset(self, name):
                self.names.append(name)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        store = DatasetStore()
        provider = MagicMock()
        provider.datasets = store
        return (
            FinetuningOrchestrator(
                telemetry_provider=provider,
                telemetry_manager=_RecordingTelemetryManager(),
            ),
            store,
        )

    @staticmethod
    def _approved_entity_record(**overrides):
        record = {
            "item_id": "syn_0",
            "confidence": 0.9,
            "status": "approved",
            "created_at": "2026-08-05T00:00:00+00:00",
            "reviewed_at": None,
            "metadata.agent_type": "entity_extraction",
            "query": "PyTorch was released by Meta AI",
            "entities": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORG"},
            ],
            "entity_types": "TECHNOLOGY,ORG",
            "relationships": [
                {
                    "source": "PyTorch",
                    "target": "Meta AI",
                    "type": "RELEASED_BY",
                }
            ],
        }
        record.update(overrides)
        return record

    @staticmethod
    def _signed_record(record):
        reviewed_at = "2026-08-05T00:00:00+00:00"
        signed = {
            **record,
            "reviewed_at": reviewed_at,
            "metadata.approval_decision_timestamp": reviewed_at,
        }
        identity_json = json.dumps(
            {
                "item_id": signed["item_id"],
                "status": signed["status"],
                "decision": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        signed["metadata.approval_decision_sha256"] = hashlib.sha256(
            identity_json.encode("utf-8")
        ).hexdigest()
        canonical_json = json.dumps(
            signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        signed["metadata.approval_record_json"] = canonical_json
        signed["metadata.approval_record_sha256"] = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()
        return signed

    @pytest.mark.asyncio
    async def test_missing_exact_tenant_dataset_returns_empty(self):
        from cogniverse_foundation.telemetry.providers.base import (
            DatasetNotFoundError,
        )

        orchestrator, store = self._orchestrator(
            DatasetNotFoundError("dataset does not exist")
        )

        result = await orchestrator._load_approved_synthetic(self._config("acme"))

        assert result == []
        assert store.names == ["approved_synthetic_data-acme:acme"]

    @pytest.mark.asyncio
    async def test_provider_none_is_not_treated_as_empty(self):
        orchestrator, store = self._orchestrator(None)

        with pytest.raises(RuntimeError) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert str(error.value) == (
            "Approved synthetic dataset provider returned no frame for "
            "tenant=acme:production agent_type=entity_extraction "
            "dataset=approved_synthetic_data-acme:production"
        )
        assert isinstance(error.value.__cause__, TypeError)
        assert str(error.value.__cause__) == (
            "approved synthetic dataset provider returned None; expected pandas DataFrame"
        )
        assert store.names == ["approved_synthetic_data-acme:production"]

    @pytest.mark.asyncio
    async def test_provider_outage_retains_tenant_agent_and_dataset_context(self):
        orchestrator, store = self._orchestrator(
            ConnectionError("Phoenix refused the request")
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load approved synthetic dataset for "
                "tenant=acme:production agent_type=entity_extraction "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert store.names == ["approved_synthetic_data-acme:production"]
        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Phoenix refused the request"

    @pytest.mark.asyncio
    async def test_malformed_nonempty_frame_raises_with_context(self):
        orchestrator, store = self._orchestrator(
            pd.DataFrame([{"output": {"query": "wrong column"}}])
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Malformed approved synthetic dataset for "
                "tenant=acme:production agent_type=entity_extraction "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert store.names == ["approved_synthetic_data-acme:production"]
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            "non-empty approved synthetic dataset must contain an input column"
        )

    @pytest.mark.asyncio
    async def test_matching_entity_record_without_query_raises_with_context(self):
        record = self._approved_entity_record()
        del record["query"]
        record = self._signed_record(record)
        orchestrator, store = self._orchestrator(pd.DataFrame([{"input": record}]))

        with pytest.raises(RuntimeError) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert str(error.value) == (
            "Malformed approved synthetic dataset for "
            "tenant=acme:production agent_type=entity_extraction "
            "dataset=approved_synthetic_data-acme:production"
        )
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            "approved entity_extraction record at position 0 requires a non-empty "
            "query string"
        )
        assert store.names == ["approved_synthetic_data-acme:production"]

    @pytest.mark.asyncio
    async def test_matching_entity_record_with_malformed_entities_raises(self):
        record = self._approved_entity_record(
            entities="[{'text': 'PyTorch', 'type': 'TECHNOLOGY'}]"
        )
        record = self._signed_record(record)
        orchestrator, store = self._orchestrator(pd.DataFrame([{"input": record}]))

        with pytest.raises(RuntimeError) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert str(error.value) == (
            "Malformed approved synthetic dataset for "
            "tenant=acme:production agent_type=entity_extraction "
            "dataset=approved_synthetic_data-acme:production"
        )
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            "approved entity_extraction record at position 0 entities must be a "
            "non-empty list"
        )
        assert store.names == ["approved_synthetic_data-acme:production"]

    @pytest.mark.asyncio
    async def test_approved_record_without_agent_tag_raises_with_context(self):
        record = self._approved_entity_record()
        del record["metadata.agent_type"]
        record = self._signed_record(record)
        orchestrator, store = self._orchestrator(pd.DataFrame([{"input": record}]))

        with pytest.raises(RuntimeError) as error:
            await orchestrator._load_approved_synthetic(self._config())

        assert str(error.value) == (
            "Malformed approved synthetic dataset for "
            "tenant=acme:production agent_type=entity_extraction "
            "dataset=approved_synthetic_data-acme:production"
        )
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            "approved record at position 0 requires a non-empty "
            "metadata.agent_type string"
        )
        assert store.names == ["approved_synthetic_data-acme:production"]


@pytest.mark.unit
class TestOrchestrationFlows:
    """Test end-to-end orchestration flows"""

    @pytest.mark.asyncio
    async def test_auto_approved_generated_examples_train_in_same_run(self):
        """A fully auto-approved generated batch immediately supplies SFT data."""
        mock_provider = MagicMock()
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=MagicMock(),
            approval_agent=MagicMock(),
        )
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            min_sft_examples=2,
            min_dpo_pairs=20,
            evaluate_after_training=True,
            enable_registry=False,
        )
        generated = [
            {"query": "Find the launch video", "chosen_agent": "video_search"},
            {"query": "Find the launch transcript", "chosen_agent": "text_search"},
        ]
        approved_batch = ApprovalBatch(
            batch_id="synthetic_routing_batch",
            items=[
                ReviewItem(
                    item_id=f"item-{position}",
                    data=example,
                    confidence=1.0,
                    status=ApprovalStatus.AUTO_APPROVED,
                )
                for position, example in enumerate(generated)
            ],
        )
        insufficient = MagicMock(
            recommended_method="insufficient",
            approved_count=0,
            preference_pairs=0,
            needs_synthetic=True,
        )
        ready = MagicMock(
            recommended_method="sft",
            approved_count=2,
            preference_pairs=0,
            needs_synthetic=False,
            total_spans=0,
        )
        mock_backend_result = MagicMock(
            adapter_path="/tmp/adapters/sft_routing_auto",
            metrics={"train_loss": 0.25},
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as mock_selector_cls,
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as mock_converter_cls,
            patch.object(
                orchestrator,
                "_load_approved_synthetic",
                new=AsyncMock(side_effect=[[], generated]),
            ) as mock_load_approved,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
            patch(
                "cogniverse_finetuning.orchestrator.AdapterEvaluator"
            ) as evaluator_class,
            patch.object(orchestrator, "_log_evaluation_to_phoenix"),
            patch.object(orchestrator, "_log_experiment_to_phoenix"),
        ):
            mock_selector = MagicMock()
            mock_selector.analyze_and_prepare = AsyncMock(
                side_effect=[(insufficient, approved_batch), (ready, None)]
            )
            mock_selector_cls.return_value = mock_selector
            mock_converter_cls.return_value.convert = AsyncMock(
                return_value=MagicMock(examples=[])
            )
            mock_backend = MagicMock()
            mock_backend.train_sft = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend
            evaluator_class.return_value.evaluate = AsyncMock(
                return_value=MagicMock(accuracy_improvement=0.1)
            )

            result = await orchestrator.run(config)

        assert result.training_method == "sft"
        assert result.used_synthetic is True
        assert result.synthetic_approval_count == 2
        assert mock_load_approved.await_count == 2
        assert mock_selector.analyze_and_prepare.await_count == 2
        assert mock_selector.analyze_and_prepare.await_args_list[1].kwargs == {
            "provider": mock_provider,
            "project": "cogniverse-tenant1",
            "agent_type": "routing",
            "tenant_id": "tenant1",
            "min_sft_examples": 2,
            "min_dpo_pairs": 20,
            "generate_synthetic": False,
            "approved_synthetic": generated,
        }
        mock_converter_cls.assert_not_called()
        trained_dataset = mock_backend.train_sft.await_args.kwargs["dataset"]
        assert trained_dataset == [
            {
                "text": (
                    "### Instruction:\nRoute the following query to the appropriate "
                    "modality agent.\n\n### Input:\nFind the launch video\n\n"
                    '### Response:\n{"recommended_agent":"video_search"}'
                ),
                "metadata": {"synthetic": True, "agent_type": "routing"},
            },
            {
                "text": (
                    "### Instruction:\nRoute the following query to the appropriate "
                    "modality agent.\n\n### Input:\nFind the launch transcript\n\n"
                    '### Response:\n{"recommended_agent":"text_search"}'
                ),
                "metadata": {"synthetic": True, "agent_type": "routing"},
            },
        ]
        assert evaluator_class.return_value.evaluate.await_args.kwargs[
            "exclude_identities"
        ] == {
            training_example_identity(
                "routing",
                example["query"],
                json.dumps(
                    {"recommended_agent": example["chosen_agent"]},
                    separators=(",", ":"),
                ),
            )
            for example in generated
        }

    @pytest.mark.asyncio
    async def test_auto_approved_persisted_reload_failure_prevents_training(self):
        """A failed canonical reload prevents readiness and training."""
        mock_provider = MagicMock()
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=[
                DatasetNotFoundError("dataset does not exist"),
                ConnectionError("Phoenix became unavailable"),
            ]
        )
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=MagicMock(),
            approval_agent=MagicMock(),
        )
        batch = ApprovalBatch(
            batch_id="persisted-read-failure",
            items=[
                ReviewItem(
                    item_id="valid-item",
                    data={
                        "query": "valid query",
                        "chosen_agent": "video_search",
                    },
                    confidence=1.0,
                    status=ApprovalStatus.AUTO_APPROVED,
                )
            ],
        )
        insufficient = MagicMock(
            recommended_method="insufficient",
            approved_count=0,
            preference_pairs=0,
            needs_synthetic=True,
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as mock_selector_cls,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
        ):
            mock_selector = MagicMock()
            mock_selector.analyze_and_prepare = AsyncMock(
                return_value=(insufficient, batch)
            )
            mock_selector_cls.return_value = mock_selector

            with pytest.raises(RuntimeError) as error:
                await orchestrator.run(
                    OrchestrationConfig(
                        tenant_id="tenant1",
                        project="cogniverse-tenant1",
                        model_type="llm",
                        agent_type="routing",
                    )
                )

        assert str(error.value) == (
            "Failed to load approved synthetic dataset for tenant=tenant1 "
            "agent_type=routing dataset=approved_synthetic_data-tenant1:tenant1"
        )
        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Phoenix became unavailable"
        assert mock_selector.analyze_and_prepare.await_count == 1
        assert mock_provider.datasets.get_dataset.await_count == 2
        mock_create_backend.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_auto_approved_batches_do_not_mix_training_data(self):
        """Concurrent runs keep their generated examples request-local."""
        mock_provider = MagicMock()
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=MagicMock(),
            approval_agent=MagicMock(),
        )
        both_entered = asyncio.Event()
        first_call_count = 0
        approved_by_tenant = {
            "alpha": [{"query": "Find alpha video", "chosen_agent": "video_search"}],
            "beta": [{"query": "Find beta text", "chosen_agent": "text_search"}],
        }
        load_counts = {"alpha": 0, "beta": 0}

        async def _load_persisted(config):
            tenant = config.tenant_id
            load_counts[tenant] += 1
            if load_counts[tenant] == 1:
                return []
            return approved_by_tenant[tenant]

        class _Selector:
            def __init__(self, tenant: str, query: str, chosen_agent: str):
                self.tenant = tenant
                self.calls = 0
                self.batch = ApprovalBatch(
                    batch_id=f"batch-{tenant}",
                    items=[
                        ReviewItem(
                            item_id=f"item-{tenant}",
                            data={"query": query, "chosen_agent": chosen_agent},
                            confidence=1.0,
                            status=ApprovalStatus.AUTO_APPROVED,
                        )
                    ],
                )

            async def analyze_and_prepare(self, **kwargs):
                nonlocal first_call_count
                self.calls += 1
                assert kwargs["tenant_id"] == self.tenant
                if self.calls == 1:
                    first_call_count += 1
                    if first_call_count == 2:
                        both_entered.set()
                    await both_entered.wait()
                    return (
                        MagicMock(
                            recommended_method="insufficient",
                            approved_count=0,
                            preference_pairs=0,
                            needs_synthetic=True,
                        ),
                        self.batch,
                    )
                assert kwargs["generate_synthetic"] is False
                return (
                    MagicMock(
                        recommended_method="sft",
                        approved_count=1,
                        preference_pairs=0,
                        needs_synthetic=False,
                    ),
                    None,
                )

        selectors = [
            _Selector("alpha", "Find alpha video", "video_search"),
            _Selector("beta", "Find beta text", "text_search"),
        ]
        trained_datasets = []

        async def _train_sft(**kwargs):
            trained_datasets.append(kwargs["dataset"])
            return MagicMock(adapter_path="/tmp/adapter", metrics={})

        backend = MagicMock()
        backend.train_sft = AsyncMock(side_effect=_train_sft)
        configs = [
            OrchestrationConfig(
                tenant_id=tenant,
                project=f"project-{tenant}",
                model_type="llm",
                agent_type="routing",
                min_sft_examples=1,
                min_dpo_pairs=20,
                evaluate_after_training=False,
                enable_registry=False,
            )
            for tenant in ("alpha", "beta")
        ]

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector",
                side_effect=selectors,
            ),
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as mock_converter_cls,
            patch.object(
                orchestrator,
                "_load_approved_synthetic",
                side_effect=_load_persisted,
            ),
            patch.object(orchestrator, "_create_backend", return_value=backend),
            patch.object(orchestrator, "_log_experiment_to_phoenix"),
        ):
            mock_converter_cls.return_value.convert = AsyncMock(
                return_value=MagicMock(examples=[])
            )
            results = await asyncio.gather(
                *(orchestrator.run(config) for config in configs)
            )

        assert [result.training_method for result in results] == ["sft", "sft"]
        assert load_counts == {"alpha": 2, "beta": 2}
        assert len(trained_datasets) == 2
        assert {
            tuple(row["text"] for row in dataset) for dataset in trained_datasets
        } == {
            (
                "### Instruction:\nRoute the following query to the appropriate "
                "modality agent.\n\n### Input:\nFind alpha video\n\n"
                '### Response:\n{"recommended_agent":"video_search"}',
            ),
            (
                "### Instruction:\nRoute the following query to the appropriate "
                "modality agent.\n\n### Input:\nFind beta text\n\n"
                '### Response:\n{"recommended_agent":"text_search"}',
            ),
        }

    @pytest.mark.asyncio
    async def test_sft_orchestration_flow(self):
        """Test SFT orchestration flow with mocked components"""
        # Create mock telemetry provider
        mock_provider = MagicMock()
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )

        # Create orchestrator
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        # Create config for SFT (LLM with routing agent)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            backend="local",
            min_sft_examples=5,
            generate_synthetic=False,
            evaluate_after_training=False,
            enable_registry=False,
        )

        # Mock the selector to return SFT recommendation
        mock_analysis = MagicMock()
        mock_analysis.recommended_method = "sft"
        mock_analysis.approved_count = 60
        mock_analysis.preference_pairs = 0
        mock_analysis.needs_synthetic = False

        # Mock the converter to return instruction examples
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.examples = [
            MagicMock(instruction="Q1", response="A1"),
            MagicMock(instruction="Q2", response="A2"),
        ]

        # Mock the backend result
        mock_backend_result = MagicMock()
        mock_backend_result.adapter_path = "/tmp/adapters/sft_routing_123"
        mock_backend_result.metrics = {"train_loss": 0.5}

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as mock_selector_cls,
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as mock_converter_cls,
            patch(
                "cogniverse_finetuning.orchestrator.InstructionFormatter.format_alpaca_text"
            ) as mock_formatter,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
        ):
            # Setup mocks
            mock_selector = MagicMock()
            mock_selector.analyze_and_prepare = AsyncMock(
                return_value=(mock_analysis, None)
            )
            mock_selector_cls.return_value = mock_selector

            mock_converter = MagicMock()
            mock_converter.convert = AsyncMock(return_value=mock_dataset_obj)
            mock_converter_cls.return_value = mock_converter

            mock_formatter.side_effect = lambda examples: (
                [
                    {"text": "Instruction: Q1\n\nResponse: A1"},
                    {"text": "Instruction: Q2\n\nResponse: A2"},
                ]
                if examples
                else []
            )

            mock_backend = MagicMock()
            mock_backend.train_sft = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend

            # Run orchestration
            result = await orchestrator.run(config)

            # Verify result
            assert result.model_type == "llm"
            assert result.training_method == "sft"
            assert result.adapter_path == "/tmp/adapters/sft_routing_123"
            assert result.metrics["train_loss"] == 0.5
            assert result.base_model == "HuggingFaceTB/SmolLM-135M"
            assert result.used_synthetic is False

            # Verify selector was called
            mock_selector.analyze_and_prepare.assert_called_once()
            mock_provider.datasets.get_dataset.assert_awaited_once_with(
                name="approved_synthetic_data-tenant1:tenant1"
            )

            # The extractor must enforce the same threshold the selector used.
            mock_converter.convert.assert_called_once_with(
                "cogniverse-tenant1", "routing", min_annotations=5
            )

            # Verify formatter was called
            assert mock_formatter.call_count == 2
            assert mock_formatter.call_args_list[0].args == (mock_dataset_obj.examples,)
            assert mock_formatter.call_args_list[1].args == ([],)

            # Verify backend train_sft was called
            mock_backend.train_sft.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_training_is_not_returned_when_experiment_export_fails(
        self,
    ):
        provider = MagicMock()
        provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=provider,
            telemetry_manager=_FailingTelemetryManager(),
        )
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            generate_synthetic=False,
            evaluate_after_training=False,
            enable_registry=False,
        )
        analysis = MagicMock(
            recommended_method="sft",
            approved_count=1,
            preference_pairs=0,
            needs_synthetic=False,
        )
        backend_result = MagicMock(
            adapter_path="/tmp/adapters/sft-routing",
            metrics={"train_loss": 0.25},
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as selector_class,
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as converter_class,
            patch(
                "cogniverse_finetuning.orchestrator.InstructionFormatter.format_alpaca_text",
                return_value=[{"text": "route launch video"}],
            ),
            patch.object(orchestrator, "_create_backend") as create_backend,
            patch.object(orchestrator, "_register_adapter") as register_adapter,
        ):
            selector_class.return_value.analyze_and_prepare = AsyncMock(
                return_value=(analysis, None)
            )
            converter_class.return_value.convert = AsyncMock(
                return_value=MagicMock(examples=[MagicMock()])
            )
            create_backend.return_value.train_sft = AsyncMock(
                return_value=backend_result
            )

            with pytest.raises(
                RuntimeError,
                match="Required experiment export for tenant=tenant1 run=.* failed",
            ):
                await orchestrator.run(config)

        register_adapter.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_result_is_not_returned_when_configured_upload_fails(self):
        provider = MagicMock()
        provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )
        registry = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=provider,
            telemetry_manager=_RecordingTelemetryManager(),
            registry=registry,
        )
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            generate_synthetic=False,
            evaluate_after_training=False,
            enable_registry=True,
            adapter_storage_uri="hf://myorg/adapters",
        )
        analysis = MagicMock(
            recommended_method="sft",
            approved_count=1,
            preference_pairs=0,
            needs_synthetic=False,
        )
        backend_result = MagicMock(
            adapter_path="/tmp/adapters/sft-routing",
            metrics={"train_loss": 0.25},
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as selector_class,
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as converter_class,
            patch(
                "cogniverse_finetuning.orchestrator.InstructionFormatter.format_alpaca_text",
                return_value=[{"text": "route launch video"}],
            ),
            patch.object(orchestrator, "_create_backend") as create_backend,
            patch(
                "cogniverse_finetuning.registry.upload_adapter",
                side_effect=ConnectionError("storage offline"),
            ),
        ):
            selector_class.return_value.analyze_and_prepare = AsyncMock(
                return_value=(analysis, None)
            )
            converter_class.return_value.convert = AsyncMock(
                return_value=MagicMock(examples=[MagicMock()])
            )
            create_backend.return_value.train_sft = AsyncMock(
                return_value=backend_result
            )

            with pytest.raises(
                RuntimeError,
                match="Adapter publication for tenant=tenant1 run=.* failed",
            ) as error:
                await orchestrator.run(config)

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == (
            "Failed to register adapter for tenant tenant1"
        )
        registry.register_adapter.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluation_export_failure_is_not_masked(self):
        provider = MagicMock()
        provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )
        manager = _EvaluationFailingTelemetryManager()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=provider,
            telemetry_manager=manager,
        )
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            generate_synthetic=False,
            evaluate_after_training=True,
        )
        analysis = MagicMock(
            recommended_method="sft",
            approved_count=1,
            preference_pairs=0,
            needs_synthetic=False,
        )
        backend_result = MagicMock(
            adapter_path="/tmp/adapters/sft-routing",
            metrics={"train_loss": 0.25},
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as selector_class,
            patch(
                "cogniverse_finetuning.orchestrator.TraceToInstructionConverter"
            ) as converter_class,
            patch(
                "cogniverse_finetuning.orchestrator.InstructionFormatter.format_alpaca_text",
                return_value=[{"text": "route launch video"}],
            ),
            patch.object(orchestrator, "_create_backend") as create_backend,
            patch(
                "cogniverse_finetuning.orchestrator.AdapterEvaluator"
            ) as evaluator_class,
            patch.object(orchestrator, "_register_adapter") as register_adapter,
        ):
            selector_class.return_value.analyze_and_prepare = AsyncMock(
                return_value=(analysis, None)
            )
            training_example = MagicMock(
                input="find the launch video",
                output='{"recommended_agent":"video_search"}',
            )
            converter_class.return_value.convert = AsyncMock(
                return_value=MagicMock(examples=[training_example])
            )
            create_backend.return_value.train_sft = AsyncMock(
                return_value=backend_result
            )
            evaluator_class.return_value.evaluate = AsyncMock(return_value=MagicMock())

            with pytest.raises(
                RuntimeError,
                match="Required evaluation export for tenant=tenant1 run=.* failed",
            ):
                await orchestrator.run(config)

        assert manager.calls == []
        assert evaluator_class.return_value.evaluate.await_args.kwargs[
            "exclude_identities"
        ] == {
            training_example_identity(
                "routing",
                "find the launch video",
                '{"recommended_agent":"video_search"}',
            )
        }
        register_adapter.assert_not_called()

    @pytest.mark.asyncio
    async def test_dpo_orchestration_flow(self):
        """Test DPO orchestration flow with mocked components"""
        # Create mock telemetry provider
        mock_provider = MagicMock()
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=DatasetNotFoundError("dataset does not exist")
        )

        # Create orchestrator
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        # Create config for DPO (LLM with routing agent)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            backend="local",
            min_dpo_pairs=1,
            generate_synthetic=False,
            evaluate_after_training=True,
            enable_registry=False,
        )

        # Mock the selector to return DPO recommendation
        mock_analysis = MagicMock()
        mock_analysis.recommended_method = "dpo"
        mock_analysis.approved_count = 30
        mock_analysis.preference_pairs = 25
        mock_analysis.needs_synthetic = False

        # Mock the extractor to return preference pairs
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.pairs = [
            MagicMock(
                prompt="Q1",
                chosen='{"recommended_agent":"video_search"}',
                rejected='{"recommended_agent":"text_search"}',
            ),
            MagicMock(
                prompt="Q2",
                chosen='{"recommended_agent":"text_search"}',
                rejected='{"recommended_agent":"video_search"}',
            ),
        ]

        # Mock the backend result
        mock_backend_result = MagicMock()
        mock_backend_result.adapter_path = "/tmp/adapters/dpo_routing_123"
        mock_backend_result.metrics = {"train_loss": 0.3}

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as mock_selector_cls,
            patch(
                "cogniverse_finetuning.orchestrator.PreferencePairExtractor"
            ) as mock_extractor_cls,
            patch(
                "cogniverse_finetuning.orchestrator.InstructionFormatter.format_dpo"
            ) as mock_formatter,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
            patch(
                "cogniverse_finetuning.orchestrator.AdapterEvaluator"
            ) as evaluator_class,
            patch.object(orchestrator, "_log_evaluation_to_phoenix"),
        ):
            # Setup mocks
            mock_selector = MagicMock()
            mock_selector.analyze_and_prepare = AsyncMock(
                return_value=(mock_analysis, None)
            )
            mock_selector_cls.return_value = mock_selector

            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=mock_dataset_obj)
            mock_extractor_cls.return_value = mock_extractor

            mock_formatter.return_value = [
                {"prompt": "Q1", "chosen": "Good1", "rejected": "Bad1"},
                {"prompt": "Q2", "chosen": "Good2", "rejected": "Bad2"},
            ]

            mock_backend = MagicMock()
            mock_backend.train_dpo = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend
            evaluator_class.return_value.evaluate = AsyncMock(
                return_value=MagicMock(accuracy_improvement=0.1)
            )

            # Run orchestration
            result = await orchestrator.run(config)

            # Verify result
            assert result.model_type == "llm"
            assert result.training_method == "dpo"
            assert result.adapter_path == "/tmp/adapters/dpo_routing_123"
            assert result.metrics["train_loss"] == 0.3
            assert result.base_model == "HuggingFaceTB/SmolLM-135M"
            assert result.used_synthetic is False

            # Verify selector was called
            mock_selector.analyze_and_prepare.assert_called_once()
            mock_provider.datasets.get_dataset.assert_awaited_once_with(
                name="approved_synthetic_data-tenant1:tenant1"
            )

            # Verify extractor was called
            mock_extractor.extract.assert_called_once_with(
                "cogniverse-tenant1", "routing", min_pairs=1
            )
            assert evaluator_class.return_value.evaluate.await_args.kwargs[
                "exclude_identities"
            ] == {
                training_example_identity("routing", "Q1", response)
                for response in (
                    '{"recommended_agent":"video_search"}',
                    '{"recommended_agent":"text_search"}',
                )
            } | {
                training_example_identity("routing", "Q2", response)
                for response in (
                    '{"recommended_agent":"text_search"}',
                    '{"recommended_agent":"video_search"}',
                )
            }

            # Verify formatter was called
            mock_formatter.assert_called_once()

            # Verify backend train_dpo was called
            mock_backend.train_dpo.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_orchestration_flow(self):
        """Test embedding orchestration flow with mocked components"""
        # Create mock telemetry provider
        mock_provider = MagicMock()

        # Create orchestrator
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        # Create config for embedding (video modality)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="embedding",
            modality="video",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            backend="local",
            min_triplets=2,
            enable_registry=False,
        )

        # Mock the extractor to return triplets
        mock_triplet1 = MagicMock()
        mock_triplet1.anchor = "Query1"
        mock_triplet1.positive = "Relevant1"
        mock_triplet1.negative = "Irrelevant1"

        mock_triplet2 = MagicMock()
        mock_triplet2.anchor = "Query2"
        mock_triplet2.positive = "Relevant2"
        mock_triplet2.negative = "Irrelevant2"

        mock_triplets = [mock_triplet1, mock_triplet2]

        # Mock the backend result
        mock_backend_result = MagicMock()
        mock_backend_result.adapter_path = "/tmp/adapters/embedding_video_123"
        mock_backend_result.metrics = {"train_loss": 0.2}

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TripletExtractor"
            ) as mock_extractor_cls,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
        ):
            # Setup mocks
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=mock_triplets)
            mock_extractor_cls.return_value = mock_extractor

            mock_backend = MagicMock()
            mock_backend.train_embedding = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend

            # Run orchestration
            result = await orchestrator.run(config)

            # Verify result
            assert result.model_type == "embedding"
            assert result.training_method == "embedding"
            assert result.adapter_path == "/tmp/adapters/embedding_video_123"
            assert result.metrics["train_loss"] == 0.2
            assert result.base_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert result.used_synthetic is False

            # Verify extractor was called with correct params
            mock_extractor.extract.assert_called_once()
            call_kwargs = mock_extractor.extract.call_args[1]
            assert call_kwargs["project"] == "cogniverse-tenant1"
            assert call_kwargs["modality"] == "video"

            # Verify backend train_embedding was called
            mock_backend.train_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_requires_minimum_triplet_count_before_backend_creation(
        self,
    ):
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=MagicMock(),
            telemetry_manager=_RecordingTelemetryManager(),
        )
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="embedding",
            modality="video",
            min_triplets=2,
        )
        only_triplet = MagicMock(
            anchor="launch query",
            positive="launch video",
            negative="cooking video",
        )

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TripletExtractor"
            ) as extractor_class,
            patch.object(orchestrator, "_create_backend") as create_backend,
        ):
            extractor_class.return_value.extract = AsyncMock(
                return_value=[only_triplet]
            )
            with pytest.raises(
                ValueError,
                match="Insufficient embedding triplets: required 2, received 1",
            ):
                await orchestrator.run(config)

        create_backend.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_orchestration_requires_agent_type(self):
        """Test that LLM orchestration raises error without agent_type"""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        # Config without agent_type
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type=None,  # Missing!
            base_model="HuggingFaceTB/SmolLM-135M",
        )

        with pytest.raises(ValueError, match="agent_type required"):
            await orchestrator.run(config)

    @pytest.mark.asyncio
    async def test_embedding_orchestration_requires_modality(self):
        """Test that embedding orchestration raises error without modality"""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        # Config without modality
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="embedding",
            modality=None,  # Missing!
            base_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        with pytest.raises(ValueError, match="modality required"):
            await orchestrator.run(config)

    @pytest.mark.asyncio
    async def test_backend_creation_local(self):
        """Test local backend creation"""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            backend="local",
        )

        with patch(
            "cogniverse_finetuning.orchestrator.LocalTrainingBackend"
        ) as mock_local_backend:
            orchestrator._create_backend(config)

            # Verify LocalTrainingBackend was instantiated
            mock_local_backend.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_creation_remote(self):
        """Test remote backend creation"""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            backend="remote",
            backend_provider="modal",
            gpu="A100-40GB",
        )

        with patch(
            "cogniverse_finetuning.orchestrator.RemoteTrainingBackend"
        ) as mock_remote_backend:
            orchestrator._create_backend(config)

            # Verify RemoteTrainingBackend was instantiated with correct provider
            mock_remote_backend.assert_called_once()
            call_args = mock_remote_backend.call_args
            assert call_args[1]["provider"] == "modal"


@pytest.mark.unit
class TestMultiTurnOrchestrationFlow:
    """Test multi-turn conversation fine-tuning orchestration."""

    @pytest.mark.asyncio
    async def test_multi_turn_sft_flow(self):
        """Test full multi-turn SFT flow uses TraceToTrajectoryConverter and train_sft."""
        from datetime import datetime

        from cogniverse_finetuning.dataset.trace_converter import (
            ConversationTrajectory,
            ConversationTurn,
            TrajectoryDataset,
        )

        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            backend="local",
            multi_turn=True,
            min_turns_per_session=2,
            system_prompt="You are a video search assistant.",
            evaluate_after_training=True,
            enable_registry=False,
        )

        # Build mock trajectory dataset
        turns = [
            ConversationTurn(
                turn_id=1,
                query="Find sports videos",
                response='{"recommended_agent":"video_search"}',
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
            ConversationTurn(
                turn_id=2,
                query="Show basketball dunks",
                response='{"recommended_agent":"video_search"}',
                timestamp=datetime(2025, 1, 1, 12, 1, 0),
                span_id="span2",
            ),
        ]
        mock_trajectory_dataset = TrajectoryDataset(
            trajectories=[ConversationTrajectory(session_id="session1", turns=turns)]
        )

        mock_backend_result = MagicMock()
        mock_backend_result.adapter_path = "/tmp/adapters/sft_multi_turn_routing_123"
        mock_backend_result.metrics = {"train_loss": 0.4}

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TraceToTrajectoryConverter"
            ) as mock_converter_cls,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
            patch(
                "cogniverse_finetuning.orchestrator.AdapterEvaluator"
            ) as evaluator_class,
            patch.object(orchestrator, "_log_evaluation_to_phoenix"),
        ):
            mock_converter = MagicMock()
            mock_converter.convert = AsyncMock(return_value=mock_trajectory_dataset)
            mock_converter_cls.return_value = mock_converter

            mock_backend = MagicMock()
            mock_backend.train_sft = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend
            evaluator_class.return_value.evaluate = AsyncMock(
                return_value=MagicMock(accuracy_improvement=0.1)
            )

            result = await orchestrator.run(config)

            # Verify result
            assert result.training_method == "sft_multi_turn"
            assert result.adapter_path == "/tmp/adapters/sft_multi_turn_routing_123"
            assert result.metrics["train_loss"] == 0.4
            assert result.used_synthetic is False

            # Verify TraceToTrajectoryConverter was used (not TraceToInstructionConverter)
            mock_converter_cls.assert_called_once_with(mock_provider)
            mock_converter.convert.assert_called_once()

            # Verify backend.train_sft was called (reuses existing SFT trainer)
            mock_backend.train_sft.assert_called_once()
            call_kwargs = mock_backend.train_sft.call_args[1]
            assert call_kwargs["config"]["dataset_text_field"] == "text"
            assert evaluator_class.return_value.evaluate.await_args.kwargs[
                "exclude_identities"
            ] == {
                training_example_identity(
                    "routing",
                    query,
                    '{"recommended_agent":"video_search"}',
                )
                for query in ("Find sports videos", "Show basketball dunks")
            }

    @pytest.mark.asyncio
    async def test_multi_turn_bypasses_method_selector(self):
        """Test that multi_turn=True completely bypasses TrainingMethodSelector."""
        from datetime import datetime

        from cogniverse_finetuning.dataset.trace_converter import (
            ConversationTrajectory,
            ConversationTurn,
            TrajectoryDataset,
        )

        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            multi_turn=True,
            evaluate_after_training=False,
            enable_registry=False,
        )

        turns = [
            ConversationTurn(
                turn_id=1,
                query="q1",
                response="r1",
                timestamp=datetime(2025, 1, 1),
                span_id="s1",
            ),
            ConversationTurn(
                turn_id=2,
                query="q2",
                response="r2",
                timestamp=datetime(2025, 1, 1),
                span_id="s2",
            ),
        ]
        mock_trajectory_dataset = TrajectoryDataset(
            trajectories=[ConversationTrajectory(session_id="session1", turns=turns)]
        )

        mock_backend_result = MagicMock()
        mock_backend_result.adapter_path = "/tmp/adapter"
        mock_backend_result.metrics = {"train_loss": 0.5}

        with (
            patch(
                "cogniverse_finetuning.orchestrator.TraceToTrajectoryConverter"
            ) as mock_converter_cls,
            patch(
                "cogniverse_finetuning.orchestrator.TrainingMethodSelector"
            ) as mock_selector_cls,
            patch.object(orchestrator, "_create_backend") as mock_create_backend,
        ):
            mock_converter = MagicMock()
            mock_converter.convert = AsyncMock(return_value=mock_trajectory_dataset)
            mock_converter_cls.return_value = mock_converter

            mock_backend = MagicMock()
            mock_backend.train_sft = AsyncMock(return_value=mock_backend_result)
            mock_create_backend.return_value = mock_backend

            await orchestrator.run(config)

            # TrainingMethodSelector should NEVER be instantiated
            mock_selector_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_turn_requires_agent_type(self):
        """Test that multi_turn still requires agent_type."""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            telemetry_manager=_RecordingTelemetryManager(),
            synthetic_service=None,
            approval_agent=None,
        )

        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type=None,  # Missing!
            multi_turn=True,
        )

        with pytest.raises(ValueError, match="agent_type required"):
            await orchestrator.run(config)


@pytest.mark.unit
class TestPhoenixLoggingUsesTelemetryManager:
    """Experiment records use the manager paired with the query provider."""

    def _config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            tenant_id="tenant1",
            project="proj",
            model_type="llm",
            agent_type="routing",
        )

    def test_log_experiment_uses_injected_manager_and_requires_export(self):
        manager = _RecordingTelemetryManager()
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(), telemetry_manager=manager
        )
        result = OrchestrationResult(
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/adapter",
            metrics={"train_loss": 0.1, "epoch": 3},
            base_model="m",
            lora_config={},
            used_synthetic=False,
            synthetic_approval_count=0,
        )
        orch._log_experiment_to_phoenix(
            config=self._config(),
            result=result,
            analysis=None,
            approved_batch=None,
            formatted_dataset=[],
            run_id="run_test",
        )
        assert manager.calls == [
            {
                "name": "experiment.routing.sft",
                "tenant_id": "tenant1",
                "project_name": "experiments",
                "attributes": manager.calls[0]["attributes"],
                "require_export": True,
            }
        ]
        assert manager.calls[0]["attributes"]["experiment.run_id"] == "run_test"
        assert manager.calls[0]["attributes"]["data.dataset_size"] == 0
        assert manager.calls[0]["attributes"]["params.method"] == "sft"

    def test_log_evaluation_uses_same_injected_manager(self):
        manager = _RecordingTelemetryManager()
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(), telemetry_manager=manager
        )
        orch._log_evaluation_to_phoenix(
            config=self._config(),
            adapter_path="/tmp/adapter",
            evaluation_result=MagicMock(),
            run_id="run-evaluation",
        )
        assert len(manager.calls) == 1
        assert manager.calls[0]["name"] == "evaluation.routing"
        assert manager.calls[0]["tenant_id"] == "tenant1"
        assert manager.calls[0]["project_name"] == "experiments"
        assert manager.calls[0]["require_export"] is True
        assert manager.calls[0]["attributes"]["experiment.run_id"] == ("run-evaluation")

    def test_concurrent_orchestrators_do_not_cross_route_experiment_spans(self):
        managers = [_RecordingTelemetryManager(), _RecordingTelemetryManager()]
        orchestrators = [
            FinetuningOrchestrator(
                telemetry_provider=_NoTracerProvider(), telemetry_manager=manager
            )
            for manager in managers
        ]
        result = OrchestrationResult(
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/adapter",
            metrics={"train_loss": 0.1, "epoch": 3},
            base_model="m",
            lora_config={},
            used_synthetic=False,
            synthetic_approval_count=0,
        )

        configs = [self._config(), self._config()]
        configs[0].tenant_id = "tenant-a"
        configs[1].tenant_id = "tenant-b"

        async def emit(index):
            await asyncio.sleep(0)
            orchestrators[index]._log_experiment_to_phoenix(
                config=configs[index],
                result=result,
                analysis=None,
                approved_batch=None,
                formatted_dataset=[],
                run_id=f"run-{index}",
            )

        async def emit_both():
            await asyncio.gather(emit(0), emit(1))

        asyncio.run(emit_both())
        assert [call["tenant_id"] for call in managers[0].calls] == ["tenant-a"]
        assert [call["tenant_id"] for call in managers[1].calls] == ["tenant-b"]
        assert managers[0].calls[0]["attributes"]["experiment.run_id"] == "run-0"
        assert managers[1].calls[0]["attributes"]["experiment.run_id"] == "run-1"


@pytest.mark.unit
class TestAdapterStorageUpload:
    """``config.hf_token`` must reach the storage backend on hf:// uploads."""

    def test_hf_token_forwarded_to_upload_adapter(self):
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(),
            telemetry_manager=_RecordingTelemetryManager(),
        )
        config = OrchestrationConfig(
            tenant_id="acme",
            project="proj",
            model_type="llm",
            agent_type="routing",
            adapter_version="2.1.0",
            adapter_storage_uri="hf://myorg/adapters",
            hf_token="hf_secret_abc",
        )
        result = MagicMock(adapter_path="/tmp/adapter", training_method="sft")

        captured = {}

        def _fake_upload(local_path, destination_uri, token=None):
            captured["local_path"] = local_path
            captured["destination_uri"] = destination_uri
            captured["token"] = token
            return destination_uri

        with patch("cogniverse_finetuning.registry.upload_adapter", _fake_upload):
            final_uri = orch._upload_adapter_to_storage(config, result)

        assert captured["token"] == "hf_secret_abc"
        assert captured["destination_uri"] == "hf://myorg/adapters/sft_routing_v2.1.0"
        assert final_uri == "hf://myorg/adapters/sft_routing_v2.1.0"

    def test_configured_upload_failure_raises_and_prevents_registration(self):
        registry = MagicMock()
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(),
            telemetry_manager=_RecordingTelemetryManager(),
            registry=registry,
        )
        config = OrchestrationConfig(
            tenant_id="acme",
            project="proj",
            model_type="llm",
            agent_type="routing",
            enable_registry=True,
            adapter_storage_uri="hf://myorg/adapters",
        )
        result = MagicMock(adapter_path="/tmp/adapter", training_method="sft")

        with patch(
            "cogniverse_finetuning.registry.upload_adapter",
            side_effect=ConnectionError("storage offline"),
        ):
            with pytest.raises(
                RuntimeError,
                match="Failed to register adapter for tenant acme",
            ) as error:
                orch._register_adapter(config, result, "run-1")

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == (
            "Failed to upload adapter for tenant acme to "
            "hf://myorg/adapters/sft_routing_v1.0.0"
        )
        registry.register_adapter.assert_not_called()

    def test_empty_uploaded_uri_is_rejected(self):
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(),
            telemetry_manager=_RecordingTelemetryManager(),
        )
        config = OrchestrationConfig(
            tenant_id="acme",
            project="proj",
            model_type="llm",
            agent_type="routing",
            adapter_storage_uri="hf://myorg/adapters",
        )
        result = MagicMock(adapter_path="/tmp/adapter", training_method="sft")

        with patch(
            "cogniverse_finetuning.registry.upload_adapter",
            return_value="",
        ):
            with pytest.raises(
                RuntimeError,
                match=(
                    "Adapter upload returned an empty URI for tenant acme at "
                    "hf://myorg/adapters/sft_routing_v1.0.0"
                ),
            ):
                orch._upload_adapter_to_storage(config, result)

    def test_empty_registry_adapter_id_is_rejected(self):
        registry = MagicMock()
        registry.register_adapter.return_value = ""
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(),
            telemetry_manager=_RecordingTelemetryManager(),
            registry=registry,
        )
        config = OrchestrationConfig(
            tenant_id="acme",
            project="proj",
            model_type="llm",
            agent_type="routing",
            enable_registry=True,
        )
        result = MagicMock(adapter_path="/tmp/adapter", training_method="sft")

        with pytest.raises(
            RuntimeError,
            match="Failed to register adapter for tenant acme",
        ) as error:
            orch._register_adapter(config, result, "run-1")

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == (
            "Adapter registry returned an empty adapter ID for tenant acme"
        )

    def test_enabled_registry_initialization_failure_raises(self):
        orch = FinetuningOrchestrator(
            telemetry_provider=_NoTracerProvider(),
            telemetry_manager=_RecordingTelemetryManager(),
        )
        config = OrchestrationConfig(
            tenant_id="acme",
            project="proj",
            model_type="llm",
            agent_type="routing",
            enable_registry=True,
        )
        result = MagicMock(adapter_path="/tmp/adapter", training_method="sft")

        with patch(
            "cogniverse_finetuning.registry.AdapterRegistry",
            side_effect=ConnectionError("Vespa registry offline"),
        ):
            with pytest.raises(
                RuntimeError,
                match="Failed to initialize adapter registry for tenant acme",
            ) as error:
                orch._register_adapter(config, result, "run-1")

        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Vespa registry offline"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finetune_helper_forwards_manager_and_evaluation_config():
    provider = _NoTracerProvider()
    manager = _RecordingTelemetryManager()
    expected = MagicMock(spec=OrchestrationResult)

    with patch(
        "cogniverse_finetuning.orchestrator.FinetuningOrchestrator"
    ) as orchestrator_class:
        orchestrator_class.return_value.run = AsyncMock(return_value=expected)
        result = await finetune(
            telemetry_provider=provider,
            telemetry_manager=manager,
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            evaluate_after_training=False,
            test_set_size=17,
        )

    assert result is expected
    orchestrator_class.assert_called_once_with(
        telemetry_provider=provider,
        telemetry_manager=manager,
        synthetic_service=None,
        approval_agent=None,
    )
    config = orchestrator_class.return_value.run.await_args.args[0]
    assert config.evaluate_after_training is False
    assert config.test_set_size == 17


class _SlowRecordingTelemetryManager(_RecordingTelemetryManager):
    def __init__(self, delay_seconds):
        super().__init__()
        self.delay_seconds = delay_seconds
        self.entered = threading.Event()

    def span(self, name, **kwargs):
        self.entered.set()
        time.sleep(self.delay_seconds)
        return super().span(name, **kwargs)


class _RecordingRegistry:
    def __init__(self, delay_seconds=0):
        self.delay_seconds = delay_seconds
        self.calls = []
        self._lock = threading.Lock()

    def register_adapter(self, **kwargs):
        time.sleep(self.delay_seconds)
        with self._lock:
            self.calls.append(kwargs)
        return f"adapter-{kwargs['tenant_id']}"


def _finished_result(training_method):
    return OrchestrationResult(
        model_type="embedding" if training_method == "embedding" else "llm",
        training_method=training_method,
        adapter_path=f"/tmp/{training_method}-adapter",
        metrics={"train_loss": 0.125, "train_samples": 3},
        base_model="exact-model",
        lora_config={"use_lora": True},
        used_synthetic=False,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("training_method", "agent_type", "modality", "expected_span"),
    [
        ("dpo", "routing", None, "experiment.routing.dpo"),
        ("sft", "profile_selection", None, "experiment.profile_selection.sft"),
        (
            "sft_multi_turn",
            "entity_extraction",
            None,
            "experiment.entity_extraction.sft_multi_turn",
        ),
        ("embedding", None, "video", "experiment.video.embedding"),
    ],
)
async def test_result_finalization_keeps_event_loop_responsive_for_every_training_mode(
    training_method,
    agent_type,
    modality,
    expected_span,
):
    manager = _SlowRecordingTelemetryManager(0.08)
    registry = _RecordingRegistry(0.08)
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=_NoTracerProvider(),
        telemetry_manager=manager,
        registry=registry,
    )
    config = OrchestrationConfig(
        tenant_id=f"tenant-{training_method}",
        project=f"project-{training_method}",
        model_type="embedding" if training_method == "embedding" else "llm",
        agent_type=agent_type,
        modality=modality,
        enable_registry=True,
    )
    result = _finished_result(training_method)
    ticks = 0
    finished = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not finished.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    try:
        finalized = await orchestrator._finalize_training_result(
            config=config,
            result=result,
            analysis=None,
            approved_batch=None,
            formatted_dataset=[{"text": training_method}],
        )
    finally:
        finished.set()
        await ticker_task

    assert finalized is result
    assert finalized.adapter_id == f"adapter-tenant-{training_method}"
    assert ticks >= 10
    assert manager.calls[0]["name"] == expected_span
    assert manager.calls[0]["tenant_id"] == f"tenant-{training_method}"
    assert manager.calls[0]["require_export"] is True
    assert manager.calls[0]["attributes"]["data.dataset_size"] == 1
    target = agent_type or modality
    assert registry.calls == [
        {
            "tenant_id": f"tenant-{training_method}",
            "name": f"{training_method}_{target}",
            "version": "1.0.0",
            "base_model": "HuggingFaceTB/SmolLM-135M",
            "model_type": "embedding" if training_method == "embedding" else "llm",
            "training_method": training_method,
            "adapter_path": f"/tmp/{training_method}-adapter",
            "adapter_uri": None,
            "agent_type": agent_type,
            "metrics": {"train_loss": 0.125, "train_samples": 3},
            "training_config": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "use_lora": True,
                "backend": "local",
            },
            "experiment_run_id": manager.calls[0]["attributes"]["experiment.run_id"],
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_finalization_initializes_registry_once_and_keeps_run_contexts_isolated():
    manager = _RecordingTelemetryManager()
    registry = _RecordingRegistry(0.03)
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=_NoTracerProvider(),
        telemetry_manager=manager,
    )
    configs = [
        OrchestrationConfig(
            tenant_id=f"tenant-{index}",
            project=f"project-{index}",
            model_type="llm",
            agent_type="routing",
            enable_registry=True,
        )
        for index in range(2)
    ]
    results = [_finished_result("sft") for _ in configs]

    with patch(
        "cogniverse_finetuning.registry.AdapterRegistry", return_value=registry
    ) as registry_class:
        finalized = await asyncio.gather(
            *(
                orchestrator._finalize_training_result(
                    config=config,
                    result=result,
                    analysis=None,
                    approved_batch=None,
                    formatted_dataset=[{"text": config.tenant_id}],
                )
                for config, result in zip(configs, results, strict=True)
            )
        )

    registry_class.assert_called_once_with()
    assert [result.adapter_id for result in finalized] == [
        "adapter-tenant-0",
        "adapter-tenant-1",
    ]
    registered = {
        call["tenant_id"]: call["experiment_run_id"] for call in registry.calls
    }
    exported = {
        call["tenant_id"]: call["attributes"]["experiment.run_id"]
        for call in manager.calls
    }
    assert registered == exported
    assert set(registered) == {"tenant-0", "tenant-1"}
    assert len(set(registered.values())) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hung_required_export_times_out_with_context_and_never_registers():
    release = threading.Event()

    class HungManager:
        def span(self, *args, **kwargs):
            release.wait(1)
            return _RecordingSpan()

    registry = _RecordingRegistry()
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=_NoTracerProvider(),
        telemetry_manager=HungManager(),
        registry=registry,
    )
    config = OrchestrationConfig(
        tenant_id="tenant-hung",
        project="project-hung",
        model_type="llm",
        agent_type="routing",
        enable_registry=True,
    )

    try:
        with patch("cogniverse_finetuning.orchestrator._FINALIZATION_TIMEOUT_S", 0.05):
            with pytest.raises(
                TimeoutError,
                match=(
                    "Required experiment export for tenant=tenant-hung run=.* "
                    "timed out after 0.05 seconds"
                ),
            ):
                await orchestrator._finalize_training_result(
                    config=config,
                    result=_finished_result("sft"),
                    analysis=None,
                    approved_batch=None,
                    formatted_dataset=[{"text": "route me"}],
                )
    finally:
        release.set()

    assert registry.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hung_registry_times_out_with_context_and_result_stays_unpublished():
    release = threading.Event()

    class HungRegistry:
        def register_adapter(self, **kwargs):
            release.wait(1)
            return f"adapter-{kwargs['tenant_id']}"

    result = _finished_result("sft")
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=_NoTracerProvider(),
        telemetry_manager=_RecordingTelemetryManager(),
        registry=HungRegistry(),
    )
    config = OrchestrationConfig(
        tenant_id="tenant-registry-hung",
        project="project-registry-hung",
        model_type="llm",
        agent_type="routing",
        enable_registry=True,
    )

    try:
        with patch("cogniverse_finetuning.orchestrator._FINALIZATION_TIMEOUT_S", 0.05):
            with pytest.raises(
                TimeoutError,
                match=(
                    "Adapter publication for tenant=tenant-registry-hung run=.* "
                    "timed out after 0.05 seconds"
                ),
            ):
                await orchestrator._finalize_training_result(
                    config=config,
                    result=result,
                    analysis=None,
                    approved_batch=None,
                    formatted_dataset=[{"text": "route me"}],
                )
    finally:
        release.set()

    assert result.adapter_id is None


class _LosslessExperimentTraces:
    def __init__(self):
        rows = [
            {
                "attributes.openinference.span.kind": "EXPERIMENT",
                "attributes.operation.name": "fine_tuning",
                "attributes.experiment.run_id": "old-run",
                "attributes.experiment.agent_type": "routing",
                "attributes.params.method": "sft",
                "attributes.params.base_model": "exact-model",
                "attributes.metrics.train_loss": 0.125,
                "attributes.output.adapter_path": "/tmp/old-adapter",
                "start_time": pd.Timestamp("2025-01-01T00:00:00Z"),
            }
        ]
        rows.extend(
            {
                "attributes.openinference.span.kind": "CHAIN",
                "attributes.operation.name": "search",
                "attributes.experiment.run_id": f"newer-{index}",
                "attributes.experiment.agent_type": "routing",
                "attributes.params.method": "sft",
                "start_time": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(microseconds=index),
            }
            for index in range(1001)
        )
        self.frame = pd.DataFrame(rows)
        self.projects = []

    async def get_all_spans(self, *, project):
        self.projects.append(project)
        return self.frame.copy()

    async def get_spans(self, **kwargs):
        raise AssertionError("paged span reads may hide older experiments")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_experiment_queries_find_an_old_record_beyond_the_default_page():
    traces = _LosslessExperimentTraces()
    provider = MagicMock()
    provider.traces = traces

    listed = await list_experiments(
        provider,
        "cogniverse-tenant-experiments",
        agent_type="routing",
        method="sft",
        limit=1,
    )
    details = await get_experiment_details(
        provider,
        "cogniverse-tenant-experiments",
        "old-run",
    )

    assert listed.to_dict("records") == [
        {
            "run_id": "old-run",
            "agent_type": "routing",
            "method": "sft",
            "base_model": "exact-model",
            "train_loss": 0.125,
            "adapter_path": "/tmp/old-adapter",
            "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
        }
    ]
    assert details == {
        "openinference.span.kind": "EXPERIMENT",
        "operation.name": "fine_tuning",
        "experiment.run_id": "old-run",
        "experiment.agent_type": "routing",
        "params.method": "sft",
        "params.base_model": "exact-model",
        "metrics.train_loss": 0.125,
        "output.adapter_path": "/tmp/old-adapter",
    }
    assert traces.projects == [
        "cogniverse-tenant-experiments",
        "cogniverse-tenant-experiments",
    ]
