"""
Unit tests for fine-tuning orchestrator.

Tests validation functions and orchestration flows for SFT, DPO, and embedding datasets.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_finetuning.orchestrator import (
    FinetuningOrchestrator,
    OrchestrationConfig,
    validate_dpo_dataset,
    validate_embedding_dataset,
    validate_sft_dataset,
)


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
class TestOrchestrationFlows:
    """Test end-to-end orchestration flows"""

    @pytest.mark.asyncio
    async def test_sft_orchestration_flow(self):
        """Test SFT orchestration flow with mocked components"""
        # Create mock telemetry provider
        mock_provider = MagicMock()

        # Create orchestrator
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            synthetic_service=None,
            approval_orchestrator=None,
        )

        # Create config for SFT (LLM with routing agent)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            backend="local",
            generate_synthetic=False,
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

            mock_formatter.return_value = [
                {"text": "Instruction: Q1\n\nResponse: A1"},
                {"text": "Instruction: Q2\n\nResponse: A2"},
            ]

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

            # Verify converter was called
            mock_converter.convert.assert_called_once_with(
                "cogniverse-tenant1", "routing"
            )

            # Verify formatter was called
            mock_formatter.assert_called_once()

            # Verify backend train_sft was called
            mock_backend.train_sft.assert_called_once()

    @pytest.mark.asyncio
    async def test_dpo_orchestration_flow(self):
        """Test DPO orchestration flow with mocked components"""
        # Create mock telemetry provider
        mock_provider = MagicMock()

        # Create orchestrator
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            synthetic_service=None,
            approval_orchestrator=None,
        )

        # Create config for DPO (LLM with routing agent)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="llm",
            agent_type="routing",
            base_model="HuggingFaceTB/SmolLM-135M",
            backend="local",
            generate_synthetic=False,
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
            MagicMock(prompt="Q1", chosen="Good1", rejected="Bad1"),
            MagicMock(prompt="Q2", chosen="Good2", rejected="Bad2"),
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

            # Verify extractor was called
            mock_extractor.extract.assert_called_once_with(
                "cogniverse-tenant1", "routing"
            )

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
            synthetic_service=None,
            approval_orchestrator=None,
        )

        # Create config for embedding (video modality)
        config = OrchestrationConfig(
            tenant_id="tenant1",
            project="cogniverse-tenant1",
            model_type="embedding",
            modality="video",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            backend="local",
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
    async def test_llm_orchestration_requires_agent_type(self):
        """Test that LLM orchestration raises error without agent_type"""
        mock_provider = MagicMock()
        orchestrator = FinetuningOrchestrator(
            telemetry_provider=mock_provider,
            synthetic_service=None,
            approval_orchestrator=None,
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
            synthetic_service=None,
            approval_orchestrator=None,
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
            synthetic_service=None,
            approval_orchestrator=None,
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
            synthetic_service=None,
            approval_orchestrator=None,
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
