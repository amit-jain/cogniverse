"""
End-to-end fine-tuning orchestration.

Combines all components into a high-level API:
- Auto-selection (SFT vs DPO vs embedding)
- Dataset extraction from telemetry
- Synthetic generation with mandatory approval
- Training with LoRA (local or remote backend)
- Adapter registration
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from opentelemetry.trace import Status, StatusCode

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor
from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector
from cogniverse_finetuning.dataset.preference_extractor import PreferencePairExtractor
from cogniverse_finetuning.dataset.trace_converter import TraceToInstructionConverter
from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator
from cogniverse_finetuning.training.backend import (
    LocalTrainingBackend,
    RemoteTrainingBackend,
    TrainingBackend,
    TrainingJobConfig,
)

logger = logging.getLogger(__name__)


def validate_sft_dataset(dataset: List[Dict]) -> None:
    """
    Validate SFT dataset has required fields.

    Args:
        dataset: List of formatted SFT examples

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No training examples available after formatting."
        )

    required_fields = ["text"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid SFT dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


def validate_dpo_dataset(dataset: List[Dict]) -> None:
    """
    Validate DPO dataset has required fields.

    Args:
        dataset: List of formatted DPO preference pairs

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No preference pairs available after formatting."
        )

    required_fields = ["prompt", "chosen", "rejected"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid DPO dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


def validate_embedding_dataset(dataset: List[Dict]) -> None:
    """
    Validate embedding dataset has required fields.

    Args:
        dataset: List of formatted triplets

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No triplets available after formatting."
        )

    required_fields = ["anchor", "positive", "negative"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid embedding dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


@dataclass
class OrchestrationConfig:
    """Configuration for fine-tuning orchestration."""

    # Tenant and project
    tenant_id: str
    project: str

    # Model type
    model_type: Literal["llm", "embedding"]

    # Agent type (for LLM)
    agent_type: Optional[Literal["routing", "profile_selection", "entity_extraction"]] = None

    # Modality (for embedding)
    modality: Optional[Literal["video", "image", "text"]] = None

    # Base model
    base_model: str = "HuggingFaceTB/SmolLM-135M"

    # Auto-selection thresholds
    min_sft_examples: int = 50
    min_dpo_pairs: int = 20
    min_triplets: int = 100  # For embeddings

    # Training config
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    use_lora: bool = True
    backend: Literal["local", "remote"] = "local"
    backend_provider: str = "modal"  # Used when backend="remote" (modal, sagemaker, azure_ml, etc.)

    # Remote backend config (GPU, timeouts, etc.)
    gpu: str = "A10G"
    gpu_count: int = 1
    cpu: int = 4
    memory: int = 16384
    timeout: int = 3600

    # Synthetic generation
    generate_synthetic: bool = True

    # Evaluation
    evaluate_after_training: bool = True  # Auto-evaluate adapter
    test_set_size: int = 50  # Number of test examples

    # Output
    output_dir: str = "outputs/adapters"


@dataclass
class OrchestrationResult:
    """Result from orchestration."""

    model_type: Literal["llm", "embedding"]
    training_method: Literal["sft", "dpo", "embedding"]
    adapter_path: str
    metrics: Dict
    base_model: str
    lora_config: Dict
    used_synthetic: bool
    synthetic_approval_count: Optional[int] = None
    evaluation_result: Optional[Any] = None  # ComparisonResult from evaluation


class FinetuningOrchestrator:
    """
    End-to-end fine-tuning orchestration.

    Workflow:
    1. Analyze available data from telemetry
    2. Auto-select training method (SFT/DPO/embedding)
    3. Generate synthetic if needed (with mandatory approval)
    4. Extract and format dataset
    5. Train with LoRA
    6. Return adapter and metrics
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        synthetic_service: Optional[any] = None,
        approval_orchestrator: Optional[any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            telemetry_provider: TelemetryProvider for querying spans/annotations
            synthetic_service: Optional SyntheticDataService for data generation
            approval_orchestrator: Optional ApprovalOrchestrator for human approval
        """
        self.provider = telemetry_provider
        self.synthetic_service = synthetic_service
        self.approval_orchestrator = approval_orchestrator

    def _log_experiment_to_phoenix(
        self,
        config: OrchestrationConfig,
        result: "OrchestrationResult",
        analysis: any,
        approved_batch: any,
        formatted_dataset: List[Dict],
    ) -> None:
        """
        Log experiment to Phoenix as EXPERIMENT span.

        Args:
            config: Orchestration configuration
            result: Training result with metrics and adapter path
            analysis: DataAnalysis from method selector
            approved_batch: Approval batch (if synthetic was used)
            formatted_dataset: Formatted training dataset
        """
        run_id = f"run_{datetime.utcnow().isoformat()}"

        experiment_span_attributes = {
            "openinference.span.kind": "EXPERIMENT",
            "operation.name": "fine_tuning",
            "experiment.run_id": run_id,
            "experiment.agent_type": config.agent_type or config.modality,
            # Hyperparameters
            "params.base_model": config.base_model,
            "params.method": result.training_method,
            "params.backend": config.backend,
            "params.backend_provider": config.backend_provider,
            "params.epochs": config.epochs,
            "params.batch_size": config.batch_size,
            "params.learning_rate": config.learning_rate,
            "params.use_lora": config.use_lora,
            "params.lora_r": 8,
            "params.lora_alpha": 16,
            # Dataset info
            "data.total_spans": analysis.total_spans if analysis else 0,
            "data.approved_count": analysis.approved_count if analysis else 0,
            "data.rejected_count": analysis.rejected_count if analysis else 0,
            "data.preference_pairs": analysis.preference_pairs if analysis else 0,
            "data.dataset_size": len(formatted_dataset),
            "data.used_synthetic": result.used_synthetic,
            "data.synthetic_approved_count": result.synthetic_approval_count or 0,
            # Results - Training Metrics
            "metrics.train_loss": result.metrics.get("train_loss"),
            "metrics.train_samples": result.metrics.get("train_samples"),
            "metrics.train_examples": result.metrics.get("train_examples"),
            "metrics.epochs_completed": result.metrics.get("epoch", config.epochs),
            # Validation Metrics (if validation split was used)
            "metrics.used_validation_split": result.metrics.get("used_validation_split", False),
            "metrics.val_examples": result.metrics.get("val_examples"),
            "metrics.eval_loss": result.metrics.get("eval_loss"),
            "metrics.eval_samples": result.metrics.get("eval_samples"),
            # DPO-specific validation metrics
            "metrics.eval_reward_accuracy": result.metrics.get("eval_reward_accuracy"),
            "metrics.eval_reward_margin": result.metrics.get("eval_reward_margin"),
            # Output
            "output.adapter_path": result.adapter_path,
        }

        # Create experiment span in Phoenix
        with self.provider.tracer.start_as_current_span(
            f"experiment.{config.agent_type or config.modality}.{result.training_method}",
            attributes=experiment_span_attributes,
        ) as span:
            span.set_status(Status(StatusCode.OK))

        logger.info(f"Experiment logged to Phoenix: {run_id}")

    def _log_evaluation_to_phoenix(
        self,
        config: OrchestrationConfig,
        adapter_path: str,
        evaluation_result: Any,  # ComparisonResult
    ) -> None:
        """
        Log adapter evaluation to Phoenix as EVALUATION span.

        Args:
            config: Orchestration configuration
            adapter_path: Path to evaluated adapter
            evaluation_result: ComparisonResult with evaluation metrics
        """
        eval_span_attributes = {
            "openinference.span.kind": "EVALUATION",
            "operation.name": "adapter_evaluation",
            "evaluation.adapter_path": adapter_path,
            "evaluation.agent_type": config.agent_type or config.modality,
            "evaluation.test_size": config.test_set_size,
            # Base metrics
            "metrics.base.accuracy": evaluation_result.base_metrics.accuracy,
            "metrics.base.confidence": evaluation_result.base_metrics.avg_confidence,
            "metrics.base.error_rate": evaluation_result.base_metrics.error_rate,
            "metrics.base.hallucination_rate": evaluation_result.base_metrics.hallucination_rate,
            "metrics.base.latency_ms": evaluation_result.base_metrics.avg_latency_ms,
            # Adapter metrics
            "metrics.adapter.accuracy": evaluation_result.adapter_metrics.accuracy,
            "metrics.adapter.confidence": evaluation_result.adapter_metrics.avg_confidence,
            "metrics.adapter.error_rate": evaluation_result.adapter_metrics.error_rate,
            "metrics.adapter.hallucination_rate": evaluation_result.adapter_metrics.hallucination_rate,
            "metrics.adapter.latency_ms": evaluation_result.adapter_metrics.avg_latency_ms,
            # Improvements
            "improvement.accuracy": evaluation_result.accuracy_improvement,
            "improvement.confidence": evaluation_result.confidence_improvement,
            "improvement.error_reduction": evaluation_result.error_reduction,
            "improvement.latency_overhead": evaluation_result.latency_overhead,
            "improvement.significant": evaluation_result.improvement_significant,
            "improvement.p_value": evaluation_result.p_value,
        }

        # Create evaluation span in Phoenix
        with self.provider.tracer.start_as_current_span(
            f"evaluation.{config.agent_type or config.modality}",
            attributes=eval_span_attributes,
        ) as span:
            span.set_status(Status(StatusCode.OK))

        logger.info("Evaluation logged to Phoenix")

    def _create_backend(self, config: OrchestrationConfig) -> TrainingBackend:
        """Create training backend based on config."""
        training_config = TrainingJobConfig(
            gpu=config.gpu,
            gpu_count=config.gpu_count,
            cpu=config.cpu,
            memory=config.memory,
            timeout=config.timeout,
        )

        if config.backend == "local":
            return LocalTrainingBackend(training_config)
        elif config.backend == "remote":
            return RemoteTrainingBackend(training_config, provider=config.backend_provider)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    async def run(self, config: OrchestrationConfig) -> OrchestrationResult:
        """
        Run end-to-end fine-tuning pipeline.

        Args:
            config: OrchestrationConfig

        Returns:
            OrchestrationResult with adapter path and metrics

        Example:
            >>> config = OrchestrationConfig(
            ...     tenant_id="tenant1",
            ...     project="cogniverse-tenant1",
            ...     model_type="llm",
            ...     agent_type="routing",
            ...     base_model="HuggingFaceTB/SmolLM-135M",
            ...     generate_synthetic=True
            ... )
            >>> result = await orchestrator.run(config)
            >>> print(f"Adapter saved to: {result.adapter_path}")
        """
        logger.info(
            f"Starting fine-tuning orchestration: type={config.model_type}, "
            f"agent={config.agent_type}, modality={config.modality}"
        )

        if config.model_type == "llm":
            return await self._run_llm_finetuning(config)
        elif config.model_type == "embedding":
            return await self._run_embedding_finetuning(config)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

    async def _run_llm_finetuning(
        self, config: OrchestrationConfig
    ) -> OrchestrationResult:
        """Run LLM fine-tuning (SFT or DPO)."""
        if not config.agent_type:
            raise ValueError("agent_type required for LLM fine-tuning")

        # 1. Analyze data and select method
        logger.info("Step 1: Analyzing available data...")
        selector = TrainingMethodSelector(
            synthetic_service=self.synthetic_service,
            approval_orchestrator=self.approval_orchestrator,
        )

        analysis, approved_batch = await selector.analyze_and_prepare(
            provider=self.provider,
            project=config.project,
            agent_type=config.agent_type,
            min_sft_examples=config.min_sft_examples,
            min_dpo_pairs=config.min_dpo_pairs,
            generate_synthetic=config.generate_synthetic,
        )

        logger.info(
            f"Data analysis: method={analysis.recommended_method}, "
            f"approved={analysis.approved_count}, "
            f"pairs={analysis.preference_pairs}, "
            f"synthetic={analysis.needs_synthetic}"
        )

        # 2. Extract and format dataset based on method
        logger.info("Step 2: Extracting and formatting dataset...")
        backend = self._create_backend(config)

        if analysis.recommended_method == "dpo":
            # DPO: Extract preference pairs
            extractor = PreferencePairExtractor(self.provider)
            dataset_obj = await extractor.extract(config.project, config.agent_type)
            pairs = dataset_obj.pairs

            logger.info(f"Extracted {len(pairs)} preference pairs for DPO")

            # Format for training
            formatted_dataset = InstructionFormatter.format_dpo(pairs)

            # Validate dataset
            validate_dpo_dataset(formatted_dataset)
            logger.info(f"Dataset validation passed: {len(formatted_dataset)} pairs")

            # Train with backend (add timestamp to avoid conflicts)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{config.output_dir}/dpo_{config.agent_type}_{timestamp}"
            training_config = {
                "use_lora": config.use_lora,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }

            result = await backend.train_dpo(
                dataset=formatted_dataset,
                base_model=config.base_model,
                output_dir=output_dir,
                config=training_config,
            )

            orchestration_result = OrchestrationResult(
                model_type="llm",
                training_method="dpo",
                adapter_path=result.adapter_path,
                metrics=result.metrics,
                base_model=config.base_model,
                lora_config={"use_lora": config.use_lora},
                used_synthetic=analysis.needs_synthetic,
                synthetic_approval_count=approved_batch.approved_count
                if approved_batch
                else None,
            )

            # Evaluate adapter automatically
            if config.evaluate_after_training:
                logger.info("Running post-training evaluation...")
                evaluator = AdapterEvaluator(
                    telemetry_provider=self.provider,
                    agent_type=config.agent_type,
                )

                try:
                    evaluation_result = await evaluator.evaluate(
                        base_model=config.base_model,
                        adapter_path=result.adapter_path,
                        project=config.project,
                        test_size=config.test_set_size,
                    )

                    # Log to Phoenix
                    self._log_evaluation_to_phoenix(
                        config=config,
                        adapter_path=result.adapter_path,
                        evaluation_result=evaluation_result,
                    )

                    orchestration_result.evaluation_result = evaluation_result

                    logger.info(
                        f"Evaluation complete: accuracy improvement={evaluation_result.accuracy_improvement:.2%}"
                    )

                except Exception as e:
                    logger.error(f"Evaluation failed: {e}", exc_info=True)
                    # Continue without evaluation

            # Log experiment to Phoenix
            self._log_experiment_to_phoenix(
                config=config,
                result=orchestration_result,
                analysis=analysis,
                approved_batch=approved_batch,
                formatted_dataset=formatted_dataset,
            )

            return orchestration_result

        elif analysis.recommended_method == "sft":
            # SFT: Extract instruction examples
            converter = TraceToInstructionConverter(self.provider)
            dataset_obj = await converter.convert(config.project, config.agent_type)
            examples = dataset_obj.examples

            logger.info(f"Extracted {len(examples)} examples for SFT")

            # Format for training (Alpaca format with "text" field for TRL)
            formatted_dataset = InstructionFormatter.format_alpaca_text(examples)

            # Validate dataset
            validate_sft_dataset(formatted_dataset)
            logger.info(f"Dataset validation passed: {len(formatted_dataset)} examples")

            # Train with backend (add timestamp to avoid conflicts)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{config.output_dir}/sft_{config.agent_type}_{timestamp}"
            training_config = {
                "use_lora": config.use_lora,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "dataset_text_field": "text",  # Required for TRL SFTTrainer
            }

            result = await backend.train_sft(
                dataset=formatted_dataset,
                base_model=config.base_model,
                output_dir=output_dir,
                config=training_config,
            )

            orchestration_result = OrchestrationResult(
                model_type="llm",
                training_method="sft",
                adapter_path=result.adapter_path,
                metrics=result.metrics,
                base_model=config.base_model,
                lora_config={"use_lora": config.use_lora},
                used_synthetic=analysis.needs_synthetic,
                synthetic_approval_count=approved_batch.approved_count
                if approved_batch
                else None,
            )

            # Evaluate adapter automatically
            if config.evaluate_after_training:
                logger.info("Running post-training evaluation...")
                evaluator = AdapterEvaluator(
                    telemetry_provider=self.provider,
                    agent_type=config.agent_type,
                )

                try:
                    evaluation_result = await evaluator.evaluate(
                        base_model=config.base_model,
                        adapter_path=result.adapter_path,
                        project=config.project,
                        test_size=config.test_set_size,
                    )

                    # Log to Phoenix
                    self._log_evaluation_to_phoenix(
                        config=config,
                        adapter_path=result.adapter_path,
                        evaluation_result=evaluation_result,
                    )

                    orchestration_result.evaluation_result = evaluation_result

                    logger.info(
                        f"Evaluation complete: accuracy improvement={evaluation_result.accuracy_improvement:.2%}"
                    )

                except Exception as e:
                    logger.error(f"Evaluation failed: {e}", exc_info=True)
                    # Continue without evaluation

            # Log experiment to Phoenix
            self._log_experiment_to_phoenix(
                config=config,
                result=orchestration_result,
                analysis=analysis,
                approved_batch=approved_batch,
                formatted_dataset=formatted_dataset,
            )

            return orchestration_result

        else:
            raise ValueError(
                f"Insufficient data and synthetic generation failed. "
                f"Analysis: {analysis}"
            )

    async def _run_embedding_finetuning(
        self, config: OrchestrationConfig
    ) -> OrchestrationResult:
        """Run embedding fine-tuning (contrastive learning)."""
        if not config.modality:
            raise ValueError("modality required for embedding fine-tuning")

        # 1. Extract triplets from search logs
        logger.info("Step 1: Extracting triplets from search logs...")
        extractor = TripletExtractor(self.provider)
        triplets = await extractor.extract(
            project=config.project,
            modality=config.modality,
            strategy="top_k",
            min_triplets=config.min_triplets,
        )

        logger.info(f"Extracted {len(triplets)} triplets")

        if len(triplets) < config.min_triplets:
            logger.warning(
                f"Insufficient triplets: {len(triplets)} < {config.min_triplets}. "
                "Proceeding with available data, but results may be suboptimal."
            )

        # 2. Format triplets for training (convert to dicts with anchor/positive/negative)
        formatted_dataset = [
            {
                "anchor": t.anchor,
                "positive": t.positive,
                "negative": t.negative,
            }
            for t in triplets
        ]

        # Validate dataset
        validate_embedding_dataset(formatted_dataset)
        logger.info(f"Dataset validation passed: {len(formatted_dataset)} triplets")

        # 3. Train with backend
        logger.info("Step 3: Training embedding model...")
        backend = self._create_backend(config)

        # Add timestamp to avoid conflicts
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.output_dir}/embedding_{config.modality}_{timestamp}"
        training_config = {
            "use_lora": config.use_lora,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }

        result = await backend.train_embedding(
            dataset=formatted_dataset,
            base_model=config.base_model,
            output_dir=output_dir,
            config=training_config,
        )

        orchestration_result = OrchestrationResult(
            model_type="embedding",
            training_method="embedding",
            adapter_path=result.adapter_path,
            metrics=result.metrics,
            base_model=config.base_model,
            lora_config={"use_lora": config.use_lora},
            used_synthetic=False,  # Embedding doesn't use synthetic yet
        )

        # Log experiment to Phoenix
        self._log_experiment_to_phoenix(
            config=config,
            result=orchestration_result,
            analysis=None,  # No analysis for embedding yet
            approved_batch=None,
            formatted_dataset=formatted_dataset,
        )

        return orchestration_result


# High-level convenience function
async def finetune(
    telemetry_provider: TelemetryProvider,
    tenant_id: str,
    project: str,
    model_type: Literal["llm", "embedding"],
    agent_type: Optional[str] = None,
    modality: Optional[str] = None,
    base_model: str = "HuggingFaceTB/SmolLM-135M",
    backend: Literal["local", "remote"] = "local",
    backend_provider: str = "modal",  # Used when backend="remote"
    # Training hyperparameters
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    # Remote backend configuration (GPU, timeouts, etc.)
    gpu: str = "A10G",  # T4, A10G, A100-40GB, A100-80GB, H100
    gpu_count: int = 1,
    cpu: int = 4,
    memory: int = 16384,
    timeout: int = 3600,  # seconds
    # Synthetic and approval
    synthetic_service: Optional[any] = None,
    approval_orchestrator: Optional[any] = None,
    # Output
    output_dir: str = "outputs/adapters",
) -> OrchestrationResult:
    """
    High-level fine-tuning function.

    Extracts dataset from telemetry and trains using specified backend (local or remote).

    Flow:
    1. Extract dataset locally from TelemetryProvider
    2. Auto-select training method (SFT vs DPO vs embedding)
    3. Generate synthetic data if needed (with approval)
    4. Train using backend (local or remote GPU)
    5. Return adapter and metrics

    Args:
        telemetry_provider: TelemetryProvider instance
        tenant_id: Tenant ID
        project: Project name
        model_type: "llm" or "embedding"
        agent_type: Agent type for LLM (routing, profile_selection, entity_extraction)
        modality: Modality for embedding (video, image, text)
        base_model: Base model name
        backend: "local" (local GPU/CPU) or "remote" (cloud GPU)
        backend_provider: Remote provider when backend="remote" (modal, sagemaker, azure_ml, etc.)
        gpu: GPU type for remote backend (T4, A10G, A100-40GB, etc.)
        synthetic_service: Optional SyntheticDataService
        approval_orchestrator: Optional ApprovalOrchestrator

    Returns:
        OrchestrationResult

    Example (Local Training):
        >>> provider = manager.get_provider("tenant1", "cogniverse-tenant1")
        >>> result = await finetune(
        ...     telemetry_provider=provider,
        ...     tenant_id="tenant1",
        ...     project="cogniverse-tenant1",
        ...     model_type="llm",
        ...     agent_type="routing",
        ...     backend="local"  # Train locally
        ... )

    Example (Remote GPU Training):
        >>> result = await finetune(
        ...     telemetry_provider=provider,
        ...     tenant_id="tenant1",
        ...     project="cogniverse-tenant1",
        ...     model_type="llm",
        ...     agent_type="routing",
        ...     backend="remote",  # Train on remote GPU
        ...     backend_provider="modal",  # Use Modal
        ...     gpu="A100-40GB",  # Choose GPU type
        ...     epochs=5,
        ...     batch_size=8,
        ...     learning_rate=1e-4
        ... )
    """
    # Create orchestration config with all parameters
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=project,
        model_type=model_type,
        agent_type=agent_type,
        modality=modality,
        base_model=base_model,
        backend=backend,
        backend_provider=backend_provider,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gpu=gpu,
        gpu_count=gpu_count,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
        output_dir=output_dir,
    )

    # Create orchestrator
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=telemetry_provider,
        synthetic_service=synthetic_service,
        approval_orchestrator=approval_orchestrator,
    )

    return await orchestrator.run(config)


# Dataset Analysis Helpers


async def analyze_dataset_status(
    telemetry_provider: TelemetryProvider,
    project: str,
    agent_type: Optional[str] = None,
    modality: Optional[str] = None,
    min_sft_examples: int = 50,
    min_dpo_pairs: int = 20,
) -> Dict[str, Any]:
    """
    Analyze dataset status for fine-tuning readiness.

    Returns dataset metrics, recommended method, and confidence level.

    Args:
        telemetry_provider: Phoenix provider
        project: Project name
        agent_type: Agent type (for LLM fine-tuning)
        modality: Modality (for embedding fine-tuning)
        min_sft_examples: Minimum examples for SFT
        min_dpo_pairs: Minimum pairs for DPO

    Returns:
        Dict with dataset status metrics

    Example:
        >>> status = await analyze_dataset_status(
        ...     provider, "cogniverse-tenant1", agent_type="routing"
        ... )
        >>> print(f"Approved: {status['approved_count']}/{status['sft_target']}")
        >>> print(f"Method: {status['recommended_method']}")
    """
    from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector

    # Create selector
    selector = TrainingMethodSelector(
        synthetic_service=None,  # No synthetic for status check
        approval_orchestrator=None,
    )

    # Analyze data
    analysis, _ = await selector.analyze_and_prepare(
        provider=telemetry_provider,
        project=project,
        agent_type=agent_type,
        min_sft_examples=min_sft_examples,
        min_dpo_pairs=min_dpo_pairs,
        generate_synthetic=False,  # Just analyze, don't generate
    )

    # Calculate progress percentages
    sft_progress = (analysis.approved_count / min_sft_examples) * 100 if min_sft_examples > 0 else 0
    dpo_progress = (analysis.preference_pairs / min_dpo_pairs) * 100 if min_dpo_pairs > 0 else 0

    # Determine status
    sft_ready = analysis.approved_count >= min_sft_examples
    dpo_ready = analysis.preference_pairs >= min_dpo_pairs

    # Calculate confidence in recommendation
    if analysis.recommended_method == "dpo":
        confidence = min(dpo_progress / 100, 1.0)
    elif analysis.recommended_method == "sft":
        confidence = min(sft_progress / 100, 1.0)
    else:
        confidence = 0.0

    return {
        # Raw counts
        "total_spans": analysis.total_spans,
        "approved_count": analysis.approved_count,
        "rejected_count": analysis.rejected_count,
        "preference_pairs": analysis.preference_pairs,

        # Targets
        "sft_target": min_sft_examples,
        "dpo_target": min_dpo_pairs,

        # Progress
        "sft_progress": sft_progress,
        "dpo_progress": dpo_progress,

        # Readiness
        "sft_ready": sft_ready,
        "dpo_ready": dpo_ready,

        # Recommendation
        "recommended_method": analysis.recommended_method,
        "confidence": confidence,
        "needs_synthetic": analysis.needs_synthetic,

        # Analysis object (for advanced use)
        "analysis": analysis,
    }


# Experiment Query Helpers


async def list_experiments(
    telemetry_provider: TelemetryProvider,
    project: str,
    agent_type: Optional[str] = None,
    method: Optional[Literal["sft", "dpo", "embedding"]] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """
    List fine-tuning experiments from Phoenix.

    Args:
        telemetry_provider: Phoenix provider
        project: Project name
        agent_type: Filter by agent type (routing, etc.) or modality (video, image, text)
        method: Filter by method (sft, dpo, embedding)
        limit: Max number of experiments to return

    Returns:
        DataFrame with experiment details

    Example:
        >>> experiments = await list_experiments(
        ...     provider, "cogniverse-tenant1", agent_type="routing"
        ... )
        >>> print(experiments[["run_id", "method", "train_loss"]])
    """
    # Query spans with EXPERIMENT kind
    spans_df = await telemetry_provider.traces.get_spans(project=project)

    if spans_df.empty:
        return pd.DataFrame()

    # Filter for EXPERIMENT spans
    mask = spans_df["attributes.openinference.span.kind"] == "EXPERIMENT"
    mask &= spans_df["attributes.operation.name"] == "fine_tuning"

    if agent_type:
        mask &= spans_df["attributes.experiment.agent_type"] == agent_type

    if method:
        mask &= spans_df["attributes.params.method"] == method

    experiments_df = spans_df[mask].copy()

    if experiments_df.empty:
        return experiments_df

    # Extract relevant columns
    result_columns = [
        "attributes.experiment.run_id",
        "attributes.experiment.agent_type",
        "attributes.params.method",
        "attributes.params.base_model",
        "attributes.params.backend",
        "attributes.params.batch_size",
        "attributes.params.learning_rate",
        "attributes.data.dataset_size",
        "attributes.data.used_synthetic",
        "attributes.metrics.train_loss",
        "attributes.output.adapter_path",
        "start_time",
    ]

    # Only select columns that exist
    available_columns = [col for col in result_columns if col in experiments_df.columns]
    result_df = experiments_df[available_columns].copy()

    # Rename columns for clarity
    column_mapping = {
        "attributes.experiment.run_id": "run_id",
        "attributes.experiment.agent_type": "agent_type",
        "attributes.params.method": "method",
        "attributes.params.base_model": "base_model",
        "attributes.params.backend": "backend",
        "attributes.params.batch_size": "batch_size",
        "attributes.params.learning_rate": "learning_rate",
        "attributes.data.dataset_size": "dataset_size",
        "attributes.data.used_synthetic": "used_synthetic",
        "attributes.metrics.train_loss": "train_loss",
        "attributes.output.adapter_path": "adapter_path",
        "start_time": "timestamp",
    }

    result_df.rename(columns={k: v for k, v in column_mapping.items() if k in result_df.columns}, inplace=True)

    # Sort by timestamp (newest first)
    if "timestamp" in result_df.columns:
        result_df = result_df.sort_values("timestamp", ascending=False)

    # Limit results
    if len(result_df) > limit:
        result_df = result_df.head(limit)

    return result_df


async def get_experiment_details(
    telemetry_provider: TelemetryProvider,
    project: str,
    run_id: str,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific experiment.

    Args:
        telemetry_provider: Phoenix provider
        project: Project name
        run_id: Experiment run ID

    Returns:
        Dict with all experiment metadata

    Example:
        >>> details = await get_experiment_details(
        ...     provider, "cogniverse-tenant1", "run_2025-11-24T10:00:00"
        ... )
        >>> print(f"Train loss: {details['metrics.train_loss']}")
    """
    # Query all spans
    spans_df = await telemetry_provider.traces.get_spans(project=project)

    if spans_df.empty:
        raise ValueError(f"Experiment not found: {run_id}")

    # Filter for this specific experiment
    mask = spans_df["attributes.openinference.span.kind"] == "EXPERIMENT"
    mask &= spans_df["attributes.experiment.run_id"] == run_id

    experiments_df = spans_df[mask]

    if experiments_df.empty:
        raise ValueError(f"Experiment not found: {run_id}")

    experiment = experiments_df.iloc[0]

    # Extract all attributes
    details = {}
    for col in experiment.index:
        if col.startswith("attributes."):
            key = col.replace("attributes.", "")
            details[key] = experiment[col]

    return details


async def compare_experiments(
    telemetry_provider: TelemetryProvider,
    project: str,
    run_ids: List[str],
) -> pd.DataFrame:
    """
    Compare multiple experiments side-by-side.

    Args:
        telemetry_provider: Phoenix provider
        project: Project name
        run_ids: List of experiment run IDs to compare

    Returns:
        DataFrame with side-by-side comparison

    Example:
        >>> comparison = await compare_experiments(
        ...     provider, "cogniverse-tenant1", ["run_001", "run_002"]
        ... )
        >>> print(comparison.T)  # Transpose for side-by-side view
    """
    experiments = []

    for run_id in run_ids:
        try:
            details = await get_experiment_details(telemetry_provider, project, run_id)
            experiments.append(details)
        except ValueError:
            # Skip experiments that don't exist
            continue

    if not experiments:
        return pd.DataFrame()

    comparison_df = pd.DataFrame(experiments)

    # Transpose for side-by-side view
    comparison_df = comparison_df.T
    comparison_df.columns = [f"Run {i+1}" for i in range(len(comparison_df.columns))]

    return comparison_df
