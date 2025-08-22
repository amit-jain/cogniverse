#!/usr/bin/env python3
"""
Custom Exception Classes for Generic Ingestion Pipeline.

Simplified exception hierarchy for any type of content ingestion.
"""

from pathlib import Path
from typing import Any


class PipelineException(Exception):
    """Base exception for all pipeline-related errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} (Context: {context_str})"
        return super().__str__()


class ContentProcessingError(PipelineException):
    """Raised when content processing fails."""

    def __init__(
        self,
        message: str,
        content_path: Path | None = None,
        stage: str | None = None,
        profile: str | None = None,
        **context,
    ):
        context.update(
            {
                "content_path": str(content_path) if content_path else None,
                "stage": stage,
                "profile": profile,
            }
        )
        super().__init__(message, context)


class EmbeddingGenerationError(PipelineException):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        segment_count: int | None = None,
        embedding_type: str | None = None,
        **context,
    ):
        context.update(
            {
                "model_name": model_name,
                "segment_count": segment_count,
                "embedding_type": embedding_type,
            }
        )
        super().__init__(message, context)


class BackendError(PipelineException):
    """Raised when backend operations fail."""

    def __init__(
        self,
        message: str,
        backend_type: str | None = None,
        operation: str | None = None,
        schema: str | None = None,
        **context,
    ):
        context.update(
            {"backend_type": backend_type, "operation": operation, "schema": schema}
        )
        super().__init__(message, context)


class ProcessorError(PipelineException):
    """Raised when a processor fails."""

    def __init__(self, message: str, processor_type: str | None = None, **context):
        context["processor_type"] = processor_type
        super().__init__(message, context)


# Helper functions for common error patterns
def wrap_processor_error(
    processor_type: str, operation: str, original_exception: Exception
) -> ProcessorError:
    """Wrap an exception as a ProcessorError with context."""
    return ProcessorError(
        f"Processor {processor_type} failed during {operation}: {original_exception}",
        processor_type=processor_type,
        operation=operation,
        original_error=str(original_exception),
        original_type=type(original_exception).__name__,
    )


def wrap_content_error(
    content_path: Path, stage: str, profile: str, original_exception: Exception
) -> ContentProcessingError:
    """Wrap an exception as a ContentProcessingError with context."""
    return ContentProcessingError(
        f"Failed to process {content_path.name} at stage {stage}: {original_exception}",
        content_path=content_path,
        stage=stage,
        profile=profile,
        original_error=str(original_exception),
        original_type=type(original_exception).__name__,
    )


def wrap_embedding_error(
    model_name: str, original_exception: Exception
) -> EmbeddingGenerationError:
    """Wrap an exception as an EmbeddingGenerationError with context."""
    return EmbeddingGenerationError(
        f"Embedding generation failed for model {model_name}: {original_exception}",
        model_name=model_name,
        original_error=str(original_exception),
        original_type=type(original_exception).__name__,
    )


def wrap_backend_error(
    backend_type: str, operation: str, schema: str, original_exception: Exception
) -> BackendError:
    """Wrap an exception as a BackendError with context."""
    return BackendError(
        f"Backend {backend_type} failed during {operation}: {original_exception}",
        backend_type=backend_type,
        operation=operation,
        schema=schema,
        original_error=str(original_exception),
        original_type=type(original_exception).__name__,
    )
