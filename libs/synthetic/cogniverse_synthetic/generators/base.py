"""
Base Generator Interface

Abstract base class for all synthetic data generators.
Defines common interface and shared functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for optimizer-specific synthetic data generators

    All generators must implement the generate() method which produces
    synthetic training examples from sampled backend content.
    """

    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
    ):
        """
        Initialize base generator

        Args:
            pattern_extractor: Utility for extracting patterns from content
            agent_inferrer: Utility for inferring correct agents (optional)
        """
        self.pattern_extractor = pattern_extractor
        self.agent_inferrer = agent_inferrer
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
    ) -> List[BaseModel]:
        """
        Generate synthetic data from sampled content

        Args:
            sampled_content: Content sampled from backend (Vespa)
            target_count: Number of examples to generate
            **kwargs: Generator-specific parameters

        Returns:
            List of generated examples conforming to optimizer schema

        Raises:
            ValueError: If target_count is invalid or sampled_content is empty
        """
        pass

    def validate_inputs(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int
    ) -> None:
        """
        Validate generator inputs

        Args:
            sampled_content: Content sampled from backend
            target_count: Number of examples to generate

        Raises:
            ValueError: If inputs are invalid
        """
        if target_count <= 0:
            raise ValueError(f"target_count must be positive, got {target_count}")

        if not sampled_content:
            logger.warning(
                "No sampled content provided - will use fallback data if available"
            )

    def get_generator_info(self) -> Dict[str, Any]:
        """
        Get information about this generator

        Returns:
            Dictionary with generator metadata
        """
        return {
            "name": self.__class__.__name__,
            "has_pattern_extractor": self.pattern_extractor is not None,
            "has_agent_inferrer": self.agent_inferrer is not None,
        }
