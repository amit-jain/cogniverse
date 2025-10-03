"""
Learned Reranking using LiteLLM

Provides unified interface for learned reranking models:
- LiteLLM: Cohere, Together AI, Jina AI, Azure AI, AWS Bedrock, HuggingFace, etc.
- Ollama: Local reranking models via OpenAI-compatible API (bge-reranker-v2-m3, mxbai-rerank-large-v2, etc.)
- Model-agnostic: works with both cross-encoder and ColBERT style models
- Simple configuration through config.json

LiteLLM handles all models through unified API.
For Ollama models, use "openai/<model_name>" with api_base set to Ollama endpoint.
"""

import logging
from typing import Any, Dict, List, Optional

from litellm import arerank, rerank

from src.app.search.multi_modal_reranker import SearchResult
from src.common.config_utils import get_config_value

logger = logging.getLogger(__name__)


class LearnedReranker:
    """
    Unified learned reranker using LiteLLM

    Supports any model that LiteLLM supports:
    - Cross-encoder models (e.g., Cohere, Jina)
    - ColBERT-style models (e.g., Together AI's Llama-Rank)
    - Any other reranking models added to LiteLLM

    Configuration is loaded from config.json under "reranking" section.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize learned reranker

        Args:
            model: Model name (e.g., "cohere/rerank-english-v3.0", "openai/bge-reranker-v2-m3")
                   For Ollama models, use "openai/<model_name>" with api_base set
                   If None, loads from config.json
        """
        # Load config
        rerank_config = get_config_value("reranking", {})

        # Determine model to use
        if model:
            self.model = model
        else:
            # Load from config
            model_key = rerank_config.get("model", "heuristic")
            supported_models = rerank_config.get("supported_models", {})

            if model_key == "heuristic":
                raise ValueError(
                    "LearnedReranker requires a learned model. "
                    "Set reranking.model in config.json to a supported model "
                    "(cohere, together_ai, jina, ollama, etc.)"
                )

            self.model = supported_models.get(model_key)
            if not self.model:
                raise ValueError(
                    f"Model '{model_key}' not found in supported_models. "
                    f"Available: {list(supported_models.keys())}"
                )

        # Get top_n from config
        self.default_top_n = rerank_config.get("top_n")
        self.max_results_to_rerank = rerank_config.get("max_results_to_rerank", 100)

        # Get api_base for OpenAI-compatible endpoints (Ollama)
        self.api_base = rerank_config.get("api_base", None)

        logger.info(f"Initialized LearnedReranker with model: {self.model}")

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank results using learned model via LiteLLM

        Args:
            query: User query
            results: Search results to rerank
            top_n: Number of top results to return (None = use config default)

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []

        # Limit results if needed
        if len(results) > self.max_results_to_rerank:
            results = results[: self.max_results_to_rerank]

        # Prepare documents for LiteLLM
        # LiteLLM expects List[str] of document texts
        documents = [f"{r.title} {r.content}" for r in results]

        # Determine top_n
        effective_top_n = top_n or self.default_top_n

        try:
            # Call LiteLLM rerank
            # For Ollama models, LiteLLM uses OpenAI-compatible API with custom api_base
            kwargs = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": effective_top_n,
            }

            # Add api_base for OpenAI-compatible endpoints (Ollama)
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await arerank(**kwargs)

            # Map LiteLLM response back to SearchResult objects
            reranked = []
            for result_item in response.results:
                original_result = results[result_item.index]
                original_result.metadata["reranking_score"] = result_item.relevance_score
                original_result.metadata["reranker_model"] = self.model
                original_result.metadata["original_rank"] = result_item.index
                reranked.append(original_result)

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed with model {self.model}: {e}")
            # Return original results on failure
            return results

    def rerank_sync(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Synchronous version of rerank

        Args:
            query: User query
            results: Search results to rerank
            top_n: Number of top results to return

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []

        # Limit results if needed
        if len(results) > self.max_results_to_rerank:
            results = results[: self.max_results_to_rerank]

        # Prepare documents
        documents = [f"{r.title} {r.content}" for r in results]

        # Determine top_n
        effective_top_n = top_n or self.default_top_n

        try:
            # Call LiteLLM rerank (synchronous)
            kwargs = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": effective_top_n,
            }

            # Add api_base for OpenAI-compatible endpoints (Ollama)
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = rerank(**kwargs)

            # Map response back to SearchResult objects
            reranked = []
            for result_item in response.results:
                original_result = results[result_item.index]
                original_result.metadata["reranking_score"] = result_item.relevance_score
                original_result.metadata["reranker_model"] = self.model
                original_result.metadata["original_rank"] = result_item.index
                reranked.append(original_result)

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed with model {self.model}: {e}")
            return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker model

        Returns:
            Dictionary with model details
        """
        return {
            "model": self.model,
            "max_results_to_rerank": self.max_results_to_rerank,
            "default_top_n": self.default_top_n,
        }
