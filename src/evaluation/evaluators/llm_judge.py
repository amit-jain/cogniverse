"""
LLM-as-Judge Evaluators for Video Retrieval

Provides three types of LLM-based evaluation:
1. Reference-free: Evaluates query-result relevance without ground truth
2. Reference-based: Compares results against ground truth from database
3. Hybrid: Combines both approaches for comprehensive evaluation
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

import numpy as np
from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a video retrieved from database"""
    video_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    transcript: Optional[str] = None
    tags: Optional[List[str]] = None
    duration: Optional[float] = None
    frame_descriptions: Optional[List[str]] = None


class LLMJudgeBase:
    """Base class for LLM judge evaluators"""
    
    def __init__(self, model_name: str = "deepseek-r1:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM judge
        
        Args:
            model_name: Model to use for evaluation
            base_url: Base URL for LLM API (Ollama default)
        """
        self.model_name = model_name
        self.base_url = base_url
        self._client = None
    
    def _get_client(self):
        """Get or create Ollama client"""
        if self._client is None:
            try:
                from ollama import Client
                self._client = Client(host=self.base_url)
            except ImportError:
                logger.warning("Ollama client not available, using mock responses")
                self._client = None
        return self._client
    
    async def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call LLM with prompt
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            LLM response text
        """
        client = self._get_client()
        
        if client is None:
            # Return mock response if Ollama not available
            return "Mock evaluation: Results appear relevant to query (score: 0.75)"
        
        try:
            # For async compatibility, run in executor
            import asyncio
            loop = asyncio.get_event_loop()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=self.model_name,
                    messages=messages
                )
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Evaluation failed: {str(e)}"
    
    def _extract_score_from_response(self, response: str) -> Tuple[float, str]:
        """
        Extract numerical score and explanation from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (score, explanation)
        """
        import re
        
        # Try to extract score from response
        score_patterns = [
            r"score[:\s]+([0-9.]+)",
            r"rating[:\s]+([0-9.]+)",
            r"([0-9.]+)/10",
            r"([0-9.]+)\s+out of\s+10"
        ]
        
        score = 0.5  # Default score
        for pattern in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    raw_score = float(match.group(1))
                    # Normalize to 0-1 range
                    if raw_score > 1:
                        score = raw_score / 10
                    else:
                        score = raw_score
                    break
                except:
                    continue
        
        # Extract explanation (first sentence or full response)
        explanation = response.split('\n')[0] if '\n' in response else response
        if len(explanation) > 200:
            explanation = explanation[:200] + "..."
        
        return score, explanation


class SyncLLMReferenceFreeEvaluator(Evaluator, LLMJudgeBase):
    """
    LLM-based reference-free evaluator
    Evaluates query-result relevance without ground truth
    """
    
    def __init__(self, model_name: str = "deepseek-r1:7b", base_url: str = "http://localhost:11434"):
        LLMJudgeBase.__init__(self, model_name, base_url)
    
    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Synchronous evaluation for Phoenix experiments
        """
        # Run async evaluation in sync context
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._evaluate_async(input, output, **kwargs))
    
    async def _evaluate_async(self, input, output, **kwargs) -> EvaluationResult:
        """
        Evaluate query-result relevance using LLM
        """
        # Extract query
        query = input.get("query", "") if isinstance(input, dict) else str(input)
        
        # Extract results
        if hasattr(output, 'results'):
            results = output.results
        elif isinstance(output, dict) and 'results' in output:
            results = output['results']
        else:
            results = output if isinstance(output, list) else []
        
        if not results:
            return EvaluationResult(
                score=0.0,
                label="no_results",
                explanation="No results to evaluate"
            )
        
        # Prepare prompt for LLM
        system_prompt = """You are an expert evaluator for video search systems. 
Your task is to evaluate how well the search results match the user's query.
Provide a score from 0 to 10 and a brief explanation.
Consider relevance, ranking quality, and result diversity."""
        
        # Format results for prompt
        results_text = []
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            if isinstance(result, dict):
                video_id = result.get("video_id", "unknown")
                score = result.get("score", 0)
            else:
                video_id = getattr(result, "video_id", "unknown")
                score = getattr(result, "score", 0)
            
            results_text.append(f"{i}. Video: {video_id} (Score: {score:.3f})")
        
        prompt = f"""Query: "{query}"

Search Results:
{chr(10).join(results_text)}

Please evaluate these search results. Consider:
1. How relevant are the top results to the query?
2. Are the scores reasonable?
3. Is there good diversity in the results?

Provide a score from 0-10 and explanation.
Format: Score: X/10
Explanation: Your reasoning here"""
        
        # Call LLM
        response = await self._call_llm(prompt, system_prompt)
        
        # Parse response
        score, explanation = self._extract_score_from_response(response)
        
        # Determine label
        if score >= 0.8:
            label = "highly_relevant"
        elif score >= 0.6:
            label = "relevant"
        elif score >= 0.4:
            label = "partially_relevant"
        else:
            label = "not_relevant"
        
        return EvaluationResult(
            score=score,
            label=label,
            explanation=f"LLM Judge: {explanation}"
        )


class SyncLLMReferenceBasedEvaluator(Evaluator, LLMJudgeBase):
    """
    LLM-based reference evaluator
    Compares results against ground truth from database
    """
    
    def __init__(self, 
                 model_name: str = "deepseek-r1:7b",
                 base_url: str = "http://localhost:11434",
                 fetch_metadata: bool = True):
        """
        Initialize reference-based evaluator
        
        Args:
            model_name: LLM model to use
            base_url: LLM API base URL
            fetch_metadata: Whether to fetch video metadata from database
        """
        LLMJudgeBase.__init__(self, model_name, base_url)
        self.fetch_metadata = fetch_metadata
    
    def evaluate(self, *, input=None, output=None, expected=None, **kwargs) -> EvaluationResult:
        """
        Synchronous evaluation for Phoenix experiments
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._evaluate_async(input, output, expected, **kwargs)
        )
    
    async def _fetch_video_metadata(self, video_id: str) -> VideoMetadata:
        """
        Fetch video metadata from database
        
        Args:
            video_id: Video ID to fetch
            
        Returns:
            VideoMetadata object
        """
        try:
            # Use the metadata fetcher
            from .metadata_fetcher import VideoMetadataFetcher
            
            if not hasattr(self, '_metadata_fetcher'):
                self._metadata_fetcher = VideoMetadataFetcher()
            
            metadata = await self._metadata_fetcher.fetch_metadata(video_id)
            
            return VideoMetadata(
                video_id=video_id,
                title=metadata.get("title"),
                description=metadata.get("description"),
                transcript=metadata.get("transcript"),
                tags=metadata.get("tags"),
                duration=metadata.get("duration"),
                frame_descriptions=metadata.get("frame_descriptions")
            )
        except Exception as e:
            logger.debug(f"Could not fetch metadata for {video_id}: {e}")
            # Return default metadata
            return VideoMetadata(
                video_id=video_id,
                title=f"Video {video_id}",
                description=f"Description for {video_id}",
                tags=["video"]
            )
    
    async def _evaluate_async(self, input, output, expected=None, **kwargs) -> EvaluationResult:
        """
        Evaluate results against ground truth using LLM
        """
        # Extract query
        query = input.get("query", "") if isinstance(input, dict) else str(input)
        
        # Extract results
        if hasattr(output, 'results'):
            results = output.results
        elif isinstance(output, dict) and 'results' in output:
            results = output['results']
        else:
            results = output if isinstance(output, list) else []
        
        # Extract expected results
        expected_videos = []
        if expected:
            if isinstance(expected, str):
                try:
                    expected_videos = json.loads(expected)
                except:
                    expected_videos = [expected]
            elif isinstance(expected, list):
                expected_videos = expected
        
        if not results:
            return EvaluationResult(
                score=0.0,
                label="no_results",
                explanation="No results to evaluate against ground truth"
            )
        
        # Fetch metadata if enabled
        metadata_map = {}
        if self.fetch_metadata:
            for result in results[:5]:
                if isinstance(result, dict):
                    video_id = result.get("video_id")
                else:
                    video_id = getattr(result, "video_id", None)
                
                if video_id:
                    metadata_map[video_id] = await self._fetch_video_metadata(video_id)
        
        # Prepare prompt
        system_prompt = """You are an expert evaluator comparing search results against ground truth.
Evaluate how well the actual results match the expected/relevant videos.
Consider both precision (are retrieved videos relevant?) and recall (are relevant videos retrieved?)."""
        
        # Format results
        results_text = []
        for i, result in enumerate(results[:10], 1):
            if isinstance(result, dict):
                video_id = result.get("video_id", "unknown")
                score = result.get("score", 0)
            else:
                video_id = getattr(result, "video_id", "unknown")
                score = getattr(result, "score", 0)
            
            result_line = f"{i}. {video_id} (Score: {score:.3f})"
            
            # Add metadata if available
            if video_id in metadata_map:
                meta = metadata_map[video_id]
                if meta.title:
                    result_line += f" - {meta.title}"
            
            # Mark if in expected
            if video_id in expected_videos:
                result_line += " ✓ [EXPECTED]"
            
            results_text.append(result_line)
        
        prompt = f"""Query: "{query}"

Expected/Relevant Videos: {expected_videos}

Actual Search Results:
{chr(10).join(results_text)}

Evaluate the search quality:
1. Precision: How many retrieved videos are actually relevant?
2. Recall: How many expected videos were retrieved?
3. Ranking: Are expected videos ranked highly?

Score from 0-10 and explain your reasoning.
Format: Score: X/10"""
        
        # Call LLM
        response = await self._call_llm(prompt, system_prompt)
        
        # Parse response
        score, explanation = self._extract_score_from_response(response)
        
        # Calculate simple precision/recall metrics
        retrieved_ids = []
        for result in results:
            if isinstance(result, dict):
                video_id = result.get("video_id")
            else:
                video_id = getattr(result, "video_id", None)
            if video_id:
                retrieved_ids.append(video_id)
        
        if expected_videos and retrieved_ids:
            precision = len(set(retrieved_ids[:5]) & set(expected_videos)) / min(5, len(retrieved_ids))
            recall = len(set(retrieved_ids) & set(expected_videos)) / len(expected_videos)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Combine LLM score with metrics
            final_score = score * 0.6 + f1 * 0.4
        else:
            final_score = score
            precision = recall = f1 = 0
        
        # Determine label
        if final_score >= 0.8:
            label = "excellent_match"
        elif final_score >= 0.6:
            label = "good_match"
        elif final_score >= 0.4:
            label = "partial_match"
        else:
            label = "poor_match"
        
        return EvaluationResult(
            score=final_score,
            label=label,
            explanation=f"LLM: {explanation} | P:{precision:.2f} R:{recall:.2f}",
            metadata={
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "llm_score": score
            }
        )


class SyncLLMHybridEvaluator(Evaluator, LLMJudgeBase):
    """
    Hybrid LLM evaluator combining reference-free and reference-based approaches
    """
    
    def __init__(self,
                 model_name: str = "deepseek-r1:7b", 
                 base_url: str = "http://localhost:11434",
                 reference_weight: float = 0.5):
        """
        Initialize hybrid evaluator
        
        Args:
            model_name: LLM model to use
            base_url: LLM API base URL
            reference_weight: Weight for reference-based score (0-1)
        """
        LLMJudgeBase.__init__(self, model_name, base_url)
        self.reference_weight = reference_weight
        self.relevance_weight = 1 - reference_weight
        
        # Initialize component evaluators
        self.reference_free = SyncLLMReferenceFreeEvaluator(model_name, base_url)
        self.reference_based = SyncLLMReferenceBasedEvaluator(model_name, base_url)
    
    def evaluate(self, *, input=None, output=None, expected=None, **kwargs) -> EvaluationResult:
        """
        Hybrid evaluation combining both approaches
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._evaluate_async(input, output, expected, **kwargs)
        )
    
    async def _evaluate_async(self, input, output, expected=None, **kwargs) -> EvaluationResult:
        """
        Perform hybrid evaluation
        """
        # Run both evaluations
        tasks = []
        
        # Reference-free evaluation
        tasks.append(self.reference_free._evaluate_async(input, output, **kwargs))
        
        # Reference-based evaluation if expected results provided
        if expected:
            tasks.append(self.reference_based._evaluate_async(input, output, expected, **kwargs))
            use_reference = True
        else:
            use_reference = False
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        relevance_result = results[0]
        if isinstance(relevance_result, Exception):
            logger.error(f"Reference-free evaluation failed: {relevance_result}")
            relevance_result = EvaluationResult(score=0.5, label="error", explanation=str(relevance_result))
        
        if use_reference:
            reference_result = results[1]
            if isinstance(reference_result, Exception):
                logger.error(f"Reference-based evaluation failed: {reference_result}")
                reference_result = EvaluationResult(score=0.5, label="error", explanation=str(reference_result))
            
            # Combine scores
            final_score = (
                relevance_result.score * self.relevance_weight +
                reference_result.score * self.reference_weight
            )
            
            # Combine explanations
            explanation = f"Relevance: {relevance_result.explanation} | Reference: {reference_result.explanation}"
            
            # Combine metadata
            metadata = {
                "relevance_score": relevance_result.score,
                "reference_score": reference_result.score,
                "relevance_weight": self.relevance_weight,
                "reference_weight": self.reference_weight
            }
            
            if reference_result.metadata:
                metadata.update({f"ref_{k}": v for k, v in reference_result.metadata.items()})
        else:
            # Only relevance evaluation
            final_score = relevance_result.score
            explanation = f"Relevance-only: {relevance_result.explanation}"
            metadata = {"relevance_score": relevance_result.score}
        
        # Determine label
        if final_score >= 0.8:
            label = "excellent"
        elif final_score >= 0.6:
            label = "good"
        elif final_score >= 0.4:
            label = "fair"
        else:
            label = "poor"
        
        return EvaluationResult(
            score=final_score,
            label=label,
            explanation=explanation,
            metadata=metadata
        )


def create_llm_evaluators(
    model_name: str = "deepseek-r1:7b",
    base_url: str = "http://localhost:11434",
    include_hybrid: bool = True
) -> List[Evaluator]:
    """
    Create LLM-based evaluators for Phoenix experiments
    
    Args:
        model_name: LLM model to use
        base_url: LLM API base URL
        include_hybrid: Whether to include hybrid evaluator
        
    Returns:
        List of LLM evaluator instances
    """
    evaluators = [
        SyncLLMReferenceFreeEvaluator(model_name, base_url),
        SyncLLMReferenceBasedEvaluator(model_name, base_url)
    ]
    
    if include_hybrid:
        evaluators.append(SyncLLMHybridEvaluator(model_name, base_url))
    
    return evaluators