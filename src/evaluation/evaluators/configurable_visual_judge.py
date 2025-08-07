"""
Configurable Visual Judge that uses config to determine provider (Ollama, Modal, etc.)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import base64
import requests
import json

from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult
from src.tools.config import get_config

logger = logging.getLogger(__name__)


class ConfigurableVisualJudge(Evaluator):
    """
    Visual judge that uses configured provider (Ollama, Modal, etc.)
    """
    
    def __init__(self, evaluator_name: str = "visual_judge"):
        """
        Initialize visual judge from config
        
        Args:
            evaluator_name: Name of evaluator config to use
        """
        config = get_config()
        
        # Get evaluator config
        evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})
        if not evaluator_config:
            # Fallback to default
            evaluator_config = {
                "provider": "ollama",
                "model": "llava:7b",
                "base_url": "http://localhost:11434",
                "api_key": None
            }
            logger.warning(f"No config for evaluator '{evaluator_name}', using defaults")
        
        self.provider = evaluator_config.get("provider", "ollama")
        self.model = evaluator_config.get("model", "llava:7b")
        self.base_url = evaluator_config.get("base_url", "http://localhost:11434")
        self.api_key = evaluator_config.get("api_key")
        
        logger.info(f"Initialized {self.provider} visual judge with model {self.model} at {self.base_url}")
    
    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Evaluate visual relevance using configured provider
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
        
        # Get frame paths for top results
        frame_paths = []
        for i, result in enumerate(results[:3], 1):  # Top 3 for evaluation
            frame_path = self._get_frame_path(result)
            if frame_path and Path(frame_path).exists():
                frame_paths.append(frame_path)
        
        if not frame_paths:
            return EvaluationResult(
                score=0.0,
                label="no_frames",
                explanation="No frame images found for evaluation"
            )
        
        # Evaluate based on provider
        try:
            if self.provider == "ollama":
                score, reasoning = self._evaluate_with_ollama(query, frame_paths)
            elif self.provider == "modal":
                score, reasoning = self._evaluate_with_modal(query, frame_paths)
            elif self.provider == "openai":
                score, reasoning = self._evaluate_with_openai(query, frame_paths)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Determine label
            if score >= 0.8:
                label = "excellent_match"
            elif score >= 0.6:
                label = "good_match"
            elif score >= 0.4:
                label = "partial_match"
            else:
                label = "poor_match"
            
            return EvaluationResult(
                score=float(score),
                label=label,
                explanation=f"{self.provider}/{self.model}: {reasoning}",
                metadata={
                    "frames_evaluated": len(frame_paths),
                    "provider": self.provider,
                    "model": self.model
                }
            )
            
        except Exception as e:
            logger.error(f"Visual evaluation failed: {e}")
            return EvaluationResult(
                score=0.0,
                label="evaluation_failed",
                explanation=f"Visual evaluation failed: {str(e)}"
            )
    
    def _get_frame_path(self, result: Dict) -> Optional[str]:
        """Extract frame path from result"""
        if isinstance(result, dict):
            video_id = result.get("video_id", result.get("source_id"))
            frame_id = result.get("frame_id", 0)
            
            # Check if frame_path is provided
            frame_path = result.get("frame_path")
            
            # If not, try to construct it
            if not frame_path and video_id:
                possible_paths = [
                    f"data/frames/{video_id}/frame_{frame_id}.jpg",
                    f"data/frames/{video_id}/frame_{frame_id}.png",
                    f"outputs/frames/{video_id}/frame_{frame_id}.jpg",
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        return path
            
            return frame_path
        
        return None
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def _evaluate_with_ollama(self, query: str, frame_paths: List[str]) -> Tuple[float, str]:
        """
        Evaluate using Ollama (supports LLaVA, Qwen2-VL via Ollama)
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]
        
        prompt = f"""You are evaluating video search results.
The user searched for: "{query}"

I'm showing you {len(frame_paths)} frames from the top search results.

Please evaluate:
1. Do these frames show content that matches "{query}"?
2. How relevant is the visual content to the search query?
3. Rate the match quality from 0 to 10.

Provide your assessment in this format:
SCORE: [0-10]
REASONING: [Your explanation of what you see and how well it matches]"""
        
        # Prepare Ollama API request
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": encoded_images
            }
        ]
        
        # Call Ollama API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            },
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        response_text = result.get("message", {}).get("content", "")
        
        # Parse response
        return self._parse_response(response_text)
    
    def _evaluate_with_modal(self, query: str, frame_paths: List[str]) -> Tuple[float, str]:
        """
        Evaluate using Modal deployment
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]
        
        prompt = f"""Evaluate if these video frames match the search query: "{query}"
Rate from 0-10 and explain what you see.
Format: SCORE: X/10, REASONING: explanation"""
        
        # Call Modal endpoint
        response = requests.post(
            self.base_url,
            json={
                "prompt": prompt,
                "images": encoded_images,
                "model": self.model
            },
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Modal API error: {response.status_code} - {response.text}")
        
        result = response.json()
        response_text = result.get("response", result.get("output", ""))
        
        # Parse response
        return self._parse_response(response_text)
    
    def _evaluate_with_openai(self, query: str, frame_paths: List[str]) -> Tuple[float, str]:
        """
        Evaluate using OpenAI API (GPT-4V)
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Do these video frames match the search query '{query}'? Rate 0-10. Format: SCORE: X/10, REASONING: explanation"
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in encoded_images
                ]
            }
        ]
        
        # Call OpenAI API
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 300
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse response
        return self._parse_response(response_text)
    
    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """Parse score and reasoning from response"""
        import re
        
        score = 0.5  # Default
        reasoning = response_text
        
        # Try to extract score
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = score / 10.0  # Normalize to 0-1
                score = min(max(score, 0.0), 1.0)
            except:
                pass
        
        # Try to extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return score, reasoning


def create_configurable_visual_evaluators(evaluator_name: str = "visual_judge") -> List[Evaluator]:
    """
    Create visual evaluators using configured provider
    
    Args:
        evaluator_name: Name of evaluator config to use
        
    Returns:
        List of visual evaluator instances
    """
    return [
        ConfigurableVisualJudge(evaluator_name)
    ]