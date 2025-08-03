#!/usr/bin/env python3
"""
Strategy Configuration - Single source of truth for ALL strategies

This replaces the fragmented strategy systems:
- StrategyResolver (processing strategies)  
- StrategyAwareProcessor (ranking strategies)
- Schema-based strategies

Provides a coherent interface for:
1. Processing strategies (how to segment/process video)
2. Ranking strategies (how to search/rank results)
3. Storage strategies (how to store embeddings)
"""

from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Complete strategy configuration for a profile"""
    
    # Processing strategy
    processing_type: Literal["single_vector", "frame_based", "direct_video"]
    segmentation: Literal["chunks", "windows", "global", "frames"]
    
    # Storage strategy
    storage_mode: Literal["single_doc", "multi_doc"]
    schema_name: str
    
    # Ranking strategies available for this schema
    ranking_strategies: Dict[str, Dict[str, Any]]
    default_ranking: str
    
    # Model configuration
    model_name: str
    model_config: Dict[str, Any]
    
    # Embedding requirements (derived from ranking strategies)
    needs_float_embeddings: bool
    needs_binary_embeddings: bool
    embedding_fields: Dict[str, str]  # float_field, binary_field names
    
    def __repr__(self):
        return (
            f"Strategy(processing={self.processing_type}/{self.segmentation}, "
            f"storage={self.storage_mode}, schema={self.schema_name})"
        )


class StrategyConfig:
    """
    Manages all strategy configurations in one place.
    This is THE single source of truth for how to process, store, and search videos.
    """
    
    def __init__(self, config_dir: Path = Path("configs")):
        self.config_dir = config_dir
        self._load_all_configs()
        self._strategy_cache = {}
    
    def _load_all_configs(self):
        """Load all configuration files"""
        # Main config
        with open(self.config_dir / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Ranking strategies
        ranking_strategies_path = self.config_dir / "schemas" / "ranking_strategies.json"
        if ranking_strategies_path.exists():
            with open(ranking_strategies_path, 'r') as f:
                self.ranking_strategies = json.load(f)
        else:
            # Generate if missing
            from src.processing.vespa.ranking_strategy_extractor import (
                extract_all_ranking_strategies, save_ranking_strategies
            )
            schemas_dir = self.config_dir / "schemas"
            strategies = extract_all_ranking_strategies(schemas_dir)
            save_ranking_strategies(strategies, ranking_strategies_path)
            logger.info("Generated ranking strategies file")
            
            # Now load from the saved JSON file
            with open(ranking_strategies_path, 'r') as f:
                self.ranking_strategies = json.load(f)
    
    def get_strategy(self, profile_name: str) -> Strategy:
        """
        Get complete unified strategy for a profile.
        This is the main API - everything else is internal.
        """
        # Check cache
        if profile_name in self._strategy_cache:
            return self._strategy_cache[profile_name]
        
        # Get profile config
        profiles = self.config.get("video_processing_profiles", {})
        profile = profiles.get(profile_name)
        
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        # Resolve processing strategy
        processing_type, segmentation = self._resolve_processing_strategy(profile_name, profile)
        
        # Resolve storage strategy
        storage_mode = self._resolve_storage_mode(profile, processing_type)
        
        # Get schema and ranking strategies
        schema_name = profile.get("vespa_schema", "video_frame")
        schema_rankings = self.ranking_strategies.get(schema_name, {})
        
        # Determine embedding requirements from ranking strategies
        embedding_reqs = self._analyze_embedding_requirements(schema_rankings)
        
        # Create strategy
        strategy = Strategy(
            processing_type=processing_type,
            segmentation=segmentation,
            storage_mode=storage_mode,
            schema_name=schema_name,
            ranking_strategies=schema_rankings,
            default_ranking=self._get_default_ranking(schema_rankings),
            model_name=profile.get("embedding_model", ""),
            model_config=profile.get("model_specific", {}),
            needs_float_embeddings=embedding_reqs["needs_float"],
            needs_binary_embeddings=embedding_reqs["needs_binary"],
            embedding_fields=embedding_reqs["field_names"]
        )
        
        # Cache it
        self._strategy_cache[profile_name] = strategy
        
        return strategy
    
    def _resolve_processing_strategy(
        self, 
        profile_name: str, 
        profile: Dict[str, Any]
    ) -> tuple[str, str]:
        """Resolve processing type and segmentation strategy"""
        
        # 1. Check for explicit processing_strategy (future format)
        if "processing_strategy" in profile:
            ps = profile["processing_strategy"]
            return ps.get("type", "frame_based"), ps.get("segmentation", "frames")
        
        # 2. Check processing_type field
        if "processing_type" in profile:
            proc_type = profile["processing_type"]
            if proc_type == "single_vector":
                seg = self._infer_segmentation(profile_name, profile)
                return "single_vector", seg
        
        # 3. Infer from profile name (legacy)
        if profile_name.startswith("single__"):
            seg = self._infer_segmentation(profile_name, profile)
            return "single_vector", seg
        elif profile_name.startswith("direct_video_"):
            return "direct_video", "windows"
        elif profile_name.startswith(("colpali__", "colvision__")):
            return "frame_based", "frames"
        
        # 4. Check process_type in profile (legacy)
        process_type = profile.get("process_type", "")
        if process_type.startswith("direct_video"):
            return "direct_video", "windows"
        elif process_type == "frame_based":
            return "frame_based", "frames"
        
        # Default
        return "frame_based", "frames"
    
    def _infer_segmentation(self, profile_name: str, profile: Dict[str, Any]) -> str:
        """Infer segmentation strategy"""
        # From name
        name_lower = profile_name.lower()
        if "global" in name_lower:
            return "global"
        elif "window" in name_lower:
            return "windows"
        elif "chunk" in name_lower or "6s" in name_lower:
            return "chunks"
        
        # From config
        model_config = profile.get("model_specific", {})
        if model_config.get("global_embedding", False):
            return "global"
        elif model_config.get("chunk_duration", 0) > 0:
            return "chunks"
        elif model_config.get("segment_duration", 0) > 20:
            return "windows"
        
        return "frames"
    
    def _resolve_storage_mode(self, profile: Dict[str, Any], processing_type: str) -> str:
        """Resolve storage mode"""
        # Explicit in model config
        if "store_as_single_doc" in profile.get("model_specific", {}):
            return "single_doc" if profile["model_specific"]["store_as_single_doc"] else "multi_doc"
        
        # Based on processing type
        if processing_type == "single_vector":
            # Single vector typically uses single doc for chunks
            return "single_doc"
        
        return "multi_doc"
    
    def _analyze_embedding_requirements(
        self, 
        schema_rankings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze what embeddings are needed based on ranking strategies"""
        needs_float = False
        needs_binary = False
        float_fields = set()
        binary_fields = set()
        
        for strategy in schema_rankings.values():
            if strategy.get("needs_float_embeddings", False):
                needs_float = True
                field = strategy.get("embedding_field", "")
                if field and "binary" not in field:
                    float_fields.add(field)
            
            if strategy.get("needs_binary_embeddings", False):
                needs_binary = True
                field = strategy.get("embedding_field", "")
                if field and "binary" in field:
                    binary_fields.add(field)
        
        return {
            "needs_float": needs_float,
            "needs_binary": needs_binary,
            "field_names": {
                "float_field": next(iter(float_fields), "embedding"),
                "binary_field": next(iter(binary_fields), "embedding_binary")
            }
        }
    
    def _get_default_ranking(self, schema_rankings: Dict[str, Dict[str, Any]]) -> str:
        """Get default ranking strategy"""
        if "default" in schema_rankings:
            return "default"
        elif "hybrid" in schema_rankings:
            return "hybrid"
        elif schema_rankings:
            return next(iter(schema_rankings.keys()))
        return "default"
    
    def get_processing_config(self, profile_name: str) -> Dict[str, Any]:
        """Get processing configuration for pipeline"""
        strategy = self.get_strategy(profile_name)
        
        return {
            "processing_type": strategy.processing_type,
            "segmentation": strategy.segmentation,
            "storage_mode": strategy.storage_mode,
            "model_config": strategy.model_config,
            "needs_embeddings": {
                "float": strategy.needs_float_embeddings,
                "binary": strategy.needs_binary_embeddings
            }
        }
    
    def get_ranking_strategies(self, profile_name: str) -> Dict[str, Dict[str, Any]]:
        """Get available ranking strategies for a profile"""
        strategy = self.get_strategy(profile_name)
        return strategy.ranking_strategies
    
    def get_embedding_fields(self, profile_name: str) -> Dict[str, str]:
        """Get embedding field names for a profile"""
        strategy = self.get_strategy(profile_name)
        return strategy.embedding_fields