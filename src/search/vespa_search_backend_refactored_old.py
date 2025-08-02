"""
Refactored Vespa search backend that integrates with existing schema infrastructure.

This module provides a production-ready Vespa search backend that:
- Uses ranking strategies extracted from schema JSON files
- Integrates with existing schema profile mappings
- Provides proper error handling without fallbacks
- Separates concerns for query building, tensor formatting, and result processing
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
from enum import Enum
from vespa.application import Vespa

from .search import SearchBackend, SearchResult
from src.core import Document, MediaType, TemporalInfo, SegmentInfo
from src.processing.vespa.schema_profile_mapping import get_profile_for_schema

logger = logging.getLogger(__name__)


class SearchStrategyType(Enum):
    """Types of search strategies"""
    PURE_VISUAL = "pure_visual"
    PURE_TEXT = "pure_text"
    HYBRID = "hybrid"


@dataclass
class RankingConfig:
    """Configuration for a ranking strategy"""
    name: str
    strategy_type: SearchStrategyType
    embedding_field: Optional[str] = None
    query_tensor_name: Optional[str] = None
    needs_float_embeddings: bool = False
    needs_binary_embeddings: bool = False
    needs_text_query: bool = False
    use_nearestneighbor: bool = False
    nearestneighbor_field: Optional[str] = None
    nearestneighbor_tensor: Optional[str] = None
    timeout: float = 2.0
    description: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankingConfig":
        """Create RankingConfig from dictionary"""
        data = data.copy()
        # Convert strategy_type string to enum
        if "strategy_type" in data:
            data["strategy_type"] = SearchStrategyType(data["strategy_type"])
        return cls(**data)


class ProfileConfig:
    """Configuration for different profile types"""
    def __init__(self, profile_name: str, schema_name: str):
        self.profile_name = profile_name
        self.schema_name = schema_name
        self.is_global = "global" in schema_name
        self.is_patch_based = not self.is_global
        self.supports_nearestneighbor = self.is_global
        
        # Determine embedding dimensions based on schema
        if "colpali" in schema_name or "colqwen" in schema_name:
            self.num_patches = 16 if "colpali" in profile_name else 1024
            self.embedding_dim = 128
            self.binary_dim = 16
        elif "videoprism" in schema_name:
            if "large" in schema_name:
                self.embedding_dim = 1024
                self.binary_dim = 128
            else:
                self.embedding_dim = 768
                self.binary_dim = 96
        else:
            # Default dimensions
            self.num_patches = 16
            self.embedding_dim = 128
            self.binary_dim = 16


class TensorFormatter(Protocol):
    """Protocol for tensor formatting strategies"""
    def format(self, embeddings: np.ndarray, profile: ProfileConfig) -> Dict[str, Any]:
        ...


class FloatTensorFormatter:
    """Formats float embeddings for Vespa"""
    def format(self, embeddings: np.ndarray, profile: ProfileConfig) -> Dict[str, Any]:
        if profile.is_global:
            # Global models: return as list
            return embeddings.tolist()
        else:
            # Patch-based models: return as cells dict
            cells = []
            for patch_idx in range(embeddings.shape[0]):
                for v_idx in range(embeddings.shape[1]):
                    cells.append({
                        "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                        "value": float(embeddings[patch_idx, v_idx])
                    })
            return {"cells": cells}


class BinaryTensorFormatter:
    """Formats binary embeddings for Vespa"""
    def format(self, embeddings: np.ndarray, profile: ProfileConfig) -> Dict[str, Any]:
        if profile.is_global:
            # Global models: return as list
            return embeddings.tolist()
        else:
            # Patch-based models: return as cells dict
            cells = []
            for patch_idx in range(embeddings.shape[0]):
                for v_idx in range(embeddings.shape[1]):
                    cells.append({
                        "address": {"querytoken": str(patch_idx), "v": str(v_idx)},
                        "value": int(embeddings[patch_idx, v_idx])
                    })
            return {"cells": cells}


class QueryBuilder:
    """Builds Vespa queries based on strategy configuration"""
    
    def __init__(self, schema_name: str, profile_config: ProfileConfig):
        self.schema_name = schema_name
        self.profile_config = profile_config
    
    def build_yql(
        self,
        strategy: RankingConfig,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build YQL query based on strategy"""
        base_fields = self._get_base_fields()
        
        if strategy.strategy_type == SearchStrategyType.PURE_TEXT:
            yql = f"select {', '.join(base_fields)} from {self.schema_name} where userInput(@userQuery)"
        elif strategy.strategy_type == SearchStrategyType.PURE_VISUAL:
            if strategy.use_nearestneighbor and self.profile_config.supports_nearestneighbor:
                yql = f"select {', '.join(base_fields)} from {self.schema_name} where {{targetHits: {top_k}}}nearestNeighbor({strategy.nearestneighbor_field}, {strategy.nearestneighbor_tensor})"
            else:
                yql = f"select {', '.join(base_fields)} from {self.schema_name} where true"
        else:  # HYBRID
            if strategy.use_nearestneighbor and self.profile_config.supports_nearestneighbor:
                yql = f"select {', '.join(base_fields)} from {self.schema_name} where {{targetHits: {top_k}}}nearestNeighbor({strategy.nearestneighbor_field}, {strategy.nearestneighbor_tensor})"
            else:
                yql = f"select {', '.join(base_fields)} from {self.schema_name} where userInput(@userQuery)"
        
        # Add filters
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
            if filter_clauses:
                if " where " in yql and not yql.endswith(" where true"):
                    yql += " AND " + " AND ".join(filter_clauses)
                else:
                    yql = yql.replace(" where true", f" where {' AND '.join(filter_clauses)}")
        
        return yql
    
    def _get_base_fields(self) -> List[str]:
        """Get base fields to select based on schema"""
        fields = ["video_id", "video_title", "start_time", "end_time"]
        
        # Add schema-specific fields
        if "colqwen" in self.schema_name:
            fields.append("segment_id")
        elif "videoprism" in self.schema_name:
            fields.append("frame_id")
        else:  # Default/ColPali
            fields.extend(["frame_id", "frame_description", "audio_transcript"])
        
        return fields
    
    def _build_filter_clauses(self, filters: Dict[str, Any]) -> List[str]:
        """Build filter clauses from filter dict"""
        clauses = []
        
        if "start_date" in filters:
            clauses.append(f"creation_timestamp >= {filters['start_date']}")
        if "end_date" in filters:
            clauses.append(f"creation_timestamp <= {filters['end_date']}")
        
        return clauses


class VespaSearchBackend(SearchBackend):
    """Production-ready Vespa search backend integrated with schema infrastructure"""
    
    def __init__(
        self,
        vespa_url: str,
        vespa_port: int,
        schema_name: str,
        profile: Optional[str] = None
    ):
        """
        Initialize Vespa search backend.
        
        Args:
            vespa_url: Vespa URL
            vespa_port: Vespa port
            schema_name: Vespa schema to search
            profile: Video processing profile (optional - will be derived from schema if not provided)
        """
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.schema_name = schema_name
        
        # Get profile from schema if not provided
        if profile is None:
            try:
                profile = get_profile_for_schema(schema_name)
            except ValueError as e:
                raise ValueError(f"Cannot determine profile for schema {schema_name}: {e}")
        
        self.profile = profile
        
        # Load ranking strategies
        self._load_ranking_strategies()
        
        # Initialize configurations
        self.profile_config = ProfileConfig(profile, schema_name)
        
        # Initialize components
        self.query_builder = QueryBuilder(schema_name, self.profile_config)
        self.float_formatter = FloatTensorFormatter()
        self.binary_formatter = BinaryTensorFormatter()
        
        # Connect to Vespa
        self.app = None
        self._connect()
    
    def _load_ranking_strategies(self):
        """Load ranking strategies from JSON file"""
        strategies_file = Path("configs/schemas/ranking_strategies.json")
        
        if not strategies_file.exists():
            # Auto-generate the ranking strategies file
            logger.warning(f"Ranking strategies file not found at {strategies_file}. Generating it now...")
            
            from src.processing.vespa.ranking_strategy_extractor import extract_all_ranking_strategies, save_ranking_strategies
            
            schemas_dir = Path("configs/schemas")
            if not schemas_dir.exists():
                raise FileNotFoundError(f"Schema directory not found at {schemas_dir}")
            
            strategies = extract_all_ranking_strategies(schemas_dir)
            save_ranking_strategies(strategies, strategies_file)
            logger.info(f"Generated ranking strategies file with {sum(len(s) for s in strategies.values())} strategies")
        
        with open(strategies_file, 'r') as f:
            all_strategies = json.load(f)
        
        if self.schema_name not in all_strategies:
            raise ValueError(
                f"No ranking strategies found for schema '{self.schema_name}'. "
                f"Available schemas: {list(all_strategies.keys())}"
            )
        
        # Convert to RankingConfig objects
        self.ranking_strategies = {}
        for name, strategy_data in all_strategies[self.schema_name].items():
            self.ranking_strategies[name] = RankingConfig.from_dict(strategy_data)
        
        logger.info(f"Loaded {len(self.ranking_strategies)} ranking strategies for schema {self.schema_name}")
    
    def _connect(self):
        """Connect to Vespa."""
        try:
            self.app = Vespa(url=self.vespa_url, port=self.vespa_port)
            logger.info(f"Connected to Vespa at {self.vespa_url}:{self.vespa_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Vespa: {e}")
            raise
    
    def _validate_inputs(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: Optional[str],
        ranking_strategy: str
    ) -> None:
        """Validate inputs for the given ranking strategy. Raises ValueError if invalid."""
        
        if ranking_strategy not in self.ranking_strategies:
            available = list(self.ranking_strategies.keys())
            raise ValueError(
                f"Unknown ranking strategy '{ranking_strategy}' for schema '{self.schema_name}'. "
                f"Available strategies: {available}"
            )
        
        strategy = self.ranking_strategies[ranking_strategy]
        
        # Validate text query
        if strategy.needs_text_query and not query_text:
            raise ValueError(f"Strategy '{ranking_strategy}' requires a text query")
        
        # Validate embeddings
        if (strategy.needs_float_embeddings or strategy.needs_binary_embeddings) and query_embeddings is None:
            raise ValueError(f"Strategy '{ranking_strategy}' requires embeddings")
        
        # Validate embedding shape
        if query_embeddings is not None:
            if self.profile_config.is_global and query_embeddings.ndim != 1:
                raise ValueError(
                    f"Global profile '{self.profile}' expects 1D embeddings, "
                    f"got shape {query_embeddings.shape}"
                )
            elif self.profile_config.is_patch_based and query_embeddings.ndim != 2:
                raise ValueError(
                    f"Patch-based profile '{self.profile}' expects 2D embeddings, "
                    f"got shape {query_embeddings.shape}"
                )
    
    def _generate_binary_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate binary embeddings from float embeddings."""
        # Binarize embeddings (>0 becomes 1, <=0 becomes 0)
        binary = np.where(embeddings > 0, 1, 0).astype(np.uint8)
        
        # Pack bits into bytes
        if embeddings.ndim == 1:
            # 1D embeddings (global)
            # Pad to multiple of 8 bits
            padding = (8 - len(binary) % 8) % 8
            if padding:
                binary = np.pad(binary, (0, padding), mode='constant')
            # Pack bits
            packed = np.packbits(binary).astype(np.int8)
        else:
            # 2D embeddings (patch-based)
            # Pack each patch separately
            packed = np.packbits(binary, axis=1).astype(np.int8)
        
        return packed
    
    def search(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query_embeddings: Query embeddings (optional)
            query_text: Query text (optional)
            top_k: Number of results to return
            filters: Additional filters
            ranking_strategy: Ranking strategy to use
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If inputs are invalid for the strategy
            RuntimeError: If not connected to Vespa
        """
        if not self.app:
            raise RuntimeError("Not connected to Vespa")
        
        # Determine ranking strategy
        if not ranking_strategy:
            ranking_strategy = self._get_default_strategy()
        
        # Validate inputs - will raise if invalid
        self._validate_inputs(query_embeddings, query_text, ranking_strategy)
        
        strategy = self.ranking_strategies[ranking_strategy]
        
        # Build base query
        yql = self.query_builder.build_yql(strategy, top_k, filters)
        
        query_body = {
            "yql": yql,
            "ranking.profile": strategy.name,
            "hits": top_k,
            "model.restrict": self.schema_name  # Must be string, not list
        }
        
        # Add text query if needed
        if strategy.needs_text_query and query_text:
            query_body["userQuery"] = query_text
        
        # Add embeddings if needed
        if query_embeddings is not None:
            if strategy.needs_float_embeddings:
                tensor_data = self.float_formatter.format(query_embeddings, self.profile_config)
                query_body[f"input.query({strategy.query_tensor_name})"] = tensor_data
            
            if strategy.needs_binary_embeddings:
                binary_embeddings = self._generate_binary_embeddings(query_embeddings)
                tensor_data = self.binary_formatter.format(binary_embeddings, self.profile_config)
                query_body["input.query(qtb)"] = tensor_data
        
        # Execute search
        try:
            response = self.app.query(body=query_body)
            return self._process_results(response)
        except Exception as e:
            logger.error(f"Search failed with strategy '{ranking_strategy}': {e}")
            raise
    
    def _get_default_strategy(self) -> str:
        """Get default ranking strategy based on profile"""
        # Try common strategies in order of preference
        preference_order = [
            "hybrid_binary_bm25",  # For patch-based
            "float_float",         # For global
            "binary_binary",       # Fast fallback
            "default"              # Ultimate fallback
        ]
        
        for strategy_name in preference_order:
            if strategy_name in self.ranking_strategies:
                return strategy_name
        
        # If nothing found, just return the first available
        return list(self.ranking_strategies.keys())[0]
    
    def _process_results(self, response) -> List[SearchResult]:
        """Process Vespa response into SearchResult objects"""
        results = []
        
        for hit in response.hits:
            doc = self._result_to_document(hit)
            score = hit.get("relevance", 0.0)
            
            # Extract highlights if available
            highlights = {}
            if "summaryfeatures" in hit:
                highlights = hit["summaryfeatures"]
            
            results.append(SearchResult(doc, score, highlights))
        
        return results
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert Vespa result to Document object."""
        fields = result.get("fields", {})
        
        # Extract document ID
        doc_id = result.get("id", "").split("::")[-1]
        
        # Determine media type
        if "frame" in self.schema_name:
            media_type = MediaType.VIDEO_FRAME
        elif "global" in self.schema_name:
            media_type = MediaType.VIDEO_SEGMENT
        else:
            media_type = MediaType.VIDEO_FRAME
        
        # Build temporal info if available
        temporal_info = None
        if "start_time" in fields and "end_time" in fields:
            temporal_info = TemporalInfo(
                start_time=fields["start_time"],
                end_time=fields["end_time"]
            )
        
        # Build segment info if available
        segment_info = None
        if "segment_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["segment_id"],
                total_segments=fields.get("total_segments", 1)
            )
        elif "frame_id" in fields:
            segment_info = SegmentInfo(
                segment_idx=fields["frame_id"],
                total_segments=1
            )
        
        # Extract metadata
        metadata = {
            "video_title": fields.get("video_title"),
            "frame_description": fields.get("frame_description"),
            "segment_description": fields.get("segment_description"),
            "audio_transcript": fields.get("audio_transcript"),
            "creation_timestamp": fields.get("creation_timestamp"),
            "source_id": fields.get("video_id", doc_id.split("_")[0])
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Create Document
        return Document(
            doc_id=doc_id,
            media_type=media_type,
            temporal_info=temporal_info,
            segment_info=segment_info,
            metadata=metadata
        )
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        if not self.app:
            raise RuntimeError("Not connected to Vespa")
        
        try:
            # Construct full document ID
            full_id = f"id:video:{self.schema_name}::{document_id}"
            
            # Get document
            response = self.app.get_data(data_id=full_id, schema=self.schema_name)
            
            if response and response.is_successful():
                # Create fake result dict for conversion
                result = {
                    "id": full_id,
                    "fields": response.get_json()
                }
                return self._result_to_document(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_available_strategies(self) -> Dict[str, str]:
        """Get available ranking strategies with descriptions"""
        return {
            name: config.description 
            for name, config in self.ranking_strategies.items()
        }