"""
Ranking strategy extractor for Vespa schemas.
Extracts ranking profile configurations from schema JSON files for use by search backends.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SearchStrategyType(Enum):
    """Types of search strategies"""
    PURE_VISUAL = "pure_visual"
    PURE_TEXT = "pure_text"
    HYBRID = "hybrid"


@dataclass
class RankingStrategyInfo:
    """Information about a ranking strategy extracted from schema"""
    name: str
    strategy_type: SearchStrategyType
    needs_float_embeddings: bool = False
    needs_binary_embeddings: bool = False
    needs_text_query: bool = False
    use_nearestneighbor: bool = False
    nearestneighbor_field: Optional[str] = None
    nearestneighbor_tensor: Optional[str] = None
    embedding_field: Optional[str] = None
    query_tensor_name: Optional[str] = None
    timeout: float = 2.0
    description: str = ""
    # New comprehensive fields
    inputs: Dict[str, str] = field(default_factory=dict)  # Full input definitions
    query_tensors_needed: List[str] = field(default_factory=list)  # List of tensor names needed
    schema_name: str = ""  # Schema this strategy belongs to


class RankingStrategyExtractor:
    """Extracts ranking strategies from Vespa schema JSON files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_from_schema(self, schema_path: Path) -> Dict[str, RankingStrategyInfo]:
        """Extract ranking strategies from a schema JSON file"""
        
        with open(schema_path, 'r') as f:
            schema_json = json.load(f)
        
        schema_name = schema_json.get("schema", schema_json.get("name", ""))
        is_global = "global" in schema_name
        
        # Extract field information
        fields = {f["name"]: f for f in schema_json["document"]["fields"]}
        
        strategies = {}
        
        for profile in schema_json.get("rank-profiles", schema_json.get("rank_profiles", [])):
            strategy_info = self._parse_ranking_profile(profile, fields, is_global, schema_json)
            strategies[strategy_info.name] = strategy_info
        
        return strategies
    
    def _parse_ranking_profile(self, profile: Dict[str, Any], fields: Dict[str, Dict], is_global: bool, schema_json: Dict[str, Any]) -> RankingStrategyInfo:
        """Parse a single ranking profile"""
        
        profile_name = profile["name"]
        
        # Parse inputs to determine what's needed
        inputs = {}
        for input_def in profile.get("inputs", []):
            input_name = input_def["name"]
            # Extract parameter name from "query(qt)" -> "qt"
            if "(" in input_name and ")" in input_name:
                param_name = input_name[input_name.find("(")+1:input_name.find(")")]
            else:
                param_name = input_name
            inputs[param_name] = input_def["type"]
        
        # Determine what the profile needs
        needs_float_embeddings = any("float" in t for t in inputs.values())
        needs_binary_embeddings = any("int8" in t for t in inputs.values())
        
        # Detect if it needs text query based on profile name or first phase expression
        first_phase = profile.get("first-phase", profile.get("first_phase", {}))
        # Handle both string and dict formats
        if isinstance(first_phase, dict):
            first_phase_expr = first_phase.get("expression", "")
        else:
            first_phase_expr = str(first_phase)
            
        needs_text_query = (
            "bm25" in profile_name.lower() or
            "bm25(" in first_phase_expr or
            "userInput" in first_phase_expr or
            "text" in profile_name.lower()
        )
        
        # Determine strategy type
        if needs_text_query and not (needs_float_embeddings or needs_binary_embeddings):
            strategy_type = SearchStrategyType.PURE_TEXT
        elif (needs_float_embeddings or needs_binary_embeddings) and not needs_text_query:
            strategy_type = SearchStrategyType.PURE_VISUAL
        else:
            strategy_type = SearchStrategyType.HYBRID
        
        # Determine if uses nearestNeighbor based on schema and strategy
        use_nearestneighbor = False
        nearestneighbor_field = None
        nearestneighbor_tensor = None
        
        # Check for video_chunks schema specifically
        schema_name = schema_json.get("schema", "")
        is_video_chunks = schema_name == "video_chunks"
        
        # For global schemas OR video_chunks, visual strategies use nearestNeighbor
        if (is_global or is_video_chunks) and strategy_type in [SearchStrategyType.PURE_VISUAL, SearchStrategyType.HYBRID]:
            if profile_name in ["float_float", "binary_binary", "float_binary", "phased", 
                               "hybrid_float_bm25", "hybrid_binary_bm25"]:
                use_nearestneighbor = True
                
                # Determine field and tensor based on profile AND schema
                if profile_name == "float_binary" and is_video_chunks:
                    # Special case: video_chunks float_binary uses float embeddings
                    nearestneighbor_field = "embedding"
                    nearestneighbor_tensor = "qt"
                elif profile_name == "float_float":
                    nearestneighbor_field = "embedding"
                    nearestneighbor_tensor = "qt"
                elif profile_name in ["binary_binary", "phased"]:
                    nearestneighbor_field = "embedding_binary"
                    nearestneighbor_tensor = "qtb"
                elif "qt" in inputs and not ("qtb" in inputs and "binary" in profile_name):
                    nearestneighbor_field = "embedding"
                    nearestneighbor_tensor = "qt"
                elif "qtb" in inputs:
                    nearestneighbor_field = "embedding_binary"
                    nearestneighbor_tensor = "qtb"
        
        # Determine primary embedding field and query tensor
        embedding_field = None
        query_tensor_name = None
        
        if "q" in inputs:
            query_tensor_name = "q"
            embedding_field = "embeddings"  # Note: plural for chunks schema
        elif "qt" in inputs:
            query_tensor_name = "qt"
            embedding_field = "embedding"
        elif "qtb" in inputs:
            query_tensor_name = "qtb"
            embedding_field = "embedding_binary"
        
        # Generate description
        description = self._generate_description(profile_name, strategy_type, needs_float_embeddings, needs_binary_embeddings)
        
        # Build list of query tensors needed
        query_tensors_needed = list(inputs.keys())
        
        return RankingStrategyInfo(
            name=profile_name,
            strategy_type=strategy_type,
            needs_float_embeddings=needs_float_embeddings,
            needs_binary_embeddings=needs_binary_embeddings,
            needs_text_query=needs_text_query,
            use_nearestneighbor=use_nearestneighbor,
            nearestneighbor_field=nearestneighbor_field,
            nearestneighbor_tensor=nearestneighbor_tensor,
            embedding_field=embedding_field,
            query_tensor_name=query_tensor_name,
            timeout=profile.get("timeout", 2.0),
            description=description,
            inputs=inputs,
            query_tensors_needed=query_tensors_needed,
            schema_name=schema_name
        )
    
    def _generate_description(self, profile_name: str, strategy_type: SearchStrategyType, 
                            needs_float: bool, needs_binary: bool) -> str:
        """Generate human-readable description"""
        
        descriptions = {
            "default": "Default ranking profile",
            "bm25_only": "Pure text search using BM25",
            "bm25_no_description": "BM25 text search excluding descriptions",
            "float_float": "Visual search with float embeddings",
            "binary_binary": "Fast visual search with binary embeddings",
            "float_binary": "Float query with binary document embeddings",
            "phased": "Two-phase ranking: binary first, float reranking",
            "hybrid_float_bm25": "Combined visual (float) and text search",
            "hybrid_binary_bm25": "Combined visual (binary) and text search",
            "hybrid_bm25_binary": "Text-first search with visual reranking",
            "hybrid_bm25_float": "Text-first search with visual reranking"
        }
        
        # Check for no_description variant
        if "no_description" in profile_name:
            base_name = profile_name.replace("_no_description", "")
            if base_name in descriptions:
                return descriptions[base_name] + " (excluding descriptions)"
        
        if profile_name in descriptions:
            return descriptions[profile_name]
        
        # Generate based on strategy type
        if strategy_type == SearchStrategyType.PURE_TEXT:
            return "Text-based search"
        elif strategy_type == SearchStrategyType.PURE_VISUAL:
            if needs_binary:
                return "Binary embedding search"
            else:
                return "Float embedding search"
        else:
            return "Hybrid text and visual search"


def extract_all_ranking_strategies(schema_dir: Path) -> Dict[str, Dict[str, RankingStrategyInfo]]:
    """Extract ranking strategies from all schemas in a directory"""
    
    extractor = RankingStrategyExtractor()
    all_strategies = {}
    
    for schema_file in schema_dir.glob("*.json"):
        # Skip ranking_strategies.json
        if schema_file.name == "ranking_strategies.json":
            continue
            
        try:
            schema_name = schema_file.stem.replace("_schema", "")
            strategies = extractor.extract_from_schema(schema_file)
            all_strategies[schema_name] = strategies
            logger.info(f"Extracted {len(strategies)} strategies from {schema_name}")
        except Exception as e:
            logger.error(f"Failed to extract strategies from {schema_file}: {e}")
    
    return all_strategies


def save_ranking_strategies(strategies: Dict[str, Dict[str, RankingStrategyInfo]], output_path: Path):
    """Save extracted ranking strategies to JSON file"""
    
    # Convert to serializable format
    output = {}
    for schema_name, schema_strategies in strategies.items():
        output[schema_name] = {}
        for strategy_name, strategy_info in schema_strategies.items():
            output[schema_name][strategy_name] = {
                "name": strategy_info.name,
                "strategy_type": strategy_info.strategy_type.value,
                "needs_float_embeddings": strategy_info.needs_float_embeddings,
                "needs_binary_embeddings": strategy_info.needs_binary_embeddings,
                "needs_text_query": strategy_info.needs_text_query,
                "use_nearestneighbor": strategy_info.use_nearestneighbor,
                "nearestneighbor_field": strategy_info.nearestneighbor_field,
                "nearestneighbor_tensor": strategy_info.nearestneighbor_tensor,
                "embedding_field": strategy_info.embedding_field,
                "query_tensor_name": strategy_info.query_tensor_name,
                "timeout": strategy_info.timeout,
                "description": strategy_info.description,
                "inputs": strategy_info.inputs,
                "query_tensors_needed": strategy_info.query_tensors_needed,
                "schema_name": strategy_info.schema_name
            }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved ranking strategies to {output_path}")
