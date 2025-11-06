"""
Strategy-aware document processor for Vespa ingestion.
This shows how processing should use the extracted ranking strategies.
"""

import json
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StrategyAwareProcessor:
    """Process documents based on ranking strategy requirements"""

    def __init__(self, schema_loader):
        """
        Initialize strategy-aware processor.

        Args:
            schema_loader: SchemaLoader instance for loading schemas (REQUIRED)
        """
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for StrategyAwareProcessor. "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )
        self.schema_loader = schema_loader
        self.ranking_strategies = self._load_ranking_strategies()

    def _load_ranking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load ranking strategies from JSON file"""
        # Get schemas directory from injected schema_loader
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        if not isinstance(self.schema_loader, FilesystemSchemaLoader):
            raise ValueError(
                f"Unsupported schema_loader type: {type(self.schema_loader)}. "
                f"StrategyAwareProcessor requires FilesystemSchemaLoader."
            )

        schemas_dir = self.schema_loader.base_path
        strategies_file = schemas_dir / "ranking_strategies.json"

        if not strategies_file.exists():
            # Auto-generate if missing
            from .ranking_strategy_extractor import (
                extract_all_ranking_strategies,
                save_ranking_strategies,
            )

            strategies = extract_all_ranking_strategies(schemas_dir)
            save_ranking_strategies(strategies, strategies_file)
            logger.info("Generated ranking strategies file")

        with open(strategies_file, 'r') as f:
            return json.load(f)
    
    def get_required_embeddings(self, schema_name: str) -> Dict[str, bool]:
        """Determine what embeddings are needed based on ranking strategies"""
        
        if schema_name not in self.ranking_strategies:
            raise ValueError(f"No ranking strategies found for schema '{schema_name}'")
        
        schema_strategies = self.ranking_strategies[schema_name]
        
        # Check what any strategy needs
        needs_float = any(
            s.get("needs_float_embeddings", False) 
            for s in schema_strategies.values()
        )
        needs_binary = any(
            s.get("needs_binary_embeddings", False) 
            for s in schema_strategies.values()
        )
        
        return {
            "needs_float": needs_float,
            "needs_binary": needs_binary
        }
    
    def get_embedding_field_names(self, schema_name: str) -> Dict[str, str]:
        """Get the actual embedding field names used by the schema"""
        
        if schema_name not in self.ranking_strategies:
            raise ValueError(f"No ranking strategies found for schema '{schema_name}'")
        
        schema_strategies = self.ranking_strategies[schema_name]
        
        # Collect unique field names from all strategies
        float_fields = set()
        binary_fields = set()
        
        for strategy in schema_strategies.values():
            if strategy.get("needs_float_embeddings", False):
                field = strategy.get("embedding_field")
                if field and "binary" not in field:
                    float_fields.add(field)
            
            if strategy.get("needs_binary_embeddings", False):
                field = strategy.get("embedding_field")
                if field and "binary" in field:
                    binary_fields.add(field)
        
        # Use the most common or first field name found
        result = {}
        if float_fields:
            result["float_field"] = next(iter(float_fields))
        if binary_fields:
            result["binary_field"] = next(iter(binary_fields))
        
        return result
    
    def get_tensor_info(self, schema_name: str) -> Dict[str, Any]:
        """Extract tensor dimension information from strategies"""
        
        schema_strategies = self.ranking_strategies[schema_name]
        tensor_info = {}
        
        # Look for tensor definitions in inputs
        for strategy in schema_strategies.values():
            if "query_tensor_name" in strategy:
                tensor_name = strategy["query_tensor_name"]
                
                # Find the tensor type from any strategy that uses it
                for s in schema_strategies.values():
                    for input_name, input_type in s.get("inputs", {}).items():
                        if tensor_name in input_name:
                            tensor_info[tensor_name] = self._parse_tensor_type(input_type)
        
        return tensor_info
    
    def _parse_tensor_type(self, tensor_type: str) -> Dict[str, Any]:
        """Parse tensor type string to extract dimensions"""
        # Example: "tensor<float>(querytoken{}, v[128])" → {"type": "float", "dims": {"v": 128}}
        
        import re
        
        # Extract type (float, int8, etc.)
        type_match = re.search(r'tensor<(\w+)>', tensor_type)
        tensor_element_type = type_match.group(1) if type_match else "float"
        
        # Extract dimensions
        dims = {}
        
        # Indexed dimensions like v[128]
        indexed_dims = re.findall(r'(\w+)\[(\d+)\]', tensor_type)
        for dim_name, dim_size in indexed_dims:
            dims[dim_name] = int(dim_size)
        
        # Mapped dimensions like querytoken{}
        mapped_dims = re.findall(r'(\w+)\{\}', tensor_type)
        for dim_name in mapped_dims:
            dims[dim_name] = -1  # -1 indicates variable size
        
        return {
            "type": tensor_element_type,
            "dimensions": dims
        }
    
    def format_document(
        self, 
        schema_name: str,
        video_id: str,
        embeddings: Optional[np.ndarray],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format document based on ranking strategy requirements"""
        
        # Get what embeddings are needed
        requirements = self.get_required_embeddings(schema_name)
        
        # Start with base document
        document = {
            "video_id": video_id,
            **metadata  # frame_id, timestamps, etc.
        }
        
        # Add embeddings based on requirements
        if embeddings is not None:
            if requirements["needs_float"]:
                # Check if patch-based (2D) or global (1D)
                if embeddings.ndim == 2:
                    # Patch-based: format as cells
                    document["embedding"] = self._format_patch_embeddings(embeddings)
                else:
                    # Global: format as list
                    document["embedding"] = embeddings.tolist()
            
            if requirements["needs_binary"]:
                # Generate binary embeddings
                binary_embeddings = self._generate_binary_embeddings(embeddings)
                
                if binary_embeddings.ndim == 2:
                    document["embedding_binary"] = self._format_patch_embeddings(binary_embeddings)
                else:
                    document["embedding_binary"] = binary_embeddings.tolist()
        
        return document
    
    def _generate_binary_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Generate binary embeddings from float embeddings"""
        # Binarize (>0 becomes 1, <=0 becomes 0)
        binary = np.where(embeddings > 0, 1, 0).astype(np.uint8)
        
        # Pack bits
        if embeddings.ndim == 1:
            # 1D: pad and pack
            padding = (8 - len(binary) % 8) % 8
            if padding:
                binary = np.pad(binary, (0, padding), mode='constant')
            return np.packbits(binary).astype(np.int8)
        else:
            # 2D: pack each row
            return np.packbits(binary, axis=1).astype(np.int8)
    
    def _format_patch_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Format patch-based embeddings as cells"""
        cells = []
        
        for patch_idx in range(embeddings.shape[0]):
            for v_idx in range(embeddings.shape[1]):
                cells.append({
                    "address": {
                        "patch": str(patch_idx),
                        "v": str(v_idx)
                    },
                    "value": float(embeddings[patch_idx, v_idx])
                })
        
        return {"cells": cells}


# Example usage in processing pipeline
def process_with_strategy_awareness(video_path: str, profile: str, schema_loader):
    """Example of how to use strategy-aware processing"""

    # Initialize processor with injected schema_loader
    processor = StrategyAwareProcessor(schema_loader)
    
    # Schema name is the same as profile name
    schema_name = profile
    
    # Check what's needed
    requirements = processor.get_required_embeddings(schema_name)
    print(f"Schema {schema_name} requires:")
    print(f"  Float embeddings: {requirements['needs_float']}")
    print(f"  Binary embeddings: {requirements['needs_binary']}")
    
    # Get tensor information
    tensor_info = processor.get_tensor_info(schema_name)
    print(f"  Tensor dimensions: {tensor_info}")
    
    # Process video and generate embeddings
    # ... (video processing code)
    embeddings = np.random.randn(16, 128)  # Example
    
    # Format document based on strategy requirements
    document = processor.format_document(
        schema_name=schema_name,
        video_id="test_video",
        embeddings=embeddings,
        metadata={
            "frame_id": 1,
            "start_time": 0.0,
            "end_time": 1.0
        }
    )
    
    print(f"Generated document fields: {list(document.keys())}")
    
    return document
