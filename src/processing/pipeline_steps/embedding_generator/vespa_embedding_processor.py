#!/usr/bin/env python3
"""
Vespa Embedding Processor - Handles Vespa-specific format conversion
"""

import torch
import numpy as np
import struct
from binascii import hexlify
from typing import Dict, Any, List, Optional
import logging


class VespaEmbeddingProcessor:
    """Processes embeddings for Vespa's specific format requirements"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, model_name: str = None, schema_name: str = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or ""
        self.schema_name = schema_name or ""
    
    def process_embeddings(self, raw_embeddings: Any) -> Dict[str, Any]:
        """
        Process raw embeddings into Vespa format
        
        Args:
            raw_embeddings: Can be numpy array, dict of arrays, or already processed
            
        Returns:
            Dict with processed embeddings ready for Vespa
        """
        if isinstance(raw_embeddings, np.ndarray):
            # Check if it's a valid 2D array
            if raw_embeddings.ndim == 2 and raw_embeddings.shape[0] > 0:
                # Single embedding array - convert to float/binary
                # All schemas now use consistent field names
                return {
                    "embedding": self._convert_to_float_dict(raw_embeddings),
                    "embedding_binary": self._convert_to_binary_dict(raw_embeddings)
                }
            else:
                # For 1D arrays (global embeddings), we still need to convert to patch format
                # Treat the entire embedding as a single patch
                if raw_embeddings.ndim == 1:
                    # Reshape to (1, embedding_dim) to treat as single patch
                    raw_embeddings = raw_embeddings.reshape(1, -1)
                    
                    # All schemas now use consistent field names
                    return {
                        "embedding": self._convert_to_float_dict(raw_embeddings),
                        "embedding_binary": self._convert_to_binary_dict(raw_embeddings)
                    }
                else:
                    # Empty arrays - return as-is
                    return raw_embeddings.tolist() if hasattr(raw_embeddings, 'tolist') else raw_embeddings
        elif isinstance(raw_embeddings, dict):
            # Multiple embeddings - process each
            processed = {}
            for key, value in raw_embeddings.items():
                if isinstance(value, np.ndarray):
                    if "binary" in key:
                        processed[key] = self._convert_to_binary_dict(value)
                    else:
                        processed[key] = self._convert_to_float_dict(value)
                else:
                    # Already processed
                    processed[key] = value
            return processed
        else:
            # Unknown format - pass through
            return raw_embeddings
    
    def _convert_to_float_dict(self, embeddings: np.ndarray) -> Any:
        """Convert numpy array to Vespa float format
        
        For global schemas using tensor<float>, return raw float values as a list.
        For patch-based schemas using tensor<bfloat16>, return hex-encoded bfloat16.
        """
        # Check if this is a global schema that uses tensor<float>
        if "global" in self.schema_name.lower() and embeddings.shape[0] == 1:
            # Global schemas use tensor<float>(v[dim]) which expects raw float values
            return embeddings[0].tolist()
        
        # For patch-based schemas, use hex-encoded bfloat16 format
        embedding_dict = {}
        for patch_idx in range(len(embeddings)):
            hex_string = self._numpy_to_hex_bfloat16(embeddings[patch_idx])
            embedding_dict[patch_idx] = hex_string
        return embedding_dict
    
    def _convert_to_binary_dict(self, embeddings: np.ndarray) -> Any:
        """Convert numpy array to binary format
        
        For global schemas, return raw int8 values as a list.
        For patch-based schemas, return hex-encoded binary.
        """
        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Binarize: positive values -> 1, negative/zero -> 0
        binarized = np.packbits(
            np.where(embeddings > 0, 1, 0),
            axis=1
        ).astype(np.int8)
        
        # Check if this is a global schema
        if "global" in self.schema_name.lower() and binarized.shape[0] == 1:
            # Global schemas use tensor<int8>(v[dim]) which expects raw int values
            return binarized[0].tolist()
        
        # For patch-based schemas, use hex-encoded format
        embedding_dict = {}
        for idx in range(len(binarized)):
            hex_string = hexlify(binarized[idx].tobytes()).decode('utf-8')
            embedding_dict[idx] = hex_string
        
        return embedding_dict
    
    def _numpy_to_hex_bfloat16(self, array: np.ndarray) -> str:
        """Convert numpy array to hex-encoded bfloat16 format"""
        tensor = torch.tensor(array, dtype=torch.float32)
        
        def float_to_bfloat16_hex(f: float) -> str:
            packed_float = struct.pack("=f", f)
            bfloat16_bits = struct.unpack("=H", packed_float[2:])[0]
            return format(bfloat16_bits, "04X")
        
        hex_list = [float_to_bfloat16_hex(float(val)) for val in tensor.flatten()]
        return "".join(hex_list)
    
    def _numpy_to_hex_float32(self, array: np.ndarray) -> str:
        """Convert numpy array to hex-encoded 32-bit float format for tensor<float>"""
        tensor = torch.tensor(array, dtype=torch.float32)
        
        def float_to_float32_hex(f: float) -> str:
            # Pack as 32-bit float and convert all 4 bytes to hex
            packed_float = struct.pack("=f", f)
            return hexlify(packed_float).decode('utf-8').upper()
        
        hex_list = [float_to_float32_hex(float(val)) for val in tensor.flatten()]
        return "".join(hex_list)