#!/usr/bin/env python3
"""
Vespa Client using official pyvespa library with format conversion.

This module implements the Vespa backend client using the official pyvespa
library. It handles Vespa-specific requirements:

1. Hex encoding for bfloat16 tensors
2. Binary encoding for int8 tensors  
3. Multi-vector document structure
4. Namespace and schema management

The client uses VespaEmbeddingProcessor internally to handle format
conversions, keeping the backend-specific logic encapsulated.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
import struct
from binascii import hexlify
from .backend_client import BackendClient
from .vespa_embedding_processor import VespaEmbeddingProcessor
from src.common.core import Document, MediaType
from src.backends.vespa.strategy_aware_processor import StrategyAwareProcessor
from src.common.utils.retry import retry_with_backoff, RetryConfig


class VespaPyClient(BackendClient):
    """
    Vespa client implementation using official pyvespa library.
    
    This client handles all Vespa-specific requirements including:
    - Connection management with health checks
    - Document format conversion (numpy → hex)
    - Batch feeding with proper error handling
    - Schema and namespace configuration
    
    The client internally uses VespaEmbeddingProcessor for format
    conversion, keeping embedding format details hidden from the
    generator.
    
    Attributes:
        vespa_url: URL of Vespa instance
        vespa_port: Port for Vespa HTTP API
        schema_name: Name of the Vespa schema to use
        app: Vespa application instance from pyvespa
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        # Extract Vespa-specific config
        schema_name = config.get("schema_name")
        if not schema_name:
            raise ValueError("schema_name is required in config for VespaPyClient")
        super().__init__(config, schema_name, logger)
        
        self.vespa_url = config.get("vespa_url", "http://localhost")
        self.vespa_port = config.get("vespa_port", 8080)
        self.app = None
        self._connected = False
        
        # Create Vespa-specific embedding processor
        # Get model name from active profile if available
        model_name = ""
        active_profile = config.get("active_profile")
        if active_profile:
            profiles = config.get("video_processing_profiles", {})
            if active_profile in profiles:
                model_name = profiles[active_profile].get("embedding_model", "")
        
        self._embedding_processor = VespaEmbeddingProcessor(logger, model_name, schema_name)
        
        # Initialize strategy-aware processor to get field names from ranking strategies
        self._strategy_processor = StrategyAwareProcessor()
        
        # Load schema fields to know what fields to populate
        self._load_schema_fields()
    
    def _load_schema_fields(self):
        """Load the fields defined in the schema"""
        import json
        from pathlib import Path
        
        # Load the schema definition to know what fields it has
        schema_path = Path(__file__).parent.parent.parent.parent.parent / "configs" / "schemas" / f"{self.schema_name}_schema.json"
        
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        
        try:
            with open(schema_path, 'r') as f:
                schema_def = json.load(f)
            
            # Extract field names from schema
            self.schema_fields = set()
            for field in schema_def.get("document", {}).get("fields", []):
                self.schema_fields.add(field["name"])
            
            if not self.schema_fields:
                raise ValueError(f"No fields found in schema {self.schema_name}")
                
            self.logger.debug(f"Loaded {len(self.schema_fields)} fields from schema {self.schema_name}")
        except Exception as e:
            self.logger.error(f"Failed to load schema fields from {schema_path}: {e}")
            raise
    
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def connect(self) -> bool:
        """Connect to Vespa using pyvespa with retry logic"""
        try:
            from vespa.application import Vespa
            
            # Create Vespa application instance
            self.app = Vespa(
                url=self.vespa_url,
                port=self.vespa_port
            )
            
            # Test connection
            health = self.app.get_application_status()
            if health:
                self._connected = True
                self.logger.info(f"Connected to Vespa at {self.vespa_url}:{self.vespa_port}")
                return True
            else:
                self.logger.error("Failed to connect to Vespa - health check failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Vespa: {e}")
            raise  # Re-raise for retry logic
    
    def process(self, doc: Document) -> Dict[str, Any]:
        """
        Convert universal Document to Vespa format.
        
        This method handles all Vespa-specific conversions:
        1. Process embeddings using VespaEmbeddingProcessor
        2. Map Document fields to Vespa schema fields
        3. Create Vespa document structure with proper ID format
        
        The embedding processor handles:
        - Float embeddings → hex-encoded bfloat16
        - Binary embeddings → hex-encoded int8
        - Multi-vector formats with patch indices
        
        Args:
            doc: Universal Document with raw numpy embeddings
            
        Returns:
            Dict with Vespa document structure:
            {
                "put": "id:video:schema_name::document_id",
                "fields": {
                    "embedding": {0: "hex", 1: "hex", ...},
                    "embedding_binary": {0: "hex", ...},
                    "video_id": "...",
                    ...
                }
            }
        """
        # Get required embeddings and field names from strategy processor
        required_embeddings = self._strategy_processor.get_required_embeddings(self.schema_name)
        field_names = self._strategy_processor.get_embedding_field_names(self.schema_name)
        
        # Process embeddings using internal processor
        all_processed_embeddings = self._embedding_processor.process_embeddings(
            doc.embeddings.embeddings if doc.embeddings else None
        )
        
        # Map processed embeddings to the correct field names based on ranking strategies
        processed_embeddings = {}
        
        # Float embeddings
        if required_embeddings.get("needs_float", False) and "embedding" in all_processed_embeddings:
            float_field = field_names.get("float_field", "embedding")
            processed_embeddings[float_field] = all_processed_embeddings["embedding"]
            self.logger.debug(f"Adding float embeddings to field '{float_field}'")
        
        # Binary embeddings
        if required_embeddings.get("needs_binary", False) and "embedding_binary" in all_processed_embeddings:
            binary_field = field_names.get("binary_field", "embedding_binary")
            processed_embeddings[binary_field] = all_processed_embeddings["embedding_binary"]
            self.logger.debug(f"Adding binary embeddings to field '{binary_field}'")
        
        # Build base fields for all schemas (all use per-document structure now)
        fields = {
            "creation_timestamp": int(time.time() * 1000),  # milliseconds
            **processed_embeddings
        }
        
        # Only add fields that exist in the schema definition
        # This makes the code completely schema-driven
        
        # Add temporal info if schema has these fields
        if doc.temporal_info:
            if "start_time" in self.schema_fields:
                fields["start_time"] = float(doc.temporal_info.start_time)
            if "end_time" in self.schema_fields:
                fields["end_time"] = float(doc.temporal_info.end_time)
        
        # Add segment info if schema has segment_id
        if doc.segment_info and "segment_id" in self.schema_fields:
            fields["segment_id"] = doc.segment_info.segment_idx
            if "total_segments" in self.schema_fields and hasattr(doc.segment_info, "total_segments"):
                fields["total_segments"] = doc.segment_info.total_segments
        
        # Add transcription if schema has audio_transcript field
        if doc.transcription and "audio_transcript" in self.schema_fields:
            fields["audio_transcript"] = doc.transcription
        
        # Handle description from metadata if schema has segment_description
        if "description" in doc.metadata and "segment_description" in self.schema_fields:
            fields["segment_description"] = doc.metadata["description"]
        
        # Add other metadata fields that directly match schema fields
        for key, value in doc.metadata.items():
            if key in self.schema_fields and key not in fields:
                fields[key] = value
        
        # Create Vespa document
        return {
            "put": f"id:video:{self.schema_name}::{doc.doc_id}",
            "fields": fields
        }
    
    # Removed _feed_prepared - using only batch method
    
    def _convert_embeddings_for_vespa(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw embeddings to Vespa's expected format"""
        
        # Standard embedding field names (all schemas now use these)
        embedding_fields = [
            "embedding",
            "embedding_binary"
        ]
        
        for field_name in embedding_fields:
            if field_name in fields:
                value = fields[field_name]
                
                # Check if it's raw numpy array or already converted
                if isinstance(value, np.ndarray):
                    if "binary" in field_name:
                        fields[field_name] = self._convert_to_binary_dict(value)
                    else:
                        fields[field_name] = self._convert_to_float_dict(value)
                elif isinstance(value, dict) and len(value) > 0:
                    # Check if values are numpy arrays (patches)
                    first_value = next(iter(value.values()))
                    if isinstance(first_value, np.ndarray):
                        if "binary" in field_name:
                            fields[field_name] = {
                                k: self._numpy_to_hex_binary(v) 
                                for k, v in value.items()
                            }
                        else:
                            fields[field_name] = {
                                k: self._numpy_to_hex_bfloat16(v) 
                                for k, v in value.items()
                            }
        
        return fields
    
    def _convert_to_float_dict(self, embeddings: np.ndarray) -> Dict[int, str]:
        """Convert numpy array to Vespa float format (hex-encoded bfloat16)"""
        embedding_dict = {}
        for patch_idx in range(len(embeddings)):
            hex_string = self._numpy_to_hex_bfloat16(embeddings[patch_idx])
            embedding_dict[patch_idx] = hex_string
        return embedding_dict
    
    def _convert_to_binary_dict(self, embeddings: np.ndarray) -> Dict[int, str]:
        """Convert numpy array to binary format"""
        # Binarize: positive values -> 1, negative/zero -> 0
        binarized = np.packbits(
            np.where(embeddings > 0, 1, 0),
            axis=1
        ).astype(np.int8)
        
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
    
    def _numpy_to_hex_binary(self, array: np.ndarray) -> str:
        """Convert numpy array to hex-encoded binary"""
        # Binarize and pack
        binary = np.packbits(np.where(array > 0, 1, 0)).astype(np.int8)
        return hexlify(binary.tobytes()).decode('utf-8')
    
    def _feed_prepared_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, List[str]]:
        """Feed prepared documents in batches using pyvespa's batch feeding"""
        if not self._connected:
            if not self.connect():
                return 0, [d["put"].split("::")[-1] for d in documents]
        
        success_count = 0
        failed_docs = []
        
        try:
            # Convert documents to pyvespa format with embedding conversion
            feed_data = []
            for doc in documents:
                doc_id = doc["put"].split("::")[-1]
                fields = self._convert_embeddings_for_vespa(doc["fields"].copy())
                feed_data.append({
                    "id": doc_id,
                    "fields": fields
                })
            
            # Process in batches
            for i in range(0, len(feed_data), batch_size):
                batch = feed_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(feed_data) + batch_size - 1) // batch_size
                
                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} documents)"
                )
                
                # Feed batch using feed_iterable for better reliability
                def feed_batch_iter():
                    for doc in batch:
                        yield doc
                
                # Track results with callback
                batch_success = 0
                batch_failed = []
                
                def callback(response, doc_id):
                    nonlocal batch_success, batch_failed
                    if response.is_successful():
                        batch_success += 1
                    else:
                        batch_failed.append(doc_id)
                        try:
                            error_msg = response.get_json()
                            self.logger.error(f"Failed to feed {doc_id}: {error_msg}")
                        except:
                            self.logger.error(f"Failed to feed {doc_id}: {response.status_code}")
                
                # Use feed_iterable - pyvespa has built-in retry mechanisms
                self.app.feed_iterable(
                    iter=feed_batch_iter(),
                    schema=self.schema_name,
                    namespace="video",
                    callback=callback
                )
                
                # Update counts
                success_count += batch_success
                failed_docs.extend(batch_failed)
                
                self.logger.info(
                    f"Batch {batch_num}: {batch_success}/{len(batch)} "
                    f"documents fed successfully"
                )
            
        except Exception as e:
            self.logger.error(f"Batch feeding failed: {e}")
            # Mark remaining as failed
            for doc in documents[success_count:]:
                failed_docs.append(doc["put"].split("::")[-1])
        
        return success_count, failed_docs
    
    def check_document_exists(self, doc_id: str) -> bool:
        """Check if document exists using pyvespa"""
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            response = self.app.get_data(
                schema=self.schema_name,
                data_id=doc_id,
                namespace="video"
            )
            return response is not None and response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error checking document existence: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document using pyvespa"""
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            response = self.app.delete_data(
                schema=self.schema_name,
                data_id=doc_id,
                namespace="video"
            )
            return response is not None and response.status_code in [200, 404]
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
    
    def close(self):
        """Close connection"""
        self._connected = False
        self.app = None