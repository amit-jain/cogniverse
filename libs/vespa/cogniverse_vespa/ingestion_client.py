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

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from cogniverse_core.common.document import Document
from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff

from .embedding_processor import VespaEmbeddingProcessor
from .strategy_aware_processor import StrategyAwareProcessor


class VespaPyClient:
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
        """
        Initialize VespaPyClient from config.
        
        Args:
            config: Config dict with schema_name, vespa_url, vespa_port, and optionally model info
            logger: Optional logger
        """
        # Extract from config
        self.schema_name = config.get("schema_name")
        if not self.schema_name:
            raise ValueError("schema_name is required in config")
        # For tenant-scoped schemas, use base_schema_name for loading schema file
        self.base_schema_name = config.get("base_schema_name", self.schema_name)
        self.vespa_url = config.get("vespa_url")
        if not self.vespa_url:
            raise ValueError("vespa_url is required in config")
        self.vespa_port = config.get("vespa_port", 8080)

        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # CRITICAL: Log backend initialization to track schema assignment
        self.logger.warning("🚀 VESPA CLIENT INITIALIZED:")
        self.logger.warning(f"   Schema name: {self.schema_name}")
        self.logger.warning(f"   Base schema name: {self.base_schema_name}")
        self.logger.warning(f"   Instance ID: {id(self)}")
        self.logger.warning(f"   Vespa URL: {self.vespa_url}:{self.vespa_port}")

        self.app = None
        self._connected = False

        # Get model name from profile config
        profile_config = config.get("profile_config", {})
        model_name = profile_config.get("model", "")

        # Create embedding processor with this client's schema
        self._embedding_processor = VespaEmbeddingProcessor(self.logger, model_name, self.schema_name)

        # Initialize strategy-aware processor
        self._strategy_processor = StrategyAwareProcessor()

        # Load schema fields for this specific schema
        self._load_schema_fields()
        
        # Production-ready feed configuration with environment variable overrides
        import os
        self.feed_config = {
            "max_queue_size": int(os.environ.get("VESPA_FEED_MAX_QUEUE_SIZE", 
                                                 config.get("feed_max_queue_size", 500))),
            "max_workers": int(os.environ.get("VESPA_FEED_MAX_WORKERS", 
                                              config.get("feed_max_workers", 4))),
            "max_connections": int(os.environ.get("VESPA_FEED_MAX_CONNECTIONS", 
                                                  config.get("feed_max_connections", 8))),
            "compress": os.environ.get("VESPA_FEED_COMPRESS", 
                                       config.get("feed_compress", "auto"))
        }
        
        self.logger.info(f"Feed configuration: {self.feed_config}")
    
    def _load_schema_fields(self):
        """Load the fields defined in the schema using base schema name"""
        import json
        from pathlib import Path

        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / "configs" / "schemas").exists():
                # Use base_schema_name for loading schema file (tenant schemas use base schema structure)
                schema_path = current / "configs" / "schemas" / f"{self.base_schema_name}_schema.json"
                break
            current = current.parent
        else:
            raise RuntimeError("Cannot find project root with configs/schemas directory")

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
                raise ValueError(f"No fields found in schema {self.base_schema_name}")

            self.logger.debug(f"Loaded {len(self.schema_fields)} fields from base schema {self.base_schema_name}")
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
            schema_name: Schema to use for this document (REQUIRED - no fallbacks)
            
        Returns:
            Dict with Vespa document structure INCLUDING schema:
            {
                "schema": "video_colpali_smol500_mv_frame",  # Explicit schema
                "put": "id:video:schema_name::document_id",
                "fields": {
                    "embedding": {0: "hex", 1: "hex", ...},
                    "embedding_binary": {0: "hex", ...},
                    "video_id": "...",
                    ...
                }
            }
        """
        # Use this client's schema (each client is dedicated to one schema)
        # Get required embeddings and field names from strategy processor
        # Use base_schema_name for strategy lookup (tenant schemas use same strategies as base)
        required_embeddings = self._strategy_processor.get_required_embeddings(self.base_schema_name)
        field_names = self._strategy_processor.get_embedding_field_names(self.base_schema_name)
        
        # Process embeddings using internal processor
        # New Document class stores embeddings directly in embeddings dict
        raw_embeddings = None
        if "embedding" in doc.embeddings:
            raw_embeddings = doc.embeddings["embedding"]["data"]
        
        all_processed_embeddings = self._embedding_processor.process_embeddings(raw_embeddings)
        
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
        
        # Add temporal info from metadata (new Document structure)
        if "start_time" in doc.metadata and "start_time" in self.schema_fields:
            fields["start_time"] = float(doc.metadata["start_time"])
        if "end_time" in doc.metadata and "end_time" in self.schema_fields:
            fields["end_time"] = float(doc.metadata["end_time"])
        
        # Add segment info from metadata (new Document structure)
        if "segment_index" in doc.metadata and "segment_id" in self.schema_fields:
            fields["segment_id"] = doc.metadata["segment_index"]
        if "total_segments" in doc.metadata and "total_segments" in self.schema_fields:
            fields["total_segments"] = doc.metadata["total_segments"]
        
        # Add transcription from metadata (new Document structure)
        if "audio_transcript" in doc.metadata and "audio_transcript" in self.schema_fields:
            fields["audio_transcript"] = doc.metadata["audio_transcript"]
        
        # Handle description from metadata if schema has segment_description
        if "description" in doc.metadata and "segment_description" in self.schema_fields:
            fields["segment_description"] = doc.metadata["description"]
        
        # Add other metadata fields that directly match schema fields
        for key, value in doc.metadata.items():
            if key in self.schema_fields and key not in fields:
                fields[key] = value
        
        # Create Vespa document with this client's schema  
        doc_id_string = f"id:video:{self.schema_name}::{doc.id}"
        
        # CRITICAL: Log schema being used for each document (first doc only to avoid spam)
        if doc.id.endswith("_0_0") or doc.id.endswith("_0"):  # Log first doc of each video
            self.logger.info("📝 FEEDING DOCUMENT TO SCHEMA:")
            self.logger.info(f"   Doc ID: {doc_id_string}")
            self.logger.info(f"   Schema: {self.schema_name}")
            # Log embedding field type to verify dimensions
            if 'embedding' in fields:
                if isinstance(fields['embedding'], dict) and 'values' in fields['embedding']:
                    emb_len = len(fields['embedding']['values'])
                    self.logger.info(f"   Embedding dimensions: {emb_len}")
                else:
                    self.logger.info(f"   Embedding type: {type(fields['embedding'])}")
        
        return {
            "put": doc_id_string,
            "fields": fields
        }
    
    # Removed _feed_prepared - using only batch method
    
    def _convert_embeddings_for_vespa(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Verify embeddings are already in correct format (processed by VespaEmbeddingProcessor)"""
        # Embeddings should already be converted by VespaEmbeddingProcessor in process()
        # This method now just passes through the already-converted fields
        return fields
    
    # Removed duplicate conversion methods - VespaEmbeddingProcessor handles all conversions
    
    def _feed_prepared_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, List[str]]:
        """Feed prepared documents in batches using pyvespa's batch feeding
        
        Production-ready configuration with proper retry and timeout handling.
        
        Args:
            documents: List of prepared documents from process()
            batch_size: Number of documents per batch
            
        Returns:
            Tuple of (success_count, list of failed doc IDs)
        """
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
                
                # Debug: log field names being sent
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Document {doc_id} fields: {list(fields.keys())}")
                
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
                    f"Processing batch {batch_num}/{total_batches} for schema '{self.schema_name}' "
                    f"({len(batch)} documents)"
                )
                
                # Feed batch using feed_iterable for better reliability
                def feed_batch_iter():
                    for doc in batch:
                        yield doc
                
                # Track results with callback
                batch_success = 0
                batch_failed = []
                batch_retries = {}  # Track retries per document
                
                def callback(response, doc_id):
                    nonlocal batch_success, batch_failed, batch_retries
                    if response.is_successful():
                        batch_success += 1
                        if doc_id in batch_retries:
                            self.logger.info(f"Document {doc_id} succeeded after {batch_retries[doc_id]} retries")
                    else:
                        # Track retry attempts
                        if doc_id not in batch_retries:
                            batch_retries[doc_id] = 0
                        batch_retries[doc_id] += 1
                        
                        # Log detailed error
                        try:
                            error_msg = response.get_json()
                            status = response.get_status_code()
                            self.logger.error(
                                f"Failed to feed {doc_id} to schema '{self.schema_name}' "
                                f"(attempt {batch_retries[doc_id]}): HTTP {status} - {error_msg}"
                            )
                        except Exception:
                            status = getattr(response, 'status_code', 'unknown')
                            self.logger.error(
                                f"Failed to feed {doc_id} to schema '{self.schema_name}' "
                                f"(attempt {batch_retries[doc_id]}): HTTP {status}"
                            )
                        
                        # Add to failed list after max retries (handled by pyvespa internally)
                        batch_failed.append(doc_id)
                
                # Use feed_iterable with production-ready configuration
                # These parameters provide robust feeding with proper resource management
                self.app.feed_iterable(
                    iter=feed_batch_iter(),
                    schema=self.schema_name,  # Use this client's schema
                    namespace="video",
                    callback=callback,
                    # Production configuration parameters from self.feed_config
                    max_queue_size=self.feed_config["max_queue_size"],
                    max_workers=self.feed_config["max_workers"],
                    max_connections=self.feed_config["max_connections"],
                    compress=self.feed_config["compress"]
                )
                
                # Update counts
                success_count += batch_success
                failed_docs.extend(batch_failed)
                
                # Log batch results with retry info
                unique_failed = list(set(batch_failed))  # Remove duplicates from retries
                self.logger.info(
                    f"Batch {batch_num} to schema '{self.schema_name}': "
                    f"{batch_success}/{len(batch)} documents fed successfully"
                )
                if unique_failed:
                    self.logger.warning(
                        f"Batch {batch_num} had {len(unique_failed)} failed documents "
                        f"(some may have been retried)"
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
