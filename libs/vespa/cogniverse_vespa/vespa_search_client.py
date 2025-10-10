"""
Vespa Search Client for Video Frame Search
Implements search functionality using the document-per-frame schema
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from vespa.application import Vespa


class RankingStrategy(Enum):
    """
    Available ranking strategies for video search with usage recommendations.
    Each strategy has different speed/accuracy tradeoffs and use cases.
    """
    
    # Text-only search (fastest for pure text queries)
    BM25_ONLY = "bm25_only"
    
    # Visual search strategies
    FLOAT_FLOAT = "float_float"  # Highest accuracy, slower
    BINARY_BINARY = "binary_binary"  # Fastest visual search
    FLOAT_BINARY = "float_binary"  # Good balance of speed/accuracy
    PHASED = "phased"  # Optimized retrieval with reranking
    
    # Hybrid strategies (visual + text)
    HYBRID_FLOAT_BM25 = "hybrid_float_bm25"  # Best overall accuracy
    HYBRID_BINARY_BM25 = "hybrid_binary_bm25"  # Fast hybrid search
    HYBRID_BM25_BINARY = "hybrid_bm25_binary"  # Text-first with visual rerank
    HYBRID_BM25_FLOAT = "hybrid_bm25_float"  # Text-first with precise rerank
    
    # No-description variants (exclude frame descriptions from BM25)
    HYBRID_FLOAT_BM25_NO_DESC = "hybrid_float_bm25_no_description"
    HYBRID_BINARY_BM25_NO_DESC = "hybrid_binary_bm25_no_description"  
    HYBRID_BM25_BINARY_NO_DESC = "hybrid_bm25_binary_no_description"
    HYBRID_BM25_FLOAT_NO_DESC = "hybrid_bm25_float_no_description"
    
    # Additional text-only strategy
    BM25_NO_DESCRIPTION = "bm25_no_description"
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, Dict[str, str]]:
        """Get detailed information about each ranking strategy for agent decision-making."""
        return {
            cls.BM25_ONLY.value: {
                "description": "Pure text search using BM25",
                "speed": "Fastest",
                "accuracy": "Good for text-only queries",
                "use_case": "When query is purely textual with no visual component",
                "requires_embeddings": "No"
            },
            cls.FLOAT_FLOAT.value: {
                "description": "Direct ColPali with float embeddings", 
                "speed": "Slowest",
                "accuracy": "Highest visual accuracy",
                "use_case": "When maximum visual precision is needed",
                "requires_embeddings": "Yes"
            },
            cls.BINARY_BINARY.value: {
                "description": "Hamming distance on binary embeddings",
                "speed": "Fastest visual",
                "accuracy": "Good approximation",
                "use_case": "When speed is critical for visual search",
                "requires_embeddings": "Yes (binary)"
            },
            cls.FLOAT_BINARY.value: {
                "description": "Float query with binary storage using unpack_bits",
                "speed": "Fast",
                "accuracy": "Very good",
                "use_case": "Balanced speed/accuracy for visual search",
                "requires_embeddings": "Yes"
            },
            cls.PHASED.value: {
                "description": "Hamming first phase, float reranking",
                "speed": "Fast",
                "accuracy": "High (combines speed + precision)",
                "use_case": "Optimized retrieval with precise reranking",
                "requires_embeddings": "Yes (both binary and float)"
            },
            cls.HYBRID_FLOAT_BM25.value: {
                "description": "Float ColPali + BM25 text reranking",
                "speed": "Slow",
                "accuracy": "Highest overall",
                "use_case": "Complex queries with both visual and text components",
                "requires_embeddings": "Yes"
            },
            cls.HYBRID_BINARY_BM25.value: {
                "description": "Binary ColPali + BM25 combined scoring",
                "speed": "Fast",
                "accuracy": "Good hybrid",
                "use_case": "Fast hybrid search for visual+text queries",
                "requires_embeddings": "Yes (binary)"
            },
            cls.HYBRID_BM25_BINARY.value: {
                "description": "BM25 first, binary visual reranking",
                "speed": "Fast",
                "accuracy": "Good",
                "use_case": "Text-heavy queries with visual validation",
                "requires_embeddings": "Yes (binary)"
            },
            cls.HYBRID_BM25_FLOAT.value: {
                "description": "BM25 first, float visual reranking",
                "speed": "Medium",
                "accuracy": "Very good",
                "use_case": "Text-heavy queries with precise visual reranking",
                "requires_embeddings": "Yes"
            },
            cls.BM25_NO_DESCRIPTION.value: {
                "description": "Pure BM25 text search (title + transcript only)",
                "speed": "Fastest",
                "accuracy": "Good for text-only queries",
                "use_case": "When frame descriptions are unreliable or unavailable",
                "requires_embeddings": "No"
            },
            cls.HYBRID_FLOAT_BM25_NO_DESC.value: {
                "description": "Float ColPali + BM25 text (no descriptions)",
                "speed": "Slow",
                "accuracy": "High overall",
                "use_case": "Complex queries avoiding noisy frame descriptions",
                "requires_embeddings": "Yes"
            },
            cls.HYBRID_BINARY_BM25_NO_DESC.value: {
                "description": "Binary ColPali + BM25 (no descriptions)",
                "speed": "Fast", 
                "accuracy": "Good hybrid",
                "use_case": "Fast hybrid search avoiding frame descriptions",
                "requires_embeddings": "Yes (binary)"
            },
            cls.HYBRID_BM25_BINARY_NO_DESC.value: {
                "description": "BM25 first, binary visual rerank (no descriptions)",
                "speed": "Fast",
                "accuracy": "Good",
                "use_case": "Text-heavy queries with fast visual validation, no descriptions",
                "requires_embeddings": "Yes (binary)"
            },
            cls.HYBRID_BM25_FLOAT_NO_DESC.value: {
                "description": "BM25 first, float visual rerank (no descriptions)",
                "speed": "Medium",
                "accuracy": "Very good",
                "use_case": "Text-heavy queries with precise visual reranking, no descriptions",
                "requires_embeddings": "Yes"
            }
        }
    
    @classmethod
    def recommend_strategy(cls, 
                          has_visual_component: bool,
                          has_text_component: bool, 
                          speed_priority: bool = False) -> 'RankingStrategy':
        """Recommend a ranking strategy based on query characteristics."""
        
        if not has_visual_component and has_text_component:
            return cls.BM25_ONLY
        
        if has_visual_component and not has_text_component:
            return cls.BINARY_BINARY if speed_priority else cls.FLOAT_FLOAT
        
        if has_visual_component and has_text_component:
            return cls.HYBRID_BINARY_BM25 if speed_priority else cls.HYBRID_FLOAT_BM25
        
        # Default fallback
        return cls.BM25_ONLY


class VespaVideoSearchClient:
    """
    Client for searching video frames using Vespa with document-per-frame schema.
    Supports all 9 ranking strategies with agent-friendly JSON query interface.
    
    Usage:
        client = VespaVideoSearchClient()
        results = client.search({
            "query": "person walking",
            "ranking": "hybrid_float_bm25", 
            "top_k": 10,
            "start_date": "2024-01-01"
        })
    """
    
    def __init__(self, vespa_url: str = "http://localhost", vespa_port: int = 8080):
        """
        Initialize Vespa search client
        
        Args:
            vespa_url: Vespa server URL
            vespa_port: Vespa server port
        """
        # Load configuration
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from cogniverse_core.config.utils import get_config
        self.config = get_config()
        # Get schema from environment or config (required)
        self.vespa_schema = os.environ.get("VESPA_SCHEMA") or self.config.get("schema_name")
        if not self.vespa_schema:
            raise ValueError("No schema_name found in VESPA_SCHEMA env var or config")
        
        self.logger = logging.getLogger(__name__)
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        
        # Initialize query encoder based on schema
        self.query_encoder = None
        self._init_query_encoder()
        
        try:
            self.vespa_app = Vespa(url=f"{vespa_url}:{vespa_port}")
            self.logger.info(f"Connected to Vespa at {vespa_url}:{vespa_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Vespa: {e}")
            self.vespa_app = None
    
    def _init_query_encoder(self):
        """Initialize query encoder based on the Vespa schema"""
        try:
            from cogniverse_agents.query.encoders import QueryEncoderFactory
            
            profile = self.vespa_schema
            
            if profile:
                # Get model name from config
                profiles = self.config.get("video_processing_profiles", {})
                if profile in profiles:
                    model_name = profiles[profile].get("embedding_model")
                else:
                    # Default models
                    default_models = {
                        "frame_based_colpali": "vidore/colsmol-500m",
                        "colqwen_chunks": "vidore/colqwen-omni-v0.1",
                        "direct_video_frame": "videoprism_public_v1_base_hf",
                        "direct_video_frame_large": "videoprism_public_v1_large_hf"
                    }
                    model_name = default_models.get(profile)
                
                self.query_encoder = QueryEncoderFactory.create_encoder(profile, model_name)
                self.logger.info(f"Initialized query encoder for {self.vespa_schema} using {profile}")
            else:
                self.logger.warning(f"No query encoder mapping for schema: {self.vespa_schema}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize query encoder: {e}")
            self.query_encoder = None
    
    def health_check(self) -> bool:
        """Check if Vespa is healthy and responsive"""
        if not self.vespa_app:
            return False
        
        try:
            # Simple query to test connectivity
            self.vespa_app.query(
                yql=f"select * from {self.vespa_schema} where true limit 1",
                hits=1
            )
            return True
        except Exception as e:
            self.logger.error(f"Vespa health check failed: {e}")
            return False
    
    def search(self, 
               query_params: Union[Dict[str, Any], str],
               embeddings: Optional[np.ndarray] = None,
               schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Unified search method supporting all 9 ranking strategies with JSON query format.
        
        Args:
            query_params: Either a query string or dict with:
                - query (str): Search query text
                - ranking (str): One of RankingStrategy values (default: "bm25_only")
                - top_k (int): Number of results (default: 10)
                - start_date (str): Filter start date YYYY-MM-DD (optional)
                - end_date (str): Filter end date YYYY-MM-DD (optional)
            embeddings: Pre-computed embeddings for visual strategies (optional)
            schema: Override the default schema name (optional)
            
        Returns:
            List of search results with frame information
        """
        if not self.vespa_app:
            raise RuntimeError("Vespa connection not available")
        
        # Normalize input to dict format
        if isinstance(query_params, str):
            params = {"query": query_params, "ranking": "bm25_only"}
        else:
            params = query_params.copy()
        
        # Set defaults
        query_text = params.get("query", "")
        ranking = params.get("ranking", "bm25_only")
        top_k = params.get("top_k", 10)
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        
        # Extract schema from params if provided, otherwise use the method parameter
        if "schema" in params:
            schema = params.get("schema")
        
        # Validate ranking strategy
        try:
            strategy = RankingStrategy(ranking)
        except ValueError:
            available = [s.value for s in RankingStrategy]
            raise ValueError(f"Invalid ranking strategy '{ranking}'. Available: {available}")
        
        # Generate embeddings if needed and not provided
        if embeddings is None:
            strategy_info = RankingStrategy.get_strategy_info()[ranking]
            if strategy_info["requires_embeddings"] != "No":
                if self.query_encoder and query_text:
                    try:
                        embeddings = self.query_encoder.encode(query_text)
                        self.logger.debug(f"Generated query embeddings: shape={embeddings.shape}")
                    except Exception as e:
                        self.logger.error(f"Failed to generate query embeddings: {e}")
                        # For strategies that require embeddings, this is fatal
                        raise RuntimeError(f"Query encoder failed for {self.vespa_schema}: {e}")
                else:
                    if not query_text:
                        raise ValueError(f"Strategy '{ranking}' requires a text query")
                    else:
                        raise RuntimeError(f"No query encoder available for schema {self.vespa_schema}")
        
        # Validate strategy inputs
        validation_errors = self.validate_strategy_inputs(ranking, query_text, embeddings)
        if validation_errors:
            raise ValueError(f"Strategy validation failed: {'; '.join(validation_errors)}")
        
        # Build search request based on strategy
        return self._execute_strategy_search(
            strategy=strategy,
            query_text=query_text,
            embeddings=embeddings,
            top_k=top_k,
            start_date=start_date,
            end_date=end_date,
            schema=schema
        )
    
    
    def _execute_strategy_search(self,
                                strategy: RankingStrategy,
                                query_text: str,
                                embeddings: Optional[np.ndarray],
                                top_k: int,
                                start_date: Optional[str],
                                end_date: Optional[str],
                                schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute search using the specified ranking strategy."""
        
        # Log search parameters
        self.logger.info(f"Executing {strategy.value} search:")
        self.logger.info(f"  Query: '{query_text}'")
        self.logger.info(f"  Top-K: {top_k}")
        if embeddings is not None:
            self.logger.info(f"  Embeddings shape: {embeddings.shape}")
        
        # Use provided schema or fall back to default
        search_schema = schema or self.vespa_schema
        self.logger.info(f"  Using schema: {search_schema}")
        
        # Build base YQL query
        yql = self._build_base_yql(strategy, start_date, end_date, query_text, schema=search_schema)
        
        # Build request body based on strategy
        body = {
            "yql": yql,
            "hits": top_k,
            "model.restrict": search_schema  # Restrict to specific schema to avoid tensor conflicts
        }
        
        # Add strategy-specific parameters
        if strategy in [RankingStrategy.BM25_ONLY, RankingStrategy.BM25_NO_DESCRIPTION]:
            body.update({
                "query": query_text,
                "ranking": strategy.value,
                "model.defaultIndex": "default"  # Use fieldset for BM25 fields
            })
        
        elif strategy in [RankingStrategy.FLOAT_FLOAT, RankingStrategy.HYBRID_FLOAT_BM25, RankingStrategy.HYBRID_BM25_FLOAT,
                         RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC, RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
            if embeddings is None:
                raise ValueError(f"Strategy '{strategy.value}' requires embeddings")
            
            # Check if this is a global embedding schema (single vector)
            is_global_schema = "global" in search_schema.lower()
            
            if is_global_schema and embeddings.ndim == 1:
                # Global embedding - single vector
                body.update({
                    "ranking": strategy.value,
                    "input.query(qt)": embeddings.tolist()
                })
            else:
                # Multi-token embeddings for patch-based models
                float_embedding = {index: vector.tolist() for index, vector in enumerate(embeddings)}
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qt)": float_embedding
                })
            
            # Add text query for hybrid strategies
            if strategy in [RankingStrategy.HYBRID_FLOAT_BM25, RankingStrategy.HYBRID_BM25_FLOAT,
                          RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC, RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
                body["query"] = query_text
                body["model.defaultIndex"] = "default"  # Use fieldset for BM25 component
        
        elif strategy in [RankingStrategy.BINARY_BINARY, RankingStrategy.HYBRID_BINARY_BM25, RankingStrategy.HYBRID_BM25_BINARY,
                         RankingStrategy.HYBRID_BINARY_BM25_NO_DESC, RankingStrategy.HYBRID_BM25_BINARY_NO_DESC]:
            if embeddings is None:
                raise ValueError(f"Strategy '{strategy.value}' requires embeddings")
            
            # Check if this is a global embedding schema (single vector)
            is_global_schema = "global" in search_schema.lower()
            
            if is_global_schema and embeddings.ndim == 1:
                # Global embedding - single vector, convert to binary hex string
                from binascii import hexlify
                
                binary_vector = np.packbits(np.where(embeddings > 0, 1, 0), axis=0).astype(np.int8)
                binary_hex = str(hexlify(binary_vector), "utf-8")
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qtb)": binary_hex
                })
            else:
                # Multi-token embeddings for patch-based models
                binary_embedding = {
                    index: np.packbits(np.where(vector > 0, 1, 0), axis=0).astype(np.int8).tolist()
                    for index, vector in enumerate(embeddings)
                }
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qtb)": binary_embedding
                })
            
            # Add text query for hybrid strategies
            if strategy in [RankingStrategy.HYBRID_BINARY_BM25, RankingStrategy.HYBRID_BM25_BINARY,
                          RankingStrategy.HYBRID_BINARY_BM25_NO_DESC, RankingStrategy.HYBRID_BM25_BINARY_NO_DESC]:
                body["query"] = query_text
                body["model.defaultIndex"] = "default"  # Use fieldset for BM25 component
        
        elif strategy == RankingStrategy.FLOAT_BINARY:
            if embeddings is None:
                raise ValueError(f"Strategy '{strategy.value}' requires embeddings")
            
            # Check if this is a global embedding schema (single vector)
            is_global_schema = "global" in search_schema.lower()
            
            if is_global_schema and embeddings.ndim == 1:
                # Global embedding - float_binary needs both qtb for search and qt for reranking
                # Based on ingestion, we store binary as list of int8, so let's use list format
                binary_vector = np.packbits(np.where(embeddings > 0, 1, 0), axis=0).astype(np.int8)
                
                # Try using list format for qtb (matching ingestion format)
                body.update({
                    "ranking": strategy.value,
                    "input.query(qtb)": binary_vector.tolist(),  # List of int8 values
                    "input.query(qt)": embeddings.tolist()  # Float values for reranking
                })
            else:
                # Use Vespa ColPali format: input.query(qt) with simple dict
                float_embedding = {index: vector.tolist() for index, vector in enumerate(embeddings)}
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qt)": float_embedding
                })
        
        elif strategy == RankingStrategy.PHASED:
            if embeddings is None:
                raise ValueError(f"Strategy '{strategy.value}' requires embeddings")
            
            # Check if this is a global embedding schema (single vector)
            is_global_schema = "global" in search_schema.lower()
            
            if is_global_schema and embeddings.ndim == 1:
                # Global embedding - single vector for both float and binary
                from binascii import hexlify
                binary_vector = np.packbits(np.where(embeddings > 0, 1, 0), axis=0).astype(np.int8)
                binary_hex = str(hexlify(binary_vector), "utf-8")
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qtb)": binary_hex,
                    "input.query(qt)": embeddings.tolist()
                })
            else:
                # PHASED strategy needs both float and binary embeddings
                float_embedding = {index: vector.tolist() for index, vector in enumerate(embeddings)}
                binary_embedding = {
                    index: np.packbits(np.where(vector > 0, 1, 0), axis=0).astype(np.int8).tolist()
                    for index, vector in enumerate(embeddings)
                }
                
                body.update({
                    "ranking": strategy.value,
                    "input.query(qtb)": binary_embedding,
                    "input.query(qt)": float_embedding
                })
        
        return self._execute_search(body, f"search_{strategy.value}")
    
    def _build_base_yql(self, strategy: RankingStrategy, start_date: Optional[str], end_date: Optional[str], query_text: str = "", schema: Optional[str] = None) -> str:
        """Build base YQL query for the strategy."""
        
        # Use the provided schema or default
        target_schema = schema or self.vespa_schema
        
        # Schema-specific field names
        if "colqwen" in target_schema.lower():
            # ColQwen uses segment_id instead of frame_id
            id_field = "segment_id"
            desc_field = ""  # ColQwen doesn't have frame_description
            transcript_field = ""  # ColQwen doesn't have audio_transcript
        elif "videoprism" in target_schema.lower():
            # VideoPrism uses frame_id but no description/transcript
            id_field = "frame_id"
            desc_field = ""  # VideoPrism doesn't have frame_description
            transcript_field = ""  # VideoPrism doesn't have audio_transcript
        else:
            # Default (ColPali) has all fields
            id_field = "frame_id"
            desc_field = "frame_description"
            transcript_field = "audio_transcript"
        
        # Build field list
        fields = ["video_id", "video_title", id_field, "start_time", "end_time"]
        
        # Add optional fields based on schema and strategy
        if desc_field and strategy not in [RankingStrategy.BM25_NO_DESCRIPTION,
                                          RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC,
                                          RankingStrategy.HYBRID_BINARY_BM25_NO_DESC,
                                          RankingStrategy.HYBRID_BM25_BINARY_NO_DESC,
                                          RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
            fields.append(desc_field)
            
        if transcript_field and strategy in [RankingStrategy.BM25_ONLY, RankingStrategy.BM25_NO_DESCRIPTION,
                                            RankingStrategy.HYBRID_FLOAT_BM25, RankingStrategy.HYBRID_BINARY_BM25, 
                                            RankingStrategy.HYBRID_BM25_BINARY, RankingStrategy.HYBRID_BM25_FLOAT,
                                            RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC, RankingStrategy.HYBRID_BINARY_BM25_NO_DESC,
                                            RankingStrategy.HYBRID_BM25_BINARY_NO_DESC, RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
            fields.append(transcript_field)
        
        yql = f"select {', '.join(fields)} from {target_schema}"
        
        # Add WHERE clause
        # For visual strategies, use nearestNeighbor if this is a global embedding schema
        is_global_schema = "global" in target_schema.lower()
        
        if strategy == RankingStrategy.FLOAT_FLOAT:
            if is_global_schema:
                # Use nearestNeighbor for global embeddings with float field
                yql += " where ({targetHits:100}nearestNeighbor(embedding, qt))"
            else:
                yql += " where true"
        elif strategy == RankingStrategy.FLOAT_BINARY:
            if is_global_schema:
                # float_binary searches on binary field but reranks with float
                yql += " where ({targetHits:100}nearestNeighbor(embedding_binary, qtb))"
            else:
                yql += " where true"
        elif strategy == RankingStrategy.PHASED:
            if is_global_schema:
                # Phased uses binary for first phase
                yql += " where ({targetHits:100}nearestNeighbor(embedding_binary, qtb))"
            else:
                yql += " where true"
        elif strategy in [RankingStrategy.BINARY_BINARY]:
            if is_global_schema:
                # Use nearestNeighbor for global embeddings
                yql += " where ({targetHits:100}nearestNeighbor(embedding_binary, qtb))"
            else:
                yql += " where true"
        elif strategy in [RankingStrategy.HYBRID_FLOAT_BM25, RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC]:
            if is_global_schema:
                # Hybrid with float embeddings - use nearestNeighbor
                yql += " where ({targetHits:100}nearestNeighbor(embedding, qt))"
            else:
                yql += " where userQuery()"
        elif strategy in [RankingStrategy.HYBRID_BINARY_BM25, RankingStrategy.HYBRID_BINARY_BM25_NO_DESC]:
            if is_global_schema:
                # Hybrid with binary embeddings - use nearestNeighbor
                yql += " where ({targetHits:100}nearestNeighbor(embedding_binary, qtb))"
            else:
                yql += " where userQuery()"
        elif strategy in [RankingStrategy.BM25_ONLY, RankingStrategy.BM25_NO_DESCRIPTION,
                       RankingStrategy.HYBRID_BM25_BINARY, RankingStrategy.HYBRID_BM25_FLOAT,
                       RankingStrategy.HYBRID_BM25_BINARY_NO_DESC, RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
            yql += " where userQuery()"
        else:
            yql += " where true"
        
        # Add time filters
        filters = self._build_time_filters(start_date, end_date)
        if filters:
            if " where " in yql and "userQuery()" in yql:
                yql += " AND " + " AND ".join(filters)
            elif " where " in yql:
                yql += " AND " + " AND ".join(filters)
            else:
                yql += " where " + " AND ".join(filters)
        
        return yql
    
    def _format_binary_tensor_hex(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Format embeddings as hex-encoded binary tensor for Vespa."""
        from binascii import hexlify
        
        # Convert to binary format matching ingestion pipeline
        if embeddings.ndim == 2:
            # Multiple tokens - convert all to binary
            vectors = embeddings.astype(np.float32)
            binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(np.int8)
            
            values = {}
            for index in range(len(binarized_token_vectors)):
                values[str(index)] = str(hexlify(binarized_token_vectors[index].tobytes()), "utf-8")
            
            return {
                "type": f"tensor<int8>(querytoken{{}},v[{binarized_token_vectors.shape[1]}])",
                "values": values
            }
        else:
            # Single token
            vector = embeddings.astype(np.float32)
            binarized_vector = np.packbits(np.where(vector > 0, 1, 0)).astype(np.int8)
            
            return {
                "type": f"tensor<int8>(querytoken{{}},v[{len(binarized_vector)}])", 
                "values": {"0": str(hexlify(binarized_vector.tobytes()), "utf-8")}
            }

    def hybrid_search(self,
                     query_text: str,
                     query_embeddings: np.ndarray,
                     top_k: int = 10,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search: visual recall + text re-ranking
        
        Args:
            query_text: Text query for re-ranking
            query_embeddings: Visual embeddings for recall
            top_k: Number of results to return
            start_date: Filter by creation date (YYYY-MM-DD)
            end_date: Filter by creation date (YYYY-MM-DD)
            
        Returns:
            List of search results with frame information
        """
        if not self.vespa_app:
            raise RuntimeError("Vespa connection not available")
        
        # Format visual embeddings
        query_tensor = self._format_float_tensor(query_embeddings)
        
        # Build YQL query
        yql = f"select video_id, video_title, frame_id, start_time, end_time, frame_description, audio_transcript from {self.vespa_schema} where userQuery()"
        
        # Add time filters
        filters = self._build_time_filters(start_date, end_date)
        if filters:
            yql += " AND " + " AND ".join(filters)
        
        body = {
            "yql": yql,
            "query": query_text,  # Text for re-ranking
            "ranking": {
                "profile": "hybrid_search",
                "features": {
                    "query(qt)": query_tensor  # Visual for recall
                }
            },
            "hits": top_k
        }
        
        return self._execute_search(body, "hybrid_search")
    
    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get all available ranking strategies with descriptions for agent decision-making."""
        return RankingStrategy.get_strategy_info()
    
    def validate_strategy_inputs(self, strategy: str, query_text: str, embeddings: Optional[np.ndarray] = None) -> List[str]:
        """
        Validate that required inputs are provided for a ranking strategy.
        
        Args:
            strategy: Ranking strategy name
            query_text: Search query text
            embeddings: Visual embeddings (if available)
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            strat_enum = RankingStrategy(strategy)
        except ValueError:
            return [f"Unknown strategy: {strategy}"]
        
        # Check text query requirement
        if not query_text.strip():
            if strat_enum in [RankingStrategy.BM25_ONLY, RankingStrategy.BM25_NO_DESCRIPTION,
                            RankingStrategy.HYBRID_FLOAT_BM25, RankingStrategy.HYBRID_BINARY_BM25, 
                            RankingStrategy.HYBRID_BM25_BINARY, RankingStrategy.HYBRID_BM25_FLOAT,
                            RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC, RankingStrategy.HYBRID_BINARY_BM25_NO_DESC,
                            RankingStrategy.HYBRID_BM25_BINARY_NO_DESC, RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC]:
                errors.append(f"Strategy '{strategy}' requires a text query")
        
        # Check embeddings requirement
        strategy_info = RankingStrategy.get_strategy_info()[strategy]
        if strategy_info["requires_embeddings"] != "No" and embeddings is None:
            errors.append(f"Strategy '{strategy}' requires visual embeddings")
        
        return errors
    
    def recommend_strategy(self, 
                          query_text: str,
                          has_embeddings: bool = False,
                          speed_priority: bool = False) -> str:
        """
        Recommend optimal ranking strategy based on query characteristics.
        
        Args:
            query_text: The search query text
            has_embeddings: Whether visual embeddings are available
            speed_priority: Whether to prioritize speed over accuracy
            
        Returns:
            Recommended ranking strategy name
        """
        # Simple heuristics for recommendation
        has_visual_terms = any(term in query_text.lower() for term in 
                             ['person', 'object', 'scene', 'visual', 'image', 'video', 'show', 'see'])
        has_text_content = len(query_text.strip()) > 0
        
        return RankingStrategy.recommend_strategy(
            has_visual_component=has_embeddings and has_visual_terms,
            has_text_component=has_text_content,
            speed_priority=speed_priority
        ).value
    
    def benchmark_strategies(self,
                           query_params: Dict[str, Any], 
                           embeddings: Optional[np.ndarray] = None,
                           strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple ranking strategies on the same query.
        
        Args:
            query_params: Query parameters (without ranking strategy)
            embeddings: Visual embeddings if available
            strategies: List of strategies to test (default: all applicable)
            
        Returns:
            Dict mapping strategy names to results and performance metrics
        """
        import time
        
        if strategies is None:
            # Default to strategies that work with/without embeddings
            if embeddings is not None:
                strategies = [s.value for s in RankingStrategy]
            else:
                strategies = [RankingStrategy.BM25_ONLY.value]
        
        results = {}
        base_params = query_params.copy()
        
        for strategy in strategies:
            try:
                start_time = time.time()
                
                # Skip strategies requiring embeddings if not provided
                strategy_info = RankingStrategy.get_strategy_info()[strategy]
                if strategy_info["requires_embeddings"] != "No" and embeddings is None:
                    continue
                
                # Execute search
                search_params = base_params.copy()
                search_params["ranking"] = strategy
                search_results = self.search(search_params, embeddings)
                
                end_time = time.time()
                
                results[strategy] = {
                    "results": search_results,
                    "response_time": end_time - start_time,
                    "result_count": len(search_results),
                    "strategy_info": strategy_info
                }
                
            except Exception as e:
                results[strategy] = {
                    "error": str(e),
                    "strategy_info": strategy_info
                }
        
        return results

    def _build_time_filters(self, start_date: Optional[str], end_date: Optional[str]) -> List[str]:
        """Build time filter conditions for YQL"""
        filters = []
        
        if start_date:
            try:
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
                filters.append(f"creation_timestamp > {start_timestamp}")
            except ValueError:
                self.logger.warning(f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
                filters.append(f"creation_timestamp < {end_timestamp}")
            except ValueError:
                self.logger.warning(f"Invalid end_date format: {end_date}")
        
        return filters
    
    def _format_float_tensor(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Format embeddings as Vespa float tensor matching ColPali format"""
        if embeddings.ndim == 1:
            # Single query token - shouldn't happen for ColPali
            values = embeddings.tolist()
            return {
                "type": f"tensor<float>(querytoken{{}},v[{len(values)}])",
                "values": {"0": values}
            }
        elif embeddings.ndim == 2:
            # Multiple query tokens (standard ColPali format)
            # embeddings shape: [num_tokens, embedding_dim]
            values = {}
            for i, token_embedding in enumerate(embeddings):
                values[str(i)] = token_embedding.tolist()
            return {
                "type": f"tensor<float>(querytoken{{}},v[{embeddings.shape[1]}])",
                "values": values
            }
        else:
            raise ValueError(f"Unsupported embedding shape: {embeddings.shape}")
    
    
    def _execute_search(self, body: Dict[str, Any], search_type: str) -> List[Dict[str, Any]]:
        """Execute search query and format results"""
        import time
        
        try:
            start_time = time.time()
            self.logger.info(f"Sending Vespa query for {search_type}")
            
            # Log request body with truncated embeddings for readability
            log_body = body.copy()
            for key in ["input.query(qt)", "input.query(qtb)"]:
                if key in log_body and isinstance(log_body[key], dict):
                    # Show only first 2 embedding vectors with first 3 values each
                    sample_embeddings = {}
                    for i, (token_key, embedding) in enumerate(list(log_body[key].items())[:2]):
                        sample_embeddings[token_key] = embedding[:3] if isinstance(embedding, list) else embedding
                    log_body[key] = f"{{sample: {sample_embeddings}, total_tokens: {len(log_body[key])}}}"
            
            self.logger.info(f"Request body: {json.dumps(log_body, indent=2)}")
            
            response = self.vespa_app.query(body=body)
            
            query_time = time.time() - start_time
            self.logger.info(f"{search_type}: Found {len(response.hits)} results in {query_time:.3f}s")
            
            # Log top results with scores
            if response.hits:
                self.logger.info(f"Top {min(3, len(response.hits))} results:")
                for i, hit in enumerate(response.hits[:3]):
                    relevance = hit.get('relevance', 0.0)
                    # Handle case where relevance might be a string
                    if isinstance(relevance, str):
                        try:
                            relevance = float(relevance)
                        except ValueError:
                            relevance = 0.0
                    video_id = hit.get('fields', {}).get('video_id', 'unknown')
                    frame_id = hit.get('fields', {}).get('frame_id', 'unknown')
                    timestamp = hit.get('fields', {}).get('start_time', 'unknown')
                    self.logger.info(f"  {i+1}. {video_id} frame {frame_id} @{timestamp}s (score: {relevance:.3f})")
            
            if not response.hits:
                self.logger.warning(f"No results found for {search_type}")
            
            # Format results with frame information
            results = []
            for hit in response.hits:
                result = {
                    "relevance": hit.get("relevance", 0.0),
                    "video_id": hit["fields"].get("video_id"),
                    "video_title": hit["fields"].get("video_title"),
                    "frame_id": hit["fields"].get("frame_id") or hit["fields"].get("segment_id"),  # Handle both frame_id and segment_id
                    "start_time": hit["fields"].get("start_time"),
                    "end_time": hit["fields"].get("end_time"),
                    "frame_description": hit["fields"].get("frame_description"),
                    "audio_transcript": hit["fields"].get("audio_transcript")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search execution failed for {search_type}: {e}")
            self.logger.error(f"Failed query body: {json.dumps(body, indent=2)}")
            return []
    
    def get_video_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process search results to create video-level summary with frame tags
        
        Args:
            results: List of frame-level search results
            
        Returns:
            Dictionary with video summaries and tagged frames
        """
        video_summaries = {}
        
        for result in results:
            video_id = result["video_id"]
            
            if video_id not in video_summaries:
                video_summaries[video_id] = {
                    "video_id": video_id,
                    "video_title": result.get("video_title"),
                    "total_relevance": 0.0,
                    "frame_count": 0,
                    "frames": []
                }
            
            # Add frame information
            frame_info = {
                "frame_id": result["frame_id"],
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "relevance": result["relevance"],
                "description": result.get("frame_description"),
                "transcript": result.get("audio_transcript")
            }
            
            video_summaries[video_id]["frames"].append(frame_info)
            video_summaries[video_id]["total_relevance"] += result["relevance"]
            video_summaries[video_id]["frame_count"] += 1
        
        # Sort frames by relevance within each video
        for video_data in video_summaries.values():
            video_data["frames"].sort(key=lambda x: x["relevance"], reverse=True)
            video_data["avg_relevance"] = video_data["total_relevance"] / video_data["frame_count"]
        
        # Convert to list and sort by total relevance
        video_list = list(video_summaries.values())
        video_list.sort(key=lambda x: x["total_relevance"], reverse=True)
        
        return {
            "videos": video_list,
            "total_videos": len(video_list),
            "total_frames": sum(len(results) for _ in video_summaries)
        }
