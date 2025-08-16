"""
OpenTelemetry instrumentation for Cogniverse components
"""

import logging
import time
import json
from typing import Optional, Dict, Any, Callable
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
import phoenix as px

logger = logging.getLogger(__name__)


class CogniverseInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentation for Cogniverse components"""
    
    def __init__(self):
        super().__init__()
        self._original_methods = {}
    
    def instrumentation_dependencies(self):
        """Return list of instrumentation dependencies"""
        return []
    
    def _instrument(self, **kwargs):
        """Instrument Cogniverse components with tracing"""
        tracer_provider = kwargs.get("tracer_provider")
        self.tracer = trace.get_tracer(__name__, "1.0.0", tracer_provider)
        
        # Instrument components
        self._instrument_video_pipeline()
        self._instrument_search_backends()
        self._instrument_query_encoders()
        self._instrument_agents()
        self._instrument_search_service()
        
        logger.info("Cogniverse instrumentation completed")
    
    def _uninstrument(self, **kwargs):
        """Remove instrumentation from Cogniverse components"""
        # Restore original methods
        for key, original_method in self._original_methods.items():
            module_name, class_name, method_name = key.split('.')
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                setattr(cls, method_name, original_method)
            except Exception as e:
                logger.warning(f"Failed to uninstrument {key}: {e}")
        
        self._original_methods.clear()
        logger.info("Cogniverse instrumentation removed")
    
    def _instrument_video_pipeline(self):
        """Add tracing to video processing pipeline"""
        try:
            from src.app.ingestion.pipeline import VideoIngestionPipeline
            
            # Store original method
            original_process = VideoIngestionPipeline.process_video
            self._original_methods['src.processing.unified_video_pipeline.VideoIngestionPipeline.process_video'] = original_process
            
            # Capture tracer in closure
            tracer = self.tracer
            
            @wraps(original_process)
            def traced_process(pipeline_self, video_path, *args, **kwargs):
                with tracer.start_as_current_span(
                    "video_pipeline.process",
                    attributes={
                        "video_path": str(video_path),
                        "profile": getattr(pipeline_self, 'active_profile', 'unknown')
                    }
                ) as span:
                    try:
                        start_time = time.time()
                        result = original_process(pipeline_self, video_path, *args, **kwargs)
                        
                        # Add result attributes
                        if isinstance(result, dict):
                            if "results" in result:
                                if "keyframes" in result["results"]:
                                    span.set_attribute("frames_extracted", 
                                                     len(result["results"]["keyframes"].get("keyframes", [])))
                                if "transcript" in result["results"]:
                                    span.set_attribute("transcript_segments", 
                                                     len(result["results"]["transcript"].get("segments", [])))
                        
                        span.set_attribute("duration_ms", (time.time() - start_time) * 1000)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            VideoIngestionPipeline.process_video = traced_process
            logger.info("Instrumented VideoIngestionPipeline.process_video")
            
        except ImportError as e:
            logger.warning(f"Could not instrument video pipeline: {e}")
    
    def _instrument_search_backends(self):
        """Add tracing to search operations at the service layer"""
        try:
            # Instrument at the app layer, not backend layer
            # This keeps evaluation independent of specific backends
            from src.app.search.service import SearchService
            
            # Store original methods
            original_search = SearchService.search
            self._original_methods['src.app.search.service.SearchService.search'] = original_search
            
            # Capture tracer in closure
            tracer = self.tracer
            
            @wraps(original_search)
            def traced_search(service_self, query: str, *args, **kwargs):
                # SearchService.search takes query as first positional arg
                query_text = query
                ranking_strategy = kwargs.get("ranking_strategy", "default")
                top_k = kwargs.get("top_k", 10)
                
                with tracer.start_as_current_span(
                    "search.execute",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "openinference.span.kind": "RETRIEVER",
                        "operation.name": "search.execute",
                        "backend": getattr(service_self.config, 'search_backend', 'vespa'),
                        "query": query_text,
                        "strategy": ranking_strategy,
                        "top_k": top_k,
                        "profile": getattr(service_self, 'profile', 'unknown'),
                        # Add input.value for Phoenix
                        "input.value": json.dumps({
                            "query": query_text,
                            "top_k": top_k,
                            "strategy": ranking_strategy or "default"
                        })
                    }
                ) as span:
                    try:
                        start_time = time.time()
                        
                        # Add query embeddings info
                        if "query_embeddings" in kwargs and kwargs["query_embeddings"] is not None:
                            span.set_attribute("has_embeddings", True)
                            span.set_attribute("embedding_shape", str(kwargs["query_embeddings"].shape))
                        else:
                            span.set_attribute("has_embeddings", False)
                        
                        result = original_search(service_self, query, *args, **kwargs)
                        
                        # Add result metrics
                        span.set_attribute("num_results", len(result))
                        
                        if result:
                            span.set_attribute("top_score", result[0].score if hasattr(result[0], 'score') else 0)
                            # Add details about top results as span event
                            top_3_results = []
                            for i, res in enumerate(result[:3]):
                                result_detail = {
                                    "rank": i + 1,
                                    "document_id": res.document.doc_id if res.document else 'unknown',
                                    "video_id": res.document.metadata.get('source_id', 'unknown') if res.document else 'unknown',
                                    "score": getattr(res, 'score', 0),
                                    "media_type": str(res.document.media_type.value) if res.document and res.document.media_type else 'unknown'
                                }
                                top_3_results.append(result_detail)
                            span.add_event("search_results", {"top_3": str(top_3_results)})
                        
                        span.set_attribute("latency_ms", (time.time() - start_time) * 1000)
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            SearchService.search = traced_search
            logger.info("Instrumented SearchService.search")
            
        except ImportError as e:
            logger.warning(f"Could not instrument search backend: {e}")
    
    def _instrument_query_encoders(self):
        """Add tracing to query encoding operations"""
        try:
            from src.app.agents.query_encoders import (
                ColPaliQueryEncoder, 
                ColQwenQueryEncoder,
                VideoPrismQueryEncoder
            )
            
            encoder_classes = [
                ('ColPaliQueryEncoder', ColPaliQueryEncoder),
                ('ColQwenQueryEncoder', ColQwenQueryEncoder),
                ('VideoPrismQueryEncoder', VideoPrismQueryEncoder)
            ]
            
            def create_traced_encode(encoder_name, original_encode, tracer):
                """Factory function to create traced encode with proper closure"""
                @wraps(original_encode)
                def traced_encode(encoder_self, query, *args, **kwargs):
                    with tracer.start_as_current_span(
                        f"encoder.{encoder_name.lower()}.encode",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            "openinference.span.kind": "EMBEDDING",
                            "operation.name": f"encode.{encoder_name.lower()}",
                            "encoder_type": encoder_name,
                            "query_length": len(query) if isinstance(query, str) else 0,
                            "input.value": query if isinstance(query, str) else str(query)
                        }
                    ) as span:
                        try:
                            start_time = time.time()
                            embeddings = original_encode(encoder_self, query, *args, **kwargs)
                            
                            if embeddings is not None:
                                span.set_attribute("embedding_shape", str(embeddings.shape))
                                span.set_attribute("embedding_dtype", str(embeddings.dtype))
                                # Add embedding statistics for debugging
                                import numpy as np
                                if hasattr(embeddings, 'shape'):
                                    if len(embeddings.shape) == 2:  # Multi-vector embeddings
                                        span.set_attribute("num_vectors", embeddings.shape[0])
                                        span.set_attribute("embedding_dim", embeddings.shape[1])
                                    span.set_attribute("embedding_norm_mean", float(np.mean(np.linalg.norm(embeddings, axis=-1))))
                                    span.set_attribute("embedding_norm_std", float(np.std(np.linalg.norm(embeddings, axis=-1))))
                                    
                                    # Set output.value with just shape info
                                    span.set_attribute("output.value", str(embeddings.shape))
                            
                            span.set_attribute("encoding_time_ms", (time.time() - start_time) * 1000)
                            span.set_status(Status(StatusCode.OK))
                            
                            return embeddings
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise
                return traced_encode
            
            for encoder_name, encoder_class in encoder_classes:
                if hasattr(encoder_class, 'encode'):
                    original_encode = encoder_class.encode
                    key = f'src.agents.query_encoders.{encoder_name}.encode'
                    self._original_methods[key] = original_encode
                    
                    # Create traced version with proper closure
                    traced_encode = create_traced_encode(encoder_name, original_encode, self.tracer)
                    
                    encoder_class.encode = traced_encode
                    logger.info(f"Instrumented {encoder_name}.encode")
            
        except ImportError as e:
            logger.warning(f"Could not instrument query encoders: {e}")
    
    def _instrument_agents(self):
        """Add tracing to agent operations"""
        try:
            from src.app.agents.video_agent_refactored import VideoAgent
            
            if hasattr(VideoAgent, 'process_query'):
                original_process = VideoAgent.process_query
                self._original_methods['src.agents.video_agent_refactored.VideoAgent.process_query'] = original_process
                
                # Capture tracer in closure
                tracer = self.tracer
                
                @wraps(original_process)
                def traced_process(agent_self, query, *args, **kwargs):
                    with tracer.start_as_current_span(
                        "agent.video.process_query",
                        attributes={
                            "query": query,
                            "agent_type": "video"
                        }
                    ) as span:
                        try:
                            start_time = time.time()
                            result = original_process(agent_self, query, *args, **kwargs)
                            
                            if isinstance(result, dict):
                                if "results" in result:
                                    span.set_attribute("num_results", len(result.get("results", [])))
                                if "metadata" in result:
                                    for key, value in result["metadata"].items():
                                        if isinstance(value, (str, int, float, bool)):
                                            span.set_attribute(f"metadata.{key}", value)
                            
                            span.set_attribute("processing_time_ms", (time.time() - start_time) * 1000)
                            span.set_status(Status(StatusCode.OK))
                            
                            return result
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise
                
                VideoAgent.process_query = traced_process
                logger.info("Instrumented VideoAgent.process_query")
                
        except ImportError as e:
            logger.warning(f"Could not instrument agents: {e}")
    
    def _instrument_search_service(self):
        """Add tracing to search service operations"""
        try:
            from src.app.search.service import SearchService
            
            # Store original method
            original_search = SearchService.search
            self._original_methods['src.search.search_service.SearchService.search'] = original_search
            
            # Capture tracer in closure
            tracer = self.tracer
            
            @wraps(original_search)
            def traced_search(service_self, query, *args, **kwargs):
                # Extract kwargs with defaults
                top_k = kwargs.get("top_k", 10)
                filters = kwargs.get("filters")
                ranking_strategy = kwargs.get("ranking_strategy")
                
                with tracer.start_as_current_span(
                    "search_service.search",
                    kind=SpanKind.SERVER,
                    attributes={
                        "openinference.span.kind": "CHAIN",
                        "operation.name": "search",
                        "query": query,
                        "top_k": top_k,
                        "profile": getattr(service_self, 'profile', 'unknown'),
                        "ranking_strategy": ranking_strategy or "default",
                        "has_filters": filters is not None,
                        "input.value": query,
                        "input.mime_type": "text/plain"
                    }
                ) as span:
                    try:
                        # Call original method - NO DUPLICATION!
                        results = original_search(service_self, query, *args, **kwargs)
                        
                        # Add result attributes
                        span.set_attribute("num_results", len(results))
                        if results:
                            span.set_attribute("top_score", results[0].score)
                            
                            # Build output documents for Phoenix
                            output_documents = []
                            for i, result in enumerate(results[:5]):  # Top 5 results
                                output_documents.append({
                                    "document_id": result.document.doc_id if result.document else 'unknown',
                                    "video_id": result.document.metadata.get('source_id', 'unknown') if result.document else 'unknown'
                                })
                            
                            span.set_attribute("output.value", json.dumps(output_documents))
                            span.set_attribute("output.mime_type", "application/json")
                        
                        span.set_status(Status(StatusCode.OK))
                        return results
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            SearchService.search = traced_search
            logger.info("Instrumented SearchService.search")
            
        except ImportError as e:
            logger.warning(f"Could not instrument search service: {e}")


def instrument_cogniverse():
    """Convenience function to instrument Cogniverse with Phoenix"""
    # Launch Phoenix if not already running
    px.launch_app()
    
    # Create and apply instrumentor
    instrumentor = CogniverseInstrumentor()
    instrumentor.instrument()
    
    logger.info("Cogniverse instrumentation with Phoenix enabled")
    return instrumentor