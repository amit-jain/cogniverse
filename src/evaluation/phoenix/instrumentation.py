"""
OpenTelemetry instrumentation for Cogniverse components
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
import phoenix as px

logger = logging.getLogger(__name__)


class CogniverseInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentation for Cogniverse components"""
    
    def __init__(self):
        super().__init__()
        self._original_methods = {}
    
    def _instrument(self, **kwargs):
        """Instrument Cogniverse components with tracing"""
        tracer_provider = kwargs.get("tracer_provider")
        self.tracer = trace.get_tracer(__name__, "1.0.0", tracer_provider)
        
        # Instrument components
        self._instrument_video_pipeline()
        self._instrument_search_backends()
        self._instrument_query_encoders()
        self._instrument_agents()
        
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
            from src.processing.unified_video_pipeline import VideoIngestionPipeline
            
            # Store original method
            original_process = VideoIngestionPipeline.process_video
            self._original_methods['src.processing.unified_video_pipeline.VideoIngestionPipeline.process_video'] = original_process
            
            @wraps(original_process)
            def traced_process(self, video_path, *args, **kwargs):
                with self.tracer.start_as_current_span(
                    "video_pipeline.process",
                    attributes={
                        "video_path": str(video_path),
                        "profile": getattr(self, 'active_profile', 'unknown')
                    }
                ) as span:
                    try:
                        start_time = time.time()
                        result = original_process(self, video_path, *args, **kwargs)
                        
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
        """Add tracing to search operations"""
        try:
            from src.search.vespa_search_backend import VespaSearchBackend
            
            # Store original method
            original_search = VespaSearchBackend.search
            self._original_methods['src.search.vespa_search_backend.VespaSearchBackend.search'] = original_search
            
            @wraps(original_search)
            def traced_search(self, *args, **kwargs):
                query_text = kwargs.get("query_text", "")
                ranking_strategy = kwargs.get("ranking_strategy", "default")
                top_k = kwargs.get("top_k", 10)
                
                with self.tracer.start_as_current_span(
                    "search.execute",
                    attributes={
                        "backend": "vespa",
                        "query": query_text,
                        "strategy": ranking_strategy,
                        "top_k": top_k,
                        "schema": getattr(self, 'schema_name', 'unknown'),
                        "profile": getattr(self, 'profile', 'unknown')
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
                        
                        result = original_search(self, *args, **kwargs)
                        
                        # Add result metrics
                        span.set_attribute("num_results", len(result))
                        if result:
                            span.set_attribute("top_score", result[0].score if hasattr(result[0], 'score') else 0)
                        
                        span.set_attribute("latency_ms", (time.time() - start_time) * 1000)
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            VespaSearchBackend.search = traced_search
            logger.info("Instrumented VespaSearchBackend.search")
            
        except ImportError as e:
            logger.warning(f"Could not instrument search backend: {e}")
    
    def _instrument_query_encoders(self):
        """Add tracing to query encoding operations"""
        try:
            from src.agents.query_encoders import (
                ColPaliQueryEncoder, 
                ColQwenQueryEncoder,
                VideoPrismQueryEncoder
            )
            
            encoder_classes = [
                ('ColPaliQueryEncoder', ColPaliQueryEncoder),
                ('ColQwenQueryEncoder', ColQwenQueryEncoder),
                ('VideoPrismQueryEncoder', VideoPrismQueryEncoder)
            ]
            
            for encoder_name, encoder_class in encoder_classes:
                if hasattr(encoder_class, 'encode'):
                    original_encode = encoder_class.encode
                    key = f'src.agents.query_encoders.{encoder_name}.encode'
                    self._original_methods[key] = original_encode
                    
                    @wraps(original_encode)
                    def traced_encode(self, query, *args, **kwargs):
                        with self.tracer.start_as_current_span(
                            f"encoder.{encoder_name.lower()}.encode",
                            attributes={
                                "encoder_type": encoder_name,
                                "query_length": len(query) if isinstance(query, str) else 0
                            }
                        ) as span:
                            try:
                                start_time = time.time()
                                embeddings = original_encode(self, query, *args, **kwargs)
                                
                                if embeddings is not None:
                                    span.set_attribute("embedding_shape", str(embeddings.shape))
                                    span.set_attribute("embedding_dtype", str(embeddings.dtype))
                                
                                span.set_attribute("encoding_time_ms", (time.time() - start_time) * 1000)
                                span.set_status(Status(StatusCode.OK))
                                
                                return embeddings
                            except Exception as e:
                                span.record_exception(e)
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                raise
                    
                    encoder_class.encode = traced_encode
                    logger.info(f"Instrumented {encoder_name}.encode")
            
        except ImportError as e:
            logger.warning(f"Could not instrument query encoders: {e}")
    
    def _instrument_agents(self):
        """Add tracing to agent operations"""
        try:
            from src.agents.video_agent_refactored import VideoAgent
            
            if hasattr(VideoAgent, 'process_query'):
                original_process = VideoAgent.process_query
                self._original_methods['src.agents.video_agent_refactored.VideoAgent.process_query'] = original_process
                
                @wraps(original_process)
                def traced_process(self, query, *args, **kwargs):
                    with self.tracer.start_as_current_span(
                        "agent.video.process_query",
                        attributes={
                            "query": query,
                            "agent_type": "video"
                        }
                    ) as span:
                        try:
                            start_time = time.time()
                            result = original_process(self, query, *args, **kwargs)
                            
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


def instrument_cogniverse():
    """Convenience function to instrument Cogniverse with Phoenix"""
    # Launch Phoenix if not already running
    px.launch_app()
    
    # Create and apply instrumentor
    instrumentor = CogniverseInstrumentor()
    instrumentor.instrument()
    
    logger.info("Cogniverse instrumentation with Phoenix enabled")
    return instrumentor