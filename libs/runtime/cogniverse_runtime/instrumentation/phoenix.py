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
        self._instrument_search_service()
        self._instrument_backend_search()
        self._instrument_query_encoders()
        self._instrument_agents()
        self._instrument_routing_agent()
        
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
            from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

            # Store original async method
            original_process = VideoIngestionPipeline.process_video_async
            self._original_methods['src.processing.unified_video_pipeline.VideoIngestionPipeline.process_video_async'] = original_process

            # Capture tracer in closure
            tracer = self.tracer

            @wraps(original_process)
            async def traced_process(pipeline_self, video_path, *args, **kwargs):
                with tracer.start_as_current_span(
                    "video_pipeline.process",
                    attributes={
                        "video_path": str(video_path),
                        "profile": getattr(pipeline_self, 'active_profile', 'unknown')
                    }
                ) as span:
                    try:
                        start_time = time.time()
                        result = await original_process(pipeline_self, video_path, *args, **kwargs)

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

            VideoIngestionPipeline.process_video_async = traced_process
            logger.info("Instrumented VideoIngestionPipeline.process_video_async")

        except ImportError as e:
            logger.warning(f"Could not instrument video pipeline: {e}")
    
    def _instrument_search_service(self):
        """Add tracing to search operations at the service layer"""
        try:
            # Instrument at the app layer, not backend layer
            # This keeps evaluation independent of specific backends
            from cogniverse_runtime.search.service import SearchService
            
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
                    "search_service.search",
                    kind=SpanKind.SERVER,
                    attributes={
                        "openinference.span.kind": "CHAIN",
                        "operation.name": "search",
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
                                    "document_id": res.document.id if res.document else 'unknown',
                                    "video_id": res.document.metadata.get('source_id', 'unknown') if res.document else 'unknown',
                                    "score": getattr(res, 'score', 0),
                                    "content_type": str(res.document.content_type.value) if res.document and res.document.content_type else 'unknown'
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
            from cogniverse_agents.query_encoders import (
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
            from cogniverse_agents.video_agent_refactored import VideoAgent
            
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
    
    def _instrument_backend_search(self):
        """Add tracing to backend search operations"""
        try:
            from cogniverse_vespa.vespa.search_backend import VespaSearchBackend
            
            # Store original method
            original_search = VespaSearchBackend.search
            self._original_methods['src.backends.vespa.search_backend.VespaSearchBackend.search'] = original_search
            
            # Capture tracer in closure
            tracer = self.tracer
            
            @wraps(original_search)
            def traced_search(backend_self, *args, **kwargs):
                query_text = kwargs.get("query_text", "")
                ranking_strategy = kwargs.get("ranking_strategy", "default")
                top_k = kwargs.get("top_k", 10)
                
                with tracer.start_as_current_span(
                    "search.execute",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "openinference.span.kind": "RETRIEVER",
                        "operation.name": "search.execute",
                        "backend": "vespa",
                        "query": query_text,
                        "strategy": ranking_strategy,
                        "top_k": top_k,
                        "schema": getattr(backend_self, 'schema_name', 'unknown'),
                        "profile": getattr(backend_self, 'profile', 'unknown'),
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
                        
                        result = original_search(backend_self, *args, **kwargs)
                        
                        # Add result metrics
                        span.set_attribute("num_results", len(result))
                        
                        if result:
                            span.set_attribute("top_score", result[0].score if hasattr(result[0], 'score') else 0)
                            # Add details about top results as span event
                            top_3_results = []
                            for i, res in enumerate(result[:3]):
                                result_detail = {
                                    "rank": i + 1,
                                    "document_id": res.document.id if res.document else 'unknown',
                                    "video_id": res.document.metadata.get('source_id', 'unknown') if res.document else 'unknown',
                                    "score": getattr(res, 'score', 0),
                                    "content_type": str(res.document.content_type.value) if res.document and res.document.content_type else 'unknown'
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
            
            VespaSearchBackend.search = traced_search
            logger.info("Instrumented VespaSearchBackend.search")
            
        except ImportError as e:
            logger.warning(f"Could not instrument backend search: {e}")

    def _instrument_routing_agent(self):
        """Add tracing to routing agent operations"""
        try:
            from cogniverse_agents.routing_agent import RoutingAgent

            # Store original method for routing decisions
            if hasattr(RoutingAgent, 'route_query'):
                original_route = RoutingAgent.route_query
                self._original_methods['src.app.agents.routing_agent.RoutingAgent.route_query'] = original_route

                # Capture tracer in closure
                tracer = self.tracer

                @wraps(original_route)
                async def traced_route(agent_self, query: str, *args, **kwargs):
                    with tracer.start_as_current_span(
                        "routing_agent.route_query",
                        kind=SpanKind.SERVER,
                        attributes={
                            "openinference.span.kind": "AGENT",
                            "operation.name": "route_query",
                            "agent_type": "routing",
                            "query": query,
                            "input.value": json.dumps({
                                "query": query,
                                "args": str(args) if args else "",
                                "kwargs": {k: str(v) for k, v in (kwargs or {}).items()}
                            })
                        }
                    ) as span:
                        try:
                            start_time = time.time()

                            # Execute routing
                            result = await original_route(agent_self, query, *args, **kwargs)

                            # Extract routing decision details
                            if isinstance(result, dict):
                                # Record routing decision
                                chosen_agent = result.get("chosen_agent", "unknown")
                                confidence = result.get("confidence", 0.0)
                                reasoning = result.get("reasoning", "")
                                workflow_steps = result.get("workflow_steps", [])

                                span.set_attribute("routing.chosen_agent", chosen_agent)
                                span.set_attribute("routing.confidence", confidence)
                                span.set_attribute("routing.num_workflow_steps", len(workflow_steps))
                                span.set_attribute("routing.reasoning", reasoning[:500])  # Truncate long reasoning

                                # Add workflow details as event
                                if workflow_steps:
                                    workflow_summary = []
                                    for step in workflow_steps:
                                        workflow_summary.append({
                                            "step": step.get("step", 0),
                                            "agent": step.get("agent", "unknown"),
                                            "action": step.get("action", "unknown")
                                        })
                                    span.add_event("routing_workflow", {"steps": str(workflow_summary)})

                                # Set output value for Phoenix
                                span.set_attribute("output.value", json.dumps({
                                    "chosen_agent": chosen_agent,
                                    "confidence": confidence,
                                    "workflow_steps": len(workflow_steps)
                                }))

                            span.set_attribute("routing.latency_ms", (time.time() - start_time) * 1000)
                            span.set_status(Status(StatusCode.OK))

                            return result
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

                RoutingAgent.route_query = traced_route
                logger.info("Instrumented RoutingAgent.route_query")

            # Also instrument agent communication
            if hasattr(RoutingAgent, '_call_agent'):
                original_call = RoutingAgent._call_agent
                self._original_methods['src.app.agents.routing_agent.RoutingAgent._call_agent'] = original_call

                @wraps(original_call)
                async def traced_call_agent(agent_self, agent_url: str, payload: dict, *args, **kwargs):
                    with tracer.start_as_current_span(
                        "routing_agent.call_agent",
                        kind=SpanKind.CLIENT,
                        attributes={
                            "openinference.span.kind": "AGENT",
                            "operation.name": "call_agent",
                            "agent_url": agent_url,
                            "agent_type": agent_url.split('/')[-1] if '/' in agent_url else "unknown",
                            "payload_size": len(str(payload)),
                            "input.value": json.dumps(payload, default=str)[:1000]  # Truncate large payloads
                        }
                    ) as span:
                        try:
                            start_time = time.time()

                            # Execute agent call
                            result = await original_call(agent_self, agent_url, payload, *args, **kwargs)

                            # Record agent response details
                            if isinstance(result, dict):
                                span.set_attribute("agent.response_size", len(str(result)))
                                if "status" in result:
                                    span.set_attribute("agent.status", result["status"])
                                if "error" in result:
                                    span.set_attribute("agent.error", result["error"][:500])

                                # Record success/failure for optimization
                                agent_success = result.get("status") == "success" or "error" not in result
                                span.set_attribute("agent.success", agent_success)

                                # Set output value
                                span.set_attribute("output.value", json.dumps({
                                    "status": result.get("status", "unknown"),
                                    "response_size": len(str(result)),
                                    "success": agent_success
                                }, default=str))

                            span.set_attribute("agent.call_latency_ms", (time.time() - start_time) * 1000)
                            span.set_status(Status(StatusCode.OK))

                            return result
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            # Record failure for optimization
                            span.set_attribute("agent.success", False)
                            span.set_attribute("agent.error", str(e)[:500])
                            raise

                RoutingAgent._call_agent = traced_call_agent
                logger.info("Instrumented RoutingAgent._call_agent")

        except ImportError as e:
            logger.warning(f"Could not instrument routing agent: {e}")


def instrument_cogniverse():
    """Convenience function to instrument Cogniverse with Phoenix"""
    # Launch Phoenix if not already running
    px.launch_app()
    
    # Create and apply instrumentor
    instrumentor = CogniverseInstrumentor()
    instrumentor.instrument()
    
    logger.info("Cogniverse instrumentation with Phoenix enabled")
    return instrumentor