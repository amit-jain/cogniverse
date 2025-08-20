#!/usr/bin/env python3
"""
Dynamic ProcessingStrategySet - No hardcoded strategies.

Container for processing strategies that works with any number and type of strategies.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from .processor_base import BaseStrategy


class ProcessingStrategySet:
    """Dynamic container for processing strategies - no hardcoded strategy types."""
    
    def __init__(self, **strategies):
        """
        Initialize with any number of strategies.
        
        Args:
            **strategies: Named strategies (e.g., segmentation=FrameStrategy(), transcription=AudioStrategy())
        """
        # Store strategies dynamically
        self._strategies: Dict[str, BaseStrategy] = {}
        
        for name, strategy in strategies.items():
            if isinstance(strategy, BaseStrategy):
                self._strategies[name] = strategy
            else:
                raise ValueError(f"Strategy '{name}' must extend BaseStrategy")
    
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all strategies - simple and explicit."""
        return list(self._strategies.values())
    
    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy by name."""
        return self._strategies.get(name)
    
    def has_strategy(self, name: str) -> bool:
        """Check if strategy exists."""
        return name in self._strategies
    
    def list_strategy_names(self) -> List[str]:
        """List all strategy names."""
        return list(self._strategies.keys())
    
    # Backward compatibility properties
    @property
    def segmentation(self):
        """Backward compatibility for segmentation strategy."""
        return self.get_strategy('segmentation')
    
    @property
    def transcription(self):
        """Backward compatibility for transcription strategy."""
        return self.get_strategy('transcription')
    
    @property
    def description(self):
        """Backward compatibility for description strategy."""
        return self.get_strategy('description')
    
    @property
    def embedding(self):
        """Backward compatibility for embedding strategy."""
        return self.get_strategy('embedding')
    
    async def process(self, video_path: Path, processor_manager, 
                     pipeline_context) -> Dict[str, Any]:
        """
        Process through all strategies dynamically.
        
        Args:
            video_path: Path to video file
            processor_manager: Manager containing initialized processors
            pipeline_context: Pipeline instance for context
            
        Returns:
            Processing results from all strategies
        """
        schema_name = getattr(pipeline_context, 'schema_name', 'unknown')
        pipeline_context.logger.info(f"🎯 ProcessingStrategySet.process() starting for {video_path.name} [Schema: {schema_name}]")
        
        results = {}
        
        # Process strategies in a defined order (configurable in the future)
        strategy_order = ['segmentation', 'transcription', 'description', 'embedding']
        
        for strategy_name in strategy_order:
            strategy = self.get_strategy(strategy_name)
            if strategy:
                pipeline_context.logger.info(f"Processing {strategy_name} strategy: {type(strategy).__name__}")
                
                try:
                    # Process through the strategy
                    strategy_result = await self._process_strategy(
                        strategy_name, strategy, video_path, processor_manager, pipeline_context, results
                    )
                    
                    # Merge results
                    if isinstance(strategy_result, dict):
                        results.update(strategy_result)
                        pipeline_context.logger.info(f"  ✅ {strategy_name} completed: {list(strategy_result.keys())}")
                    
                except Exception as e:
                    pipeline_context.logger.error(f"  ❌ {strategy_name} failed: {e}")
                    results[f"{strategy_name}_error"] = str(e)
        
        pipeline_context.logger.info(f"🏁 ProcessingStrategySet completed with: {list(results.keys())}")
        return results
    
    async def _process_strategy(self, strategy_name: str, strategy: BaseStrategy,
                               video_path: Path, processor_manager, pipeline_context,
                               accumulated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single strategy with proper method dispatch."""
        
        # For backward compatibility, handle known strategy types
        if strategy_name == 'segmentation':
            return await self._process_segmentation(strategy, video_path, processor_manager, pipeline_context)
        elif strategy_name == 'transcription':
            return await self._process_transcription(strategy, video_path, processor_manager, pipeline_context, accumulated_results)
        elif strategy_name == 'description':
            return await self._process_description(strategy, video_path, processor_manager, pipeline_context, accumulated_results)
        elif strategy_name == 'embedding':
            return await self._process_embedding(strategy, video_path, processor_manager, pipeline_context, accumulated_results)
        else:
            # Future: could have strategies implement a generic process() method
            pipeline_context.logger.warning(f"Unknown strategy type: {strategy_name}")
            return {}
    
    async def _process_segmentation(self, strategy, video_path: Path, processor_manager, pipeline_context) -> Dict[str, Any]:
        """Process segmentation strategy."""
        # Get the required processor type from strategy requirements
        requirements = strategy.get_required_processors()
        
        # Process based on what processor is required
        if 'keyframe' in requirements:
            processor = processor_manager.get_processor('keyframe')
            if processor:
                result = processor.extract_keyframes(video_path, pipeline_context.profile_output_dir)
                num_frames = len(result.get('keyframes', [])) if result else 0
                pipeline_context.logger.info(f"  🖼️ Extracted {num_frames} keyframes")
                return {'keyframes': result}
        
        elif 'chunk' in requirements:
            processor = processor_manager.get_processor('chunk')
            if processor:
                result = processor.extract_chunks(video_path, pipeline_context.profile_output_dir)
                num_chunks = len(result.get('chunks', [])) if result else 0
                pipeline_context.logger.info(f"  🎬 Extracted {num_chunks} video chunks")
                return {'video_chunks': result}
        
        elif 'single_vector' in requirements:
            # Handle single-vector processing through strategy's segment method
            transcript_data = accumulated_results.get('transcript')
            
            result = await strategy.segment(video_path, pipeline_context, transcript_data)
            num_segments = len(result.get('single_vector_processing', {}).get('segments', [])) if result else 0
            pipeline_context.logger.info(f"  📦 Processed {num_segments} single-vector segments")
            return result
        
        elif hasattr(strategy, 'segment') and callable(strategy.segment):
            # Handle strategies that do their own processing (like SingleVectorSegmentationStrategy)
            transcript_data = accumulated_results.get('transcript')
            
            result = await strategy.segment(video_path, pipeline_context, transcript_data)
            if 'single_vector_processing' in result:
                num_segments = len(result.get('single_vector_processing', {}).get('segments', [])) if result else 0
                pipeline_context.logger.info(f"  📦 Processed {num_segments} single-vector segments")
            return result
        
        return {}
    
    async def _process_transcription(self, strategy, video_path: Path, processor_manager, 
                                   pipeline_context, accumulated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process transcription strategy."""
        if not pipeline_context.config.transcribe_audio:
            return {}
        
        requirements = strategy.get_required_processors()
        
        if 'audio' in requirements:
            processor = processor_manager.get_processor('audio')
            if processor:
                # Pass cache to processor for caching support
                cache = getattr(pipeline_context, 'cache', None)
                result = processor.transcribe_audio(video_path, pipeline_context.profile_output_dir, cache)
                return {'transcript': result}
        
        return {}
    
    async def _process_description(self, strategy, video_path: Path, processor_manager,
                                 pipeline_context, accumulated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process description strategy."""
        if not pipeline_context.config.generate_descriptions:
            return {}
        
        requirements = strategy.get_required_processors()
        
        if 'vlm' in requirements:
            # Use the existing strategy's generate_descriptions method
            if hasattr(strategy, 'generate_descriptions'):
                segments = accumulated_results.get('segments', {})
                result = await strategy.generate_descriptions(
                    segments, video_path, pipeline_context, {}
                )
                return {'descriptions': result} if result else {}
        
        return {}
    
    async def _process_embedding(self, strategy, video_path: Path, processor_manager,
                               pipeline_context, accumulated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process embedding strategy."""
        if not pipeline_context.config.generate_embeddings:
            return {}
        
        pipeline_context.logger.info(f"🧬 Generating embeddings with strategy: {type(strategy).__name__}")
        pipeline_context.logger.info(f"  Data available: {list(accumulated_results.keys())}")
        
        # Use existing strategy's generate_embeddings_with_processor method
        if hasattr(strategy, 'generate_embeddings_with_processor'):
            embeddings = await strategy.generate_embeddings_with_processor(
                accumulated_results, 
                pipeline_context,
                processor_manager
            )
            if isinstance(embeddings, dict):
                docs_fed = embeddings.get('documents_fed', 0)
                pipeline_context.logger.info(f"  ✅ Embeddings generated: {docs_fed} documents fed to backend")
            return {'embeddings': embeddings}
        
        return {}
