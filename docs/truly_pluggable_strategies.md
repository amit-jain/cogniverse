# Truly Pluggable Strategy Architecture

## Current Problems

The current design hardcodes strategy types in multiple places:

### 1. ProcessingStrategySet Constructor
```python
def __init__(self,
             segmentation: SegmentationStrategy,     # HARDCODED
             transcription: TranscriptionStrategy,   # HARDCODED  
             description: DescriptionStrategy,       # HARDCODED
             embedding: EmbeddingStrategy):          # HARDCODED
```

### 2. ProcessorManager.initialize_from_strategies
```python
for strategy in [strategies.segmentation_strategy,      # HARDCODED
                 strategies.transcription_strategy,     # HARDCODED
                 strategies.description_strategy,       # HARDCODED
                 strategies.embedding_strategy]:        # HARDCODED
```

### 3. ProcessingStrategySet.process() Method
```python
if isinstance(self.segmentation, FrameSegmentationStrategy):     # HARDCODED
    # ...
elif isinstance(self.segmentation, ChunkSegmentationStrategy):   # HARDCODED
    # ...
elif isinstance(self.segmentation, SingleVectorSegmentationStrategy): # HARDCODED
```

## Solution: Dynamic Strategy Architecture

### Base Strategy Protocol

```python
# src/app/ingestion/strategy_base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol
from pathlib import Path

class ProcessingStrategy(Protocol):
    """Protocol that all processing strategies must implement."""
    
    def get_strategy_type(self) -> str:
        """Return the type/category of this strategy (e.g., 'segmentation', 'transcription')."""
        ...
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Return required processors and their configurations."""
        ...
    
    async def process(self, data: Any, pipeline_context: Any, **kwargs) -> Dict[str, Any]:
        """Process the data according to this strategy."""
        ...


class BaseStrategy(ABC):
    """Base class for all processing strategies."""
    
    @abstractmethod
    def get_strategy_type(self) -> str:
        """Return the type/category of this strategy."""
        pass
    
    @abstractmethod
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Return required processors and their configurations."""
        pass
    
    @abstractmethod
    async def process(self, data: Any, pipeline_context: Any, **kwargs) -> Dict[str, Any]:
        """Process the data according to this strategy."""
        pass
```

### Dynamic ProcessingStrategySet

```python
# src/app/ingestion/processing_strategy_set.py

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

class ProcessingStrategySet:
    """Dynamic container for processing strategies."""
    
    def __init__(self, strategies: List[ProcessingStrategy]):
        # Store strategies dynamically by type
        self._strategies: Dict[str, ProcessingStrategy] = {}
        
        for strategy in strategies:
            strategy_type = strategy.get_strategy_type()
            self._strategies[strategy_type] = strategy
    
    def get_strategy(self, strategy_type: str) -> Optional[ProcessingStrategy]:
        """Get a strategy by type."""
        return self._strategies.get(strategy_type)
    
    def get_all_strategies(self) -> List[ProcessingStrategy]:
        """Get all strategies."""
        return list(self._strategies.values())
    
    def get_strategy_types(self) -> List[str]:
        """Get all strategy types."""
        return list(self._strategies.keys())
    
    async def process(self, video_path: Path, processor_manager: Any, 
                     pipeline_context: Any) -> Dict[str, Any]:
        """Process through all strategies dynamically."""
        schema_name = getattr(pipeline_context, 'schema_name', 'unknown')
        pipeline_context.logger.info(f"🎯 ProcessingStrategySet.process() starting for {video_path.name} [Schema: {schema_name}]")
        
        results = {}
        
        # Process strategies in a defined order
        strategy_order = ['segmentation', 'transcription', 'description', 'embedding']
        
        for strategy_type in strategy_order:
            strategy = self.get_strategy(strategy_type)
            if strategy:
                pipeline_context.logger.info(f"Processing {strategy_type} strategy: {type(strategy).__name__}")
                
                try:
                    # Pass current results as context for chaining
                    strategy_result = await strategy.process(
                        data=results,
                        pipeline_context=pipeline_context,
                        video_path=video_path,
                        processor_manager=processor_manager
                    )
                    
                    # Merge results
                    if isinstance(strategy_result, dict):
                        results.update(strategy_result)
                        pipeline_context.logger.info(f"  ✅ {strategy_type} completed: {list(strategy_result.keys())}")
                    
                except Exception as e:
                    pipeline_context.logger.error(f"  ❌ {strategy_type} failed: {e}")
                    results[f"{strategy_type}_error"] = str(e)
        
        pipeline_context.logger.info(f"🏁 ProcessingStrategySet completed with: {list(results.keys())}")
        return results
```

### Dynamic ProcessorManager

```python
# src/app/ingestion/processor_manager.py

class ProcessorManager:
    """Plugin-based processor manager with auto-discovery."""
    
    def __init__(self, logger: logging.Logger, plugin_dir: Optional[Path] = None):
        self.logger = logger
        self._processors: Dict[str, Any] = {}
        self._processor_classes: Dict[str, type] = {}
        
        # Auto-discover processors
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent / "processors"
        
        self._discover_processors(plugin_dir)
    
    def initialize_from_strategies(self, strategy_set: 'ProcessingStrategySet'):
        """Initialize processors dynamically from ALL strategies."""
        all_requirements = {}
        
        # Collect requirements from ALL strategies dynamically
        for strategy in strategy_set.get_all_strategies():
            requirements = strategy.get_required_processors()
            all_requirements.update(requirements)
            
            strategy_type = strategy.get_strategy_type()
            self.logger.info(f"🔧 {strategy_type} strategy requires: {list(requirements.keys())}")
        
        self._init_from_requirements(all_requirements)
    
    def _discover_processors(self, plugin_dir: Path):
        """Auto-discover processor plugins."""
        if not plugin_dir.exists():
            self.logger.warning(f"Plugin directory not found: {plugin_dir}")
            return
        
        # Import all modules in processors directory
        package_name = "src.app.ingestion.processors"
        
        for importer, modname, ispkg in pkgutil.iter_modules([str(plugin_dir)]):
            if not ispkg:  # Skip packages, only load modules
                full_module_name = f"{package_name}.{modname}"
                try:
                    module = importlib.import_module(full_module_name)
                    
                    # Look for processor classes with PROCESSOR_NAME attribute
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, 'PROCESSOR_NAME') and 
                            hasattr(attr, 'from_config')):
                            
                            processor_name = attr.PROCESSOR_NAME
                            self._processor_classes[processor_name] = attr
                            self.logger.debug(f"Discovered processor: {processor_name} from {modname}")
                            
                except Exception as e:
                    self.logger.error(f"Failed to load module {full_module_name}: {e}")
    
    def _init_from_requirements(self, required_processors: Dict[str, Dict[str, Any]]):
        """Dynamically create processors from requirements."""
        for processor_name, processor_config in required_processors.items():
            if processor_name not in self._processor_classes:
                self._try_load_processor(processor_name)
            
            if processor_name in self._processor_classes:
                processor_class = self._processor_classes[processor_name]
                self.logger.info(f"🔧 Creating {processor_name} with config: {processor_config}")
                
                processor = processor_class.from_config(processor_config, self.logger)
                self._processors[processor_name] = processor
            else:
                raise ValueError(f"Unknown processor type: {processor_name}")
    
    def get_processor(self, name: str) -> Optional[Any]:
        """Get a processor by name."""
        return self._processors.get(name)
```

### Updated Strategy Implementations

```python
# src/app/ingestion/strategies/segmentation_strategies.py

class FrameSegmentationStrategy(BaseStrategy):
    """Extract frames from video (e.g., for ColPali)"""
    
    def __init__(self, fps: float = 1.0):
        self.fps = fps
    
    def get_strategy_type(self) -> str:
        return "segmentation"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {
            'keyframe': {
                'fps': self.fps,
                'threshold': 0.98
            }
        }
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any, 
                     video_path: Path, processor_manager: Any, **kwargs) -> Dict[str, Any]:
        """Extract keyframes using the keyframe processor."""
        processor = processor_manager.get_processor('keyframe')
        if not processor:
            return {"error": "Keyframe processor not available"}
        
        segmentation_result = processor.extract_keyframes(
            video_path, pipeline_context.profile_output_dir
        )
        
        num_frames = len(segmentation_result.get('keyframes', [])) if segmentation_result else 0
        pipeline_context.logger.info(f"  🖼️ Extracted {num_frames} keyframes")
        
        return {'keyframes': segmentation_result}


class ChunkSegmentationStrategy(BaseStrategy):
    """Extract video chunks (e.g., for ColQwen)"""
    
    def __init__(self, chunk_duration: float = 30.0):
        self.chunk_duration = chunk_duration
    
    def get_strategy_type(self) -> str:
        return "segmentation"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {
            'chunk': {
                'duration': self.chunk_duration,
                'overlap': 2.0
            }
        }
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any,
                     video_path: Path, processor_manager: Any, **kwargs) -> Dict[str, Any]:
        """Extract video chunks."""
        processor = processor_manager.get_processor('chunk')
        if not processor:
            return {"error": "Chunk processor not available"}
        
        segmentation_result = processor.extract_chunks(
            video_path, pipeline_context.profile_output_dir
        )
        
        num_chunks = len(segmentation_result.get('chunks', [])) if segmentation_result else 0
        pipeline_context.logger.info(f"  🎬 Extracted {num_chunks} video chunks")
        
        return {'video_chunks': segmentation_result}


class AudioTranscriptionStrategy(BaseStrategy):
    """Transcribe audio from video"""
    
    def get_strategy_type(self) -> str:
        return "transcription"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {
            'audio': {
                'model': 'whisper-large-v3',
                'language': 'auto'
            }
        }
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any,
                     video_path: Path, processor_manager: Any, **kwargs) -> Dict[str, Any]:
        """Transcribe audio if enabled."""
        if not pipeline_context.config.transcribe_audio:
            return {}
        
        processor = processor_manager.get_processor('audio')
        if not processor:
            return {"error": "Audio processor not available"}
        
        transcript = processor.transcribe_audio(
            video_path, pipeline_context.profile_output_dir
        )
        
        return {'transcript': transcript}


class VLMDescriptionStrategy(BaseStrategy):
    """Generate descriptions using VLM"""
    
    def __init__(self, model_name: str = "gpt-4-vision", batch_size: int = 10):
        self.model_name = model_name
        self.batch_size = batch_size
    
    def get_strategy_type(self) -> str:
        return "description"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {
            'vlm': {
                'model_name': self.model_name,
                'batch_size': self.batch_size
            }
        }
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any,
                     video_path: Path, processor_manager: Any, **kwargs) -> Dict[str, Any]:
        """Generate descriptions if enabled."""
        if not pipeline_context.config.generate_descriptions:
            return {}
        
        processor = processor_manager.get_processor('vlm')
        if not processor:
            return {}
        
        # Use existing keyframes or segments
        segments = data.get('keyframes', data.get('segments', {}))
        
        descriptions = await self._generate_descriptions_for_segments(
            segments, video_path, pipeline_context, processor
        )
        
        return {'descriptions': descriptions}


class MultiVectorEmbeddingStrategy(BaseStrategy):
    """Generate multi-vector embeddings"""
    
    def get_strategy_type(self) -> str:
        return "embedding"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        # Embedding generation uses the generic embedding generator
        return {
            'embedding': {
                'type': 'multi_vector'
            }
        }
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any,
                     video_path: Path, processor_manager: Any, **kwargs) -> Dict[str, Any]:
        """Generate embeddings if enabled."""
        if not pipeline_context.config.generate_embeddings:
            return {}
        
        # Prepare data for embedding generation
        wrapped_results = {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "results": data
        }
        
        embeddings = pipeline_context.generate_embeddings(wrapped_results)
        
        if isinstance(embeddings, dict):
            docs_fed = embeddings.get('documents_fed', 0)
            pipeline_context.logger.info(f"  ✅ Embeddings generated: {docs_fed} documents fed to backend")
        
        return {'embeddings': embeddings}
```

### Strategy Factory

```python
# src/app/ingestion/strategy_factory.py

from typing import Dict, Any, List
from .strategy_base import ProcessingStrategy
from .processing_strategy_set import ProcessingStrategySet

class StrategyFactory:
    """Factory for creating strategy sets from configuration."""
    
    _strategy_classes: Dict[str, type] = {}
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_name: str, strategy_class: type):
        """Register a strategy class."""
        key = f"{strategy_type}.{strategy_name}"
        cls._strategy_classes[key] = strategy_class
    
    @classmethod
    def create_strategy_set(cls, profile_config: Dict[str, Any]) -> ProcessingStrategySet:
        """Create a strategy set from profile configuration."""
        strategies = []
        
        # Parse strategy configuration
        strategy_configs = profile_config.get('strategies', {})
        
        for strategy_type, strategy_config in strategy_configs.items():
            strategy_name = strategy_config.get('name')
            strategy_params = strategy_config.get('params', {})
            
            key = f"{strategy_type}.{strategy_name}"
            if key in cls._strategy_classes:
                strategy_class = cls._strategy_classes[key]
                strategy = strategy_class(**strategy_params)
                strategies.append(strategy)
            else:
                raise ValueError(f"Unknown strategy: {key}")
        
        return ProcessingStrategySet(strategies)


# Register strategies
StrategyFactory.register_strategy("segmentation", "frame", FrameSegmentationStrategy)
StrategyFactory.register_strategy("segmentation", "chunk", ChunkSegmentationStrategy)
StrategyFactory.register_strategy("segmentation", "single_vector", SingleVectorSegmentationStrategy)
StrategyFactory.register_strategy("transcription", "audio", AudioTranscriptionStrategy)
StrategyFactory.register_strategy("description", "vlm", VLMDescriptionStrategy)
StrategyFactory.register_strategy("embedding", "multi_vector", MultiVectorEmbeddingStrategy)
StrategyFactory.register_strategy("embedding", "single_vector", SingleVectorEmbeddingStrategy)
```

### Configuration Format

```json
{
  "video_processing_profiles": {
    "frame_based_colpali": {
      "schema_name": "video_colpali_smol500_mv_frame",
      "strategies": {
        "segmentation": {
          "name": "frame",
          "params": {"fps": 1.0}
        },
        "transcription": {
          "name": "audio",
          "params": {"model": "whisper-large-v3"}
        },
        "description": {
          "name": "vlm",
          "params": {"model_name": "gpt-4-vision", "batch_size": 10}
        },
        "embedding": {
          "name": "multi_vector",
          "params": {}
        }
      }
    },
    "pdf_extraction": {
      "schema_name": "document_pdf_colpali",
      "strategies": {
        "segmentation": {
          "name": "page",
          "params": {"extract_images": true, "dpi": 150}
        },
        "transcription": {
          "name": "ocr",
          "params": {"language": "eng"}
        },
        "description": {
          "name": "vlm",
          "params": {"model_name": "gpt-4-vision"}
        },
        "embedding": {
          "name": "multi_vector",
          "params": {}
        }
      }
    }
  }
}
```

### Adding New Strategy Types

To add a new strategy type (e.g., "preprocessing"):

1. **Create the strategy class:**
```python
class ImagePreprocessingStrategy(BaseStrategy):
    def get_strategy_type(self) -> str:
        return "preprocessing"
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {"image_processor": {"resize": True, "normalize": True}}
    
    async def process(self, data: Dict[str, Any], pipeline_context: Any, **kwargs) -> Dict[str, Any]:
        # Implementation
        pass
```

2. **Register it:**
```python
StrategyFactory.register_strategy("preprocessing", "image", ImagePreprocessingStrategy)
```

3. **Update processing order:**
```python
# In ProcessingStrategySet.process()
strategy_order = ['preprocessing', 'segmentation', 'transcription', 'description', 'embedding']
```

4. **Add to config:**
```json
{
  "strategies": {
    "preprocessing": {
      "name": "image",
      "params": {"resize_width": 1024}
    },
    // ... other strategies
  }
}
```

## Benefits

1. **Zero hardcoding** - No strategy types hardcoded anywhere
2. **Dynamic discovery** - Strategies and processors auto-discovered
3. **Easy extension** - Add new strategy types without modifying existing code
4. **Clean separation** - Each strategy is self-contained
5. **Configuration-driven** - Everything configured via JSON
6. **Order flexibility** - Processing order can be configured
7. **Error isolation** - One strategy failure doesn't break others

This architecture is truly pluggable - adding new document types, processing steps, or strategy variations requires no changes to the core pipeline code!