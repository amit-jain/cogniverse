# Truly Pluggable ProcessorManager Design

## The Problem

The current ProcessorManager is NOT pluggable because:
1. It has hardcoded processor attributes (`self.pdf_processor`, `self.ocr_processor`, etc.)
2. It has hardcoded initialization methods for each processor type
3. Adding a new processor type requires modifying the ProcessorManager class

## The Solution: Registry-Based Plugin Architecture

### Option 1: Dynamic Processor Registry

```python
# src/app/ingestion/processor_manager.py

from typing import Dict, Any, Type, Optional
import logging
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Base class for all processors."""
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        pass


class ProcessorRegistry:
    """Global registry for processor types."""
    
    _processors: Dict[str, Type[BaseProcessor]] = {}
    
    @classmethod
    def register(cls, name: str, processor_class: Type[BaseProcessor]):
        """Register a processor class."""
        cls._processors[name] = processor_class
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any], logger: logging.Logger) -> BaseProcessor:
        """Create a processor instance."""
        if name not in cls._processors:
            raise ValueError(f"Unknown processor type: {name}")
        
        processor_class = cls._processors[name]
        return processor_class(**config, logger=logger)
    
    @classmethod
    def list_processors(cls) -> list:
        """List all registered processors."""
        return list(cls._processors.keys())


class ProcessorManager:
    """Truly pluggable processor manager using dynamic registry."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._processors: Dict[str, BaseProcessor] = {}  # Dynamic storage
    
    def initialize_from_strategies(self, strategies: 'ProcessingStrategySet'):
        """Initialize processors based on strategy requirements."""
        all_requirements = {}
        
        for strategy in [strategies.segmentation_strategy,
                         strategies.transcription_strategy,
                         strategies.description_strategy,
                         strategies.embedding_strategy]:
            if strategy:
                requirements = strategy.get_required_processors()
                all_requirements.update(requirements)
        
        self._init_from_requirements(all_requirements)
    
    def _init_from_requirements(self, required_processors: Dict[str, Dict[str, Any]]):
        """Initialize processors dynamically from requirements."""
        for processor_name, processor_config in required_processors.items():
            self.logger.info(f"🔧 Creating processor: {processor_name}")
            try:
                processor = ProcessorRegistry.create(
                    processor_name,
                    processor_config,
                    self.logger
                )
                self._processors[processor_name] = processor
            except ValueError as e:
                self.logger.error(f"Failed to create processor {processor_name}: {e}")
                raise
    
    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """Get a processor by name."""
        return self._processors.get(name)
    
    def __getattr__(self, name: str):
        """Dynamic attribute access for backward compatibility."""
        # Convert snake_case to processor name (e.g., pdf_processor -> pdf_processor)
        if name in self._processors:
            return self._processors[name]
        raise AttributeError(f"No processor named '{name}'")
```

### Option 2: Plugin-Based Architecture with Auto-Discovery

```python
# src/app/ingestion/processor_manager.py

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Any, Optional, Protocol
import logging


class ProcessorProtocol(Protocol):
    """Protocol that all processors must follow."""
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger: logging.Logger) -> 'ProcessorProtocol':
        """Factory method to create processor from config."""
        ...
    
    def process(self, *args, **kwargs) -> Any:
        """Process method that all processors must implement."""
        ...


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
                    
                    # Look for processor classes
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
    
    def initialize_from_strategies(self, strategies: 'ProcessingStrategySet'):
        """Initialize processors based on strategy requirements."""
        all_requirements = {}
        
        for strategy in [strategies.segmentation_strategy,
                         strategies.transcription_strategy,
                         strategies.description_strategy,
                         strategies.embedding_strategy]:
            if strategy:
                requirements = strategy.get_required_processors()
                all_requirements.update(requirements)
        
        self._init_from_requirements(all_requirements)
    
    def _init_from_requirements(self, required_processors: Dict[str, Dict[str, Any]]):
        """Dynamically create processors from requirements."""
        for processor_name, processor_config in required_processors.items():
            if processor_name not in self._processor_classes:
                # Try to load it dynamically
                self._try_load_processor(processor_name)
            
            if processor_name in self._processor_classes:
                processor_class = self._processor_classes[processor_name]
                self.logger.info(f"🔧 Creating {processor_name} with config: {processor_config}")
                
                processor = processor_class.from_config(processor_config, self.logger)
                self._processors[processor_name] = processor
            else:
                raise ValueError(f"Unknown processor type: {processor_name}")
    
    def _try_load_processor(self, processor_name: str):
        """Try to load a processor module dynamically."""
        # Convert processor_name to module name (e.g., pdf_processor -> pdf_processor)
        module_name = processor_name.replace('_processor', '')
        
        try:
            module_path = f"src.app.ingestion.processors.{module_name}_processor"
            module = importlib.import_module(module_path)
            
            # Look for the processor class
            class_name = ''.join(word.capitalize() for word in processor_name.split('_'))
            if hasattr(module, class_name):
                processor_class = getattr(module, class_name)
                self._processor_classes[processor_name] = processor_class
                self.logger.info(f"Dynamically loaded processor: {processor_name}")
        except Exception as e:
            self.logger.debug(f"Could not dynamically load {processor_name}: {e}")
    
    def get_processor(self, name: str) -> Optional[Any]:
        """Get a processor by name."""
        return self._processors.get(name)
    
    def __getattr__(self, name: str):
        """Dynamic attribute access."""
        if name in self._processors:
            return self._processors[name]
        return None
```

### Option 3: Factory Pattern with Processor Factories

```python
# src/app/ingestion/processor_factories.py

from typing import Dict, Any, Callable, Optional
import logging
from abc import ABC, abstractmethod


class ProcessorFactory(ABC):
    """Abstract factory for creating processors."""
    
    @abstractmethod
    def create(self, config: Dict[str, Any], logger: logging.Logger) -> Any:
        """Create a processor instance."""
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """Get the name of the processor this factory creates."""
        pass


class FactoryRegistry:
    """Registry of processor factories."""
    
    _factories: Dict[str, ProcessorFactory] = {}
    
    @classmethod
    def register_factory(cls, factory: ProcessorFactory):
        """Register a processor factory."""
        name = factory.get_processor_name()
        cls._factories[name] = factory
    
    @classmethod
    def get_factory(cls, name: str) -> Optional[ProcessorFactory]:
        """Get a factory by processor name."""
        return cls._factories.get(name)
    
    @classmethod
    def create_processor(cls, name: str, config: Dict[str, Any], logger: logging.Logger) -> Any:
        """Create a processor using the appropriate factory."""
        factory = cls.get_factory(name)
        if not factory:
            raise ValueError(f"No factory registered for processor: {name}")
        return factory.create(config, logger)


class ProcessorManager:
    """Manager using factory pattern."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._processors: Dict[str, Any] = {}
    
    def initialize_from_strategies(self, strategies: 'ProcessingStrategySet'):
        """Initialize processors from strategy requirements."""
        all_requirements = {}
        
        for strategy in [strategies.segmentation_strategy,
                         strategies.transcription_strategy,
                         strategies.description_strategy,
                         strategies.embedding_strategy]:
            if strategy:
                requirements = strategy.get_required_processors()
                all_requirements.update(requirements)
        
        for processor_name, processor_config in all_requirements.items():
            self.logger.info(f"🔧 Creating {processor_name}")
            processor = FactoryRegistry.create_processor(
                processor_name,
                processor_config,
                self.logger
            )
            self._processors[processor_name] = processor
    
    def get_processor(self, name: str) -> Optional[Any]:
        """Get processor by name."""
        return self._processors.get(name)
    
    def __getattr__(self, name: str):
        """Dynamic attribute access."""
        return self._processors.get(name)


# Example: Registering PDF processor factory
class PDFProcessorFactory(ProcessorFactory):
    def get_processor_name(self) -> str:
        return "pdf_processor"
    
    def create(self, config: Dict[str, Any], logger: logging.Logger) -> Any:
        from src.app.ingestion.processors.pdf_processor import PDFProcessor
        return PDFProcessor(**config, logger=logger)

# Register at module load time
FactoryRegistry.register_factory(PDFProcessorFactory())
```

## Example: How to Add a New Processor

### With Option 1 (Registry):

```python
# src/app/ingestion/processors/excel_processor.py

from src.app.ingestion.processor_manager import BaseProcessor, ProcessorRegistry
import pandas as pd

class ExcelProcessor(BaseProcessor):
    """Processor for Excel files."""
    
    def __init__(self, sheet_names: list = None, 
                 extract_formulas: bool = False,
                 logger: logging.Logger = None, **kwargs):
        self.sheet_names = sheet_names
        self.extract_formulas = extract_formulas
        self.logger = logger
    
    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel file."""
        df = pd.read_excel(file_path, sheet_name=self.sheet_names)
        return {"sheets": df, "metadata": {...}}

# Register the processor
ProcessorRegistry.register("excel_processor", ExcelProcessor)
```

### With Option 2 (Auto-Discovery):

```python
# src/app/ingestion/processors/excel_processor.py

class ExcelProcessor:
    """Processor for Excel files."""
    
    PROCESSOR_NAME = "excel_processor"  # This triggers auto-discovery
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger: logging.Logger):
        """Factory method for creating from config."""
        return cls(
            sheet_names=config.get('sheet_names'),
            extract_formulas=config.get('extract_formulas', False),
            logger=logger
        )
    
    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel file."""
        # Implementation
        pass

# No registration needed - auto-discovered!
```

### With Option 3 (Factory Pattern):

```python
# src/app/ingestion/processors/excel_processor.py

from src.app.ingestion.processor_factories import ProcessorFactory, FactoryRegistry

class ExcelProcessor:
    """Excel file processor."""
    def __init__(self, **config):
        self.config = config
    
    def process(self, file_path: Path):
        # Implementation
        pass

class ExcelProcessorFactory(ProcessorFactory):
    def get_processor_name(self) -> str:
        return "excel_processor"
    
    def create(self, config: Dict[str, Any], logger: logging.Logger) -> ExcelProcessor:
        return ExcelProcessor(**config, logger=logger)

# Register the factory
FactoryRegistry.register_factory(ExcelProcessorFactory())
```

## Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Registry** | Simple, explicit, easy to debug | Requires manual registration | Small to medium projects |
| **Auto-Discovery** | Zero configuration, truly pluggable | More complex, harder to debug | Large projects with many plugins |
| **Factory Pattern** | Clean separation, testable | More boilerplate code | Enterprise applications |

## Recommended Approach for Your Project

Given your refactoring goals, I recommend **Option 1 (Registry)** because:

1. **Explicit is better than implicit** - You know exactly what processors are available
2. **Easy to test** - You can mock the registry for testing
3. **Simple to understand** - New developers can easily see how to add processors
4. **No magic** - No auto-discovery surprises
5. **Flexible** - Can easily add validation, versioning, etc.

## Integration with Current Code

Here's how the truly pluggable ProcessorManager integrates with your current strategies:

```python
# Strategy declares what it needs
class PDFSegmentationStrategy(SegmentationStrategy):
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {
            'pdf_processor': {
                'extract_images': self.config.get('extract_images', True),
                'extract_text': self.config.get('extract_text', True),
                # ... config from config.json
            }
        }
    
    async def segment(self, document_path: Path, pipeline_context: Any, 
                     metadata: Optional[Dict] = None) -> Dict[str, Any]:
        # Access processor dynamically
        pdf_processor = pipeline_context.get_processor('pdf_processor')
        if not pdf_processor:
            raise ValueError("PDF processor not available")
        
        return pdf_processor.process(document_path)
```

## Summary

The key insight is that ProcessorManager should:
1. **NOT have hardcoded processor attributes**
2. **NOT have hardcoded initialization methods**
3. **Store processors in a dynamic dictionary**
4. **Use a registry or factory pattern for creation**
5. **Provide dynamic access to processors**

This makes the system truly pluggable - adding a new processor type requires:
1. Creating the processor class
2. Registering it (explicitly or via auto-discovery)
3. NO changes to ProcessorManager itself!