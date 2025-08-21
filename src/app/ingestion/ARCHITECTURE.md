# Video Ingestion Pipeline Architecture

## Overview

The video ingestion pipeline has been refactored into a flexible, strategy-driven architecture that supports multiple video processing models and backends through a unified interface.

## Key Components

### 1. Pipeline Core (`pipeline.py`)
- **VideoIngestionPipeline**: Main orchestrator with async processing
- **PipelineConfig**: Configuration for processing steps
- **Async Processing**: Concurrent video processing with configurable limits

### 2. Strategy Pattern (`processing_strategy_set.py`, `strategies.py`)
- **ProcessingStrategySet**: Orchestrates multiple strategies in sequence
- **Strategy Types**:
  - `FrameSegmentationStrategy`: Extract keyframes from videos
  - `ChunkSegmentationStrategy`: Extract video chunks
  - `SingleVectorSegmentationStrategy`: Process entire videos as single vectors
  - `AudioTranscriptionStrategy`: Transcribe audio tracks
  - `VLMDescriptionStrategy`: Generate visual descriptions
  - `MultiVectorEmbeddingStrategy`: Generate embeddings per frame/chunk
  - `SingleVectorEmbeddingStrategy`: Generate single embedding per video

### 3. Processor Manager (`processor_manager.py`)
- **Auto-Discovery**: Automatically discovers and initializes processors
- **Pluggable Architecture**: Processors can be added without code changes
- **Requirement-Based**: Strategies declare processor requirements, manager provides them

### 4. Document System (`src/common/document.py`)
- **Generic Document**: Universal document structure for all content types
- **ContentType**: Enum for VIDEO, AUDIO, IMAGE, TEXT, DATAFRAME, DOCUMENT
- **ProcessingStatus**: Tracks document processing state
- **Flexible Metadata**: Content-specific fields stored in metadata dict
- **Embedding Storage**: Multiple embeddings per document with metadata

### 5. Embedding Generation (`processors/embedding_generator/`)
- **EmbeddingGeneratorImpl**: Unified generator for all model types
- **Model Support**: ColPali, ColQwen, VideoPrism (base, large, LVT variants)
- **Thread-Safe**: Proper PyTorch model loading on main thread
- **Backend Integration**: Direct feeding to Vespa/other backends

## Processing Flow

```
Video File
    ↓
1. Strategy Selection (based on profile config)
    ↓
2. Segmentation Strategy
    ├── Frame-based: Extract keyframes
    ├── Chunk-based: Extract video segments
    └── Single-vector: Process entire video
    ↓
3. Transcription Strategy (parallel)
    └── Extract audio transcript
    ↓
4. Description Strategy (parallel)
    └── Generate visual descriptions
    ↓
5. Embedding Strategy
    ├── Multi-vector: One embedding per frame/chunk
    └── Single-vector: One embedding per video
    ↓
6. Document Creation
    └── Generic Document with metadata
    ↓
7. Backend Storage
    └── Feed to Vespa/other search backends
```

## Configuration Profiles

The system supports multiple processing profiles defined in config:

- `video_colpali_smol500_mv_frame`: ColPali model, multi-vector frame-based
- `video_colqwen_omni_mv_chunk_30s`: ColQwen model, 30-second chunks
- `video_videoprism_base_mv_chunk_30s`: VideoPrism base, 30-second chunks
- `video_videoprism_large_mv_chunk_30s`: VideoPrism large, 30-second chunks
- `video_videoprism_lvt_base_sv_chunk_6s`: VideoPrism LVT base, single-vector 6s chunks
- `video_videoprism_lvt_large_sv_chunk_6s`: VideoPrism LVT large, single-vector 6s chunks

## Key Design Principles

### 1. Strategy Pattern
- **Flexible**: Easy to add new processing modes
- **Composable**: Mix and match strategies for different use cases
- **Testable**: Each strategy can be tested independently

### 2. Generic Document
- **Universal**: Single document type for all content
- **Extensible**: Metadata system allows content-specific fields
- **Backend-Agnostic**: Converts to any backend format

### 3. Async Architecture
- **Non-blocking**: Async methods with proper thread handling
- **Concurrent**: Process multiple videos simultaneously
- **Efficient**: Cooperative multitasking for I/O operations

### 4. Pluggable Processors
- **Auto-Discovery**: Processors found automatically
- **Loose Coupling**: Strategies declare requirements, don't instantiate directly
- **Scalable**: Add new processors without changing core code

## Testing

The architecture supports comprehensive testing:

```bash
# Test single profile
uv run python scripts/run_ingestion.py --profile video_colpali_smol500_mv_frame

# Test all profiles
uv run python scripts/run_ingestion.py --profile video_colpali_smol500_mv_frame video_colqwen_omni_mv_chunk_30s video_videoprism_base_mv_chunk_30s video_videoprism_large_mv_chunk_30s video_videoprism_lvt_base_sv_chunk_6s video_videoprism_lvt_large_sv_chunk_6s
```

## Performance

- **Concurrent Processing**: 2+ videos processed simultaneously
- **Async I/O**: Non-blocking file and network operations
- **Model Efficiency**: Proper PyTorch thread handling
- **Cache Support**: Built-in caching for expensive operations

## Extensibility

Adding new functionality:

1. **New Model**: Add to `src/common/models/` with loader
2. **New Strategy**: Inherit from `BaseStrategy` in `strategies.py`
3. **New Processor**: Add to `processors/` with auto-discovery
4. **New Backend**: Implement backend interface

### Example: Adding PDF Processing Profile

Here's how you would add PDF document processing to demonstrate the architecture's flexibility:

#### 1. Create PDF Processor (`processors/pdf_processor.py`)
```python
class PDFProcessor:
    """Extract text and images from PDF documents."""
    
    def extract_pages(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extract pages as images and text."""
        # Use PyMuPDF or similar to extract pages
        return {
            "pages": [
                {"page_num": 1, "image_path": "page_1.png", "text": "Page 1 content"},
                {"page_num": 2, "image_path": "page_2.png", "text": "Page 2 content"}
            ],
            "metadata": {"total_pages": 2, "title": "Document Title"}
        }
```

#### 2. Create PDF Segmentation Strategy (`strategies.py`)
```python
class PDFPageSegmentationStrategy(BaseStrategy):
    """Segment PDF by pages."""
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {"pdf": {"type": "pdf_extraction"}}
    
    async def segment(self, file_path: Path, pipeline_context: Any, 
                     transcript_data: Any = None) -> Dict[str, Any]:
        processor = pipeline_context.processor_manager.get_processor('pdf')
        pages = processor.extract_pages(file_path, pipeline_context.profile_output_dir)
        return {"pages": pages}
```

#### 3. Create PDF Embedding Strategy (`strategies.py`)
```python
class PDFEmbeddingStrategy(BaseStrategy):
    """Generate embeddings for PDF pages using vision models."""
    
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        return {"embedding": {"model_name": "colpali", "type": "vision"}}
    
    async def generate_embeddings_with_processor(self, results: Dict[str, Any],
                                               pipeline_context: Any,
                                               processor_manager: Any) -> Dict[str, Any]:
        # Process each page image with ColPali or similar vision model
        pages = results.get("pages", {}).get("pages", [])
        documents = []
        
        for page in pages:
            # Create Document with PDF-specific metadata
            doc = Document(
                id=f"{pipeline_context.file_path.stem}_page_{page['page_num']}",
                content_type=ContentType.DOCUMENT,
                content_id=pipeline_context.file_path.stem,
                text_content=page["text"],
                status=ProcessingStatus.COMPLETED
            )
            
            # Add PDF-specific metadata
            doc.add_metadata("page_number", page["page_num"])
            doc.add_metadata("document_title", results.get("metadata", {}).get("title"))
            doc.add_metadata("total_pages", results.get("metadata", {}).get("total_pages"))
            
            # Generate embeddings from page image
            if page.get("image_path"):
                embeddings = await self._generate_page_embeddings(page["image_path"])
                doc.add_embedding("page_embedding", embeddings, {"type": "vision", "model": "colpali"})
            
            documents.append(doc)
        
        # Feed to backend
        return {"documents_fed": len(documents), "total_documents": len(pages)}
```

#### 4. Add Configuration Profile
```yaml
# configs/config.yaml
document_processing_profiles:
  pdf_colpali_page_based:
    schema_name: "pdf_colpali_page_schema"
    embedding_model: "vidore/colsmol-500m"
    processing_strategy: "document_page_based"
    storage_mode: "multi_doc"
    strategies:
      segmentation:
        type: "PDFPageSegmentationStrategy"
      embedding:
        type: "PDFEmbeddingStrategy"
        model_name: "vidore/colsmol-500m"
    pipeline_config:
      extract_pages: true
      generate_embeddings: true
      max_pages_per_document: 50
```

#### 5. Usage
```bash
# Process PDF documents
uv run python scripts/run_ingestion.py \
  --document_dir data/pdfs \
  --backend vespa \
  --profile pdf_colpali_page_based
```

#### 6. Document Structure
The same generic Document class handles PDFs:
```python
Document(
    id="report_2024_page_1",
    content_type=ContentType.DOCUMENT,
    content_id="report_2024",
    text_content="Executive Summary: This report analyzes...",
    embeddings={
        "page_embedding": {
            "data": [0.1, 0.2, ...],  # ColPali embeddings
            "metadata": {"type": "vision", "model": "colpali"}
        }
    },
    metadata={
        "page_number": 1,
        "total_pages": 25,
        "document_title": "Annual Report 2024",
        "file_type": "pdf"
    }
)
```

This example shows how the **same architecture pattern** that handles videos seamlessly extends to PDFs, demonstrating the power of the generic Document system and strategy-based processing.

The architecture is designed for easy extension without breaking existing functionality.