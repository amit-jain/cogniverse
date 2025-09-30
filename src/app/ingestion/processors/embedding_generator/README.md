# Embedding Generator

This is a clean, modular implementation of the embedding generation pipeline with proper separation of concerns.

## Key Improvements

1. **Universal Document Structure**: All media types (video, image, text, audio) use the same `Document` class
2. **Backend Encapsulation**: Backends handle all format conversion and document preparation internally
3. **Clean Architecture**: Generator only deals with raw embeddings, backends handle everything else
4. **No Document Builders**: Backend creates its own document format internally
5. **No Exposed Processors**: Embedding format conversion is hidden inside backend

## Architecture

```
EmbeddingGeneratorImpl
    ↓ (generates raw numpy embeddings)
Document (media-agnostic structure)
    ↓
BackendClient (VespaPyClient)
    ↓ (internally uses)
    ├── VespaEmbeddingProcessor (format conversion)
    └── Document preparation logic
    ↓
Vespa
```

## Usage Example

```python
from embedding_generator import create_embedding_generator

# Configuration
config = {
    "backend": "vespa",
    "vespa_url": "http://localhost",
    "vespa_port": 8080,
    "schema_name": "video_frame"
}

# Create generator
generator = create_embedding_generator(config, logger)

# Process video
video_data = {
    "video_id": "test_video",
    "video_path": "/path/to/video.mp4",
    "duration": 120.0
}

result = generator.generate_embeddings(video_data, output_dir)
```

## Document Structure

The universal `Document` class supports all media types:

```python
# Video segment
doc = Document(
    media_type=MediaType.VIDEO,
    document_id="video123_segment_0",
    source_id="video123",
    raw_embeddings=numpy_array,
    temporal_info=TemporalInfo(start_time=0.0, end_time=30.0),
    segment_info=SegmentInfo(segment_idx=0, total_segments=4),
    metadata={"video_title": "Sample Video", "fps": 30}
)

# Image
doc = Document(
    media_type=MediaType.IMAGE,
    document_id="img_001",
    source_id="photo.jpg",
    raw_embeddings=image_embeddings,
    metadata={"width": 1920, "height": 1080, "caption": "A sunset"}
)

# Text
doc = Document(
    media_type=MediaType.TEXT,
    document_id="doc_page5_chunk2",
    source_id="document.pdf",
    raw_embeddings=text_embeddings,
    metadata={"page_number": 5, "chunk_idx": 2, "text_content": "..."}
)
```

## Backend Processing

The backend client handles all format-specific processing:

```python
class VespaPyClient(BackendClient):
    def process(self, doc: Document) -> Dict[str, Any]:
        # 1. Convert embeddings to Vespa format (hex bfloat16)
        processed = self._embedding_processor.process_embeddings(doc.raw_embeddings)
        
        # 2. Build Vespa document structure
        return {
            "put": f"id:video:{self.schema_name}::{doc.document_id}",
            "fields": {
                "document_id": doc.document_id,
                "source_id": doc.source_id,
                **processed,  # Converted embeddings
                # ... other fields based on media type
            }
        }
    
    def feed(self, documents: Union[Document, List[Document]], batch_size: int = 100):
        # Accepts single Document or list of Documents
        # Returns (success_count, failed_doc_ids)
```

## Key Benefits

1. **True Backend Abstraction**: Generator knows nothing about backend formats
2. **Extensible**: Easy to add new backends without changing generator code
3. **Clean Interfaces**: Clear separation between raw data and backend-specific formats
4. **Universal Structure**: Same Document class works for all media types
5. **Simplified Interface**: Single `feed()` method accepts both single and batch documents
6. **Clean Code**: No redundant methods or exposed internal processors

## Files

- `base_embedding_generator.py`: Base classes including Document structure
- `embedding_generator_impl.py`: Main generator implementation
- `backend_client.py`: Abstract backend interface
- `vespa_pyvespa_client.py`: Vespa backend implementation
- `vespa_embedding_processor.py`: Vespa-specific format conversion (internal to backend)
- `model_loaders.py`: Model loading logic
- `backend_factory.py`: Factory for creating backends
- `embedding_generator_factory.py`: Main factory for creating generators