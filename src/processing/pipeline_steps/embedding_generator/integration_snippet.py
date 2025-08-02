#!/usr/bin/env python3
"""
Simple integration snippet showing how to use the refactored embedding generator
in the existing ingestion pipeline.
"""

# This is what you would change in run_ingestion.py:

# OLD CODE (using raw HTTP requests):
"""
from src.processing.pipeline_steps.embedding_generator import EmbeddingGenerator

# In the process_videos function:
generator = EmbeddingGenerator(config, logger, args.backend)

# Process video
embeddings_data = generator.generate_embeddings(video_data, output_dir)
"""

# NEW CODE (cleaner with pyvespa):
"""
from src.processing.pipeline_steps.embedding_generator import create_embedding_generator

# In the process_videos function:
# Everything is determined from config:
# - Backend from config["embedding_backend"] 
# - Process type from active profile's "process_type"
# - Model from active profile's "embedding_model"
generator = create_embedding_generator(config, logger)

# Process video (exact same interface)
embeddings_data = generator.generate_embeddings(video_data, output_dir)
"""

# Example config structure:
"""
config = {
    "embedding_backend": "vespa",
    "active_profile": "direct_video_videoprism_base",
    "video_processing_profiles": {
        "direct_video_videoprism_base": {
            "process_type": "direct_video",     # How to process
            "embedding_model": "videoprism_base" # Which model
        }
    }
}
"""

# Or if you want more control:
"""
from src.processing.pipeline_steps.embedding_generator import EmbeddingGeneratorFactory

# Create generator with explicit control
generator = EmbeddingGeneratorFactory.create(
    backend="vespa",  # Could be "elasticsearch", "weaviate", etc
    config=config,
    logger=logger,
    embedding_type="direct_video_segment"
)
"""

# That's it! The interface is identical, but now you get:
# - Official pyvespa client with better performance
# - Proper connection pooling and retries
# - Cleaner, more maintainable code
# - Easy to extend with new backends in the future