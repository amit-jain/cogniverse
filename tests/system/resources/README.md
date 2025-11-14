# System Test Resources

**Last Updated:** 2025-11-13

This directory contains resources for system integration tests across the 10-package layered architecture, organized like Java's test resources folder. Resources support multi-modal testing (video, audio, images, documents, text, dataframes) and multi-tenant schema validation:

## Structure

- **videos/**: Small test video files for ingestion testing
- **configs/**: Test-specific configuration files  
- **schemas/**: Vespa schema definitions used in tests

## Usage

System tests use these resources for:
- Testing real video ingestion pipelines
- Validating search functionality with known data
- Integration testing with isolated Vespa instances

## Videos

- `v_-6dz6tBH77I.mp4`: Small test video (1.3MB)  
- `v_-D1gdv_gQyw.mp4`: Medium test video (5.5MB)

## Configuration

- `system_test_config.json`: Configuration for isolated test Vespa instance (port 8081)

## Schemas

All Vespa schema definitions for video processing profiles:

- `video_colpali_smol500_mv_frame_schema.json`: ColPali multi-vector frame-based schema
- `video_colqwen_omni_mv_chunk_30s_schema.json`: ColQwen multi-vector 30s chunk schema
- `video_videoprism_base_mv_chunk_30s_schema.json`: VideoPrism base 30s chunk schema  
- `video_videoprism_large_mv_chunk_30s_schema.json`: VideoPrism large 30s chunk schema
- `video_videoprism_lvt_base_sv_chunk_6s_schema.json`: VideoPrism LVT base 6s chunk schema
- `video_videoprism_lvt_large_sv_chunk_6s_schema.json`: VideoPrism LVT large 6s chunk schema
- `ranking_strategies.json`: Ranking strategy definitions for all schemas