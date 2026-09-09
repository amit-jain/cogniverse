# System Test Resources

Fixed inputs for system and integration tests. Vespa schemas are not
mirrored here — tests load them from `configs/schemas/` through the
production `FilesystemSchemaLoader`, and `tests/common/unit/test_schema_corpus_single_source.py`
refuses a restated copy.

## videos/

- `v_-6dz6tBH77I.mp4`: small test video (1.3 MB)
- `v_-D1gdv_gQyw.mp4`: medium test video (5.5 MB)

## configs/

Test-specific configuration files for isolated Vespa instances.
