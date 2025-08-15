"""
Mapping between Vespa schemas and video processing profiles

With the new standardized naming, multiple profiles can share the same schema
since schemas only describe data structure, not the model.
"""

SCHEMA_TO_PROFILE = {
    # Multi-vector frame-based (ColPali)
    "video_mv_frame": "video_colpali_smol500_mv_frame",
    
    # Single-vector chunk arrays (ColQwen, VideoPrism chunks)
    "video_sv_chunk": [
        "video_colqwen_omni_sv_chunk",
        "video_videoprism_large_sv_chunk_6s"
    ],
    
    # Multi-vector chunk-based (VideoPrism base/large)
    "video_mv_chunk": [
        "video_videoprism_base_mv_chunk",
        "video_videoprism_large_mv_chunk"
    ],
    
    # Single-vector global (VideoPrism LVT)
    "video_sv_global": [
        "video_videoprism_base_sv_global",
        "video_videoprism_large_sv_global"
    ]
}

def get_profiles_for_schema(schema_name: str) -> list:
    """Get the video processing profiles for a given schema
    
    Returns a list since multiple profiles can share the same schema.
    """
    if schema_name not in SCHEMA_TO_PROFILE:
        raise ValueError(f"Unknown schema: {schema_name}")
    
    profiles = SCHEMA_TO_PROFILE[schema_name]
    if isinstance(profiles, str):
        return [profiles]
    return profiles

def get_profile_for_schema(schema_name: str) -> str:
    """Get the first video processing profile for a given schema
    
    For backwards compatibility - returns the first profile if multiple exist.
    """
    profiles = get_profiles_for_schema(schema_name)
    return profiles[0]

def get_supported_schemas() -> list:
    """Get list of supported schemas"""
    return list(SCHEMA_TO_PROFILE.keys())