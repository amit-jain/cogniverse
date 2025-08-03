"""
Mapping between Vespa schemas and video processing profiles
"""

SCHEMA_TO_PROFILE = {
    "video_frame": "frame_based_colpali",
    "video_colqwen": "direct_video_colqwen", 
    "video_videoprism_base": "direct_video_frame",
    "video_videoprism_large": "direct_video_frame_large",
    "video_videoprism_global": "direct_video_global",
    "video_videoprism_global_large": "direct_video_global_large",
    "video_chunks": "single__video_videoprism_large_6s"
}

def get_profile_for_schema(schema_name: str) -> str:
    """Get the video processing profile for a given schema"""
    if schema_name not in SCHEMA_TO_PROFILE:
        raise ValueError(f"Unknown schema: {schema_name}")
    return SCHEMA_TO_PROFILE[schema_name]

def get_supported_schemas() -> list:
    """Get list of supported schemas"""
    return list(SCHEMA_TO_PROFILE.keys())