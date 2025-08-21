"""
Utilities for handling video processing profiles
"""

def get_supported_ranking_strategies(profile: str) -> list:
    """
    Get the ranking strategies supported by each profile.
    
    ColPali (frame_based_colpali) supports all strategies because it has:
    - Text data (descriptions, transcripts) for BM25
    - Visual embeddings (binary and float) for visual search
    
    VideoPrism profiles have video_title for BM25 and support hybrid strategies.
    ColQwen only supports visual search strategies.
    """
    
    # All visual-only strategies (work with embeddings only)
    visual_strategies = [
        "float_float",
        "binary_binary", 
        "float_binary",
        "phased"
    ]
    
    if profile == "frame_based_colpali":
        # ColPali supports all strategies including text and hybrid
        return [
            # Text-only
            "bm25_only",
            "bm25_no_description",
            # Visual-only
            "float_float",
            "binary_binary", 
            "float_binary",
            "phased",
            # Hybrid (text + visual)
            "hybrid_float_bm25",
            "hybrid_binary_bm25",
            "hybrid_bm25_binary", 
            "hybrid_bm25_float",
            "hybrid_float_bm25_no_description",
            "hybrid_binary_bm25_no_description",
            "hybrid_bm25_binary_no_description",
            "hybrid_bm25_float_no_description"
        ]
    elif profile in ["direct_video_frame", "direct_video_frame_large"]:
        # VideoPrism profiles have video_title field and support BM25/hybrid strategies
        return [
            # Text-only
            "bm25_only",
            # Visual-only
            "float_float",
            "binary_binary", 
            "float_binary",
            "phased",
            # Hybrid (text + visual)
            "hybrid_float_bm25",
            "hybrid_binary_bm25",
            "hybrid_bm25_binary", 
            "hybrid_bm25_float"
        ]
    else:
        # ColQwen only supports visual search
        # It doesn't have text fields for BM25
        return visual_strategies