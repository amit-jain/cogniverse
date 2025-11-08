#!/usr/bin/env python3
"""
Comprehensive Video Query Test v2 - Enhanced with dataset ground truth

Tests video search functionality using two complementary query sets:

1. **Ground Truth QA Queries**: Official Q&A pairs from the dataset's human annotations
   - Sourced from data/testset/queries/sample_*_qa.json
   - Categories: temporal, action, object_detection, scene_description, etc.
   - Provides ground truth answers for evaluation

2. **Visual Content Queries**: Targeted visual similarity queries
   - Based on actual video content analysis
   - Categories: action, scene, object, person, lighting, etc.
   - Tests visual understanding and retrieval

**Sample Videos (10 total):**
- v_-6dz6tBH77I: Discus throwing athletics
- v_-D1gdv_gQyw: Outdoor camping/fishing in forest  
- v_-HpCLXdtcas: Weightlifting in gym
- v_-IMXSEIabMM: Snow shoveling safety video
- v_-MbZ-W0AbN0: Furniture polish product
- v_-cAcA8dO7kA: Desert motorbike scene
- v_-nl4G-00PtA: Kitchen cooking scene
- v_-pkfcMUIEMo: Snow shoveling demonstration
- v_-uJnucdW6DY: Nighttime baseball game
- v_-vnSFKJNB94: Dark dramatic scene

Comprehensive evaluation with MRR, Recall@k, and NDCG@k metrics.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))
from cogniverse_agents.search.service import SearchService
from cogniverse_core.config.utils import create_default_config_manager, get_config

# Enhanced ground truth queries combining frame descriptions, transcripts, and human annotations
GROUND_TRUTH_QA_QUERIES = [
    # Discus throwing (v_-6dz6tBH77I) - enhanced with visual descriptions
    {
        "query": "What direction did the athlete in white tank top look after throwing the discus in the safety net?",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "temporal_visual",
        "ground_truth": "The man looked towards the direction where he threw the disk.",
        "source": "enhanced_multimodal",
        "visual_details": "athlete in white tank top and dark shorts, safety net, spectators in bleachers, outdoor athletic field"
    },
    {
        "query": "What sporting equipment is the muscular athlete holding while standing behind the protective net?",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "object_detection_contextual",
        "ground_truth": "The man was holding a disk in his right hand.",
        "source": "enhanced_multimodal",
        "visual_details": "discus throwing event, protective netting, spectators watching, outdoor sports setting"
    },
    
    # Snow shoveling safety (v_-IMXSEIabMM) - from sample_generic_qa.json
    {
        "query": "What are the people doing behind the car at the beginning of the video?",
        "expected_videos": ["v_-IMXSEIabMM"],
        "category": "action",
        "ground_truth": "The people are shoveling snow from around the car.",
        "source": "dataset_qa"
    },
    {
        "query": "What is the setting of the snow shoveling safety video?",
        "expected_videos": ["v_-IMXSEIabMM"],
        "category": "scene_description",
        "ground_truth": "The setting of the video is during the daytime outside a house with a red brick facade.",
        "source": "dataset_qa"
    },
    
    # Weightlifting (v_-HpCLXdtcas) - from sample_generic_qa.json
    {
        "query": "What is the weightlifter doing in the gym and what is he wearing?",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "action_and_appearance",
        "ground_truth": "The man is lifting a barbell over his head. He is wearing a black polo shirt and shorts.",
        "source": "dataset_qa"
    },
    {
        "query": "What color are the weights on the barbell?",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "object_details",
        "ground_truth": "The barbell in front of the man has multi-colored weights on it.",
        "source": "dataset_qa"
    },
    
    # Snow shoveling demo (v_-pkfcMUIEMo) - enhanced but practical for visual search
    {
        "query": "What safety technique is being demonstrated with the snow shovel by someone speaking to camera?",
        "expected_videos": ["v_-pkfcMUIEMo"],
        "category": "instructional_video",
        "ground_truth": "The man is demonstrating how to shovel snow using proper technique.",
        "source": "enhanced_multimodal",
        "visual_details": "person speaking directly to camera while demonstrating snow shoveling technique"
    },
    {
        "query": "What winter activity is being taught by someone in winter clothing outdoors?",
        "expected_videos": ["v_-pkfcMUIEMo"],
        "category": "educational_content",
        "ground_truth": "Snow shoveling technique demonstration",
        "source": "enhanced_multimodal", 
        "visual_details": "instructional video showing proper snow removal technique with winter clothing"
    },
    
    # Additional sample videos from dataset
    {
        "query": "What happens after the biker rides towards the middle of the dirt field?",
        "expected_videos": ["v_-cAcA8dO7kA"],
        "category": "temporal",
        "ground_truth": "The biker tries to jump over a slight elevation but lands with the front wheel first and crashes.",
        "source": "dataset_qa"
    },
    {
        "query": "What is the man in yellow shirt lighting on fire in the forest?",
        "expected_videos": ["v_-D1gdv_gQyw"],
        "category": "action",
        "ground_truth": "The man is lighting a stack of firewood on fire using a knife and a fire starter.",
        "source": "dataset_qa"
    },
    {
        "query": "What is the shirtless man doing at the kitchen sink?",
        "expected_videos": ["v_-nl4G-00PtA"],
        "category": "action",
        "ground_truth": "The man is washing dishes in a kitchen sink.",
        "source": "dataset_qa"
    },
    {
        "query": "Who is shown standing in front of the platform in the video and what are they holding?",
        "expected_videos": ["v_-MbZ-W0AbN0"],
        "category": "person_and_object",
        "ground_truth": "The video shows a man wearing a white lab coat standing in front of a platform and holding a can of furniture polish.",
        "source": "dataset_qa"
    },
    {
        "query": "What types of diving maneuvers does Michal Navratil perform from the diving board?",
        "expected_videos": ["v_-vnSFKJNB94"],
        "category": "action_details",
        "ground_truth": "The man performs dives using tucks, twists, forwards dives, and other maneuvers from the diving board.",
        "source": "dataset_qa"
    },
    {
        "query": "What are the kids doing in the nighttime baseball field?",
        "expected_videos": ["v_-uJnucdW6DY"],
        "category": "action",
        "ground_truth": "The kids in the video are playing with a ball in a grassy field.",
        "source": "dataset_qa"
    },
    
    # Additional multimodal queries testing visual + context understanding
    {
        "query": "Which video shows someone giving instructions while demonstrating a winter outdoor activity?",
        "expected_videos": ["v_-pkfcMUIEMo"],
        "category": "instructional_demonstration",
        "ground_truth": "Snow shoveling safety demonstration with spoken instructions.",
        "source": "enhanced_multimodal",
        "details": "combines instructional speaking with physical demonstration"
    },
    {
        "query": "What type of professional setting is shown with someone in a white lab coat holding a product?",
        "expected_videos": ["v_-MbZ-W0AbN0"],
        "category": "professional_context",
        "ground_truth": "A laboratory or demonstration setting with someone holding furniture polish.",
        "source": "enhanced_multimodal",
        "details": "professional product demonstration environment"
    }
]

# Curated sample of 10 representative queries for quick testing (deterministic)
SAMPLE_TEST_QUERIES = [
    # 1. Discus throwing - temporal visual
    {
        "query": "What direction did the athlete in white tank top look after throwing the discus in the safety net?",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "temporal_visual",
        "source": "sample_curated"
    },
    # 2. Snow shoveling - instructional 
    {
        "query": "What safety technique is being demonstrated with the snow shovel by someone speaking to camera?",
        "expected_videos": ["v_-pkfcMUIEMo"],
        "category": "instructional_video",
        "source": "sample_curated"
    },
    # 3. Weightlifting - object detection
    {
        "query": "What color are the weights on the barbell?",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "object_details",
        "source": "sample_curated"
    },
    # 4. Kitchen - action
    {
        "query": "What is the shirtless man doing at the kitchen sink?",
        "expected_videos": ["v_-nl4G-00PtA"],
        "category": "action",
        "source": "sample_curated"
    },
    # 5. Product demo - professional context
    {
        "query": "What type of professional setting is shown with someone in a white lab coat holding a product?",
        "expected_videos": ["v_-MbZ-W0AbN0"],
        "category": "professional_context",
        "source": "sample_curated"
    },
    # 6. Visual similarity - winter scene
    {
        "query": "person wearing winter clothes outdoors shoveling snow",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "action",
        "source": "sample_curated"
    },
    # 7. Sports context - scene description
    {
        "query": "athletic field with spectators in bleachers",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "scene",
        "source": "sample_curated"
    },
    # 8. Cross-cutting - multiple videos
    {
        "query": "man performing work or exercise",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo", "v_-HpCLXdtcas", "v_-D1gdv_gQyw", "v_-nl4G-00PtA"],
        "category": "action",
        "source": "sample_curated"
    },
    # 9. Specific activity - motorbike
    {
        "query": "What happens after the biker rides towards the middle of the dirt field?",
        "expected_videos": ["v_-cAcA8dO7kA"],
        "category": "temporal",
        "source": "sample_curated"
    },
    # 10. Lighting condition - indoor/outdoor contrast
    {
        "query": "indoor scene with artificial lighting",
        "expected_videos": ["v_-HpCLXdtcas", "v_-nl4G-00PtA", "v_-vnSFKJNB94"],
        "category": "lighting",
        "source": "sample_curated"
    }
]

# Visual queries based on actual videos in sample_videos directory
VISUAL_TEST_QUERIES = [
    # Snow/Winter scenes (v_-IMXSEIabMM, v_-pkfcMUIEMo)
    {
        "query": "person wearing winter clothes outdoors shoveling snow",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "action"
    },
    {
        "query": "white snow on ground with stone wall",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "scene"
    },
    {
        "query": "person holding snow shovel with yellow handle",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "object"
    },
    {
        "query": "man in green cap and beige jacket",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "person"
    },
    
    # Discus throwing athletics (v_-6dz6tBH77I)
    {
        "query": "person throwing discus in athletic field",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "action"
    },
    {
        "query": "athletic field with spectators in bleachers",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "scene"
    },
    {
        "query": "outdoor sports event with safety net",
        "expected_videos": ["v_-6dz6tBH77I"],
        "category": "scene"
    },
    
    # Weightlifting in gym (v_-HpCLXdtcas)
    {
        "query": "man lifting barbell with colored weight plates",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "action"
    },
    {
        "query": "gym interior with weightlifting equipment",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "scene"
    },
    {
        "query": "colorful weight plates red green yellow",
        "expected_videos": ["v_-HpCLXdtcas"],
        "category": "object"
    },
    
    # Outdoor camping/fishing (v_-D1gdv_gQyw)
    {
        "query": "man in yellow shirt outdoors in forest",
        "expected_videos": ["v_-D1gdv_gQyw"],
        "category": "person"
    },
    {
        "query": "red folding chair in outdoor setting",
        "expected_videos": ["v_-D1gdv_gQyw"],
        "category": "object"
    },
    {
        "query": "forest camping scene with trees",
        "expected_videos": ["v_-D1gdv_gQyw"],
        "category": "scene"
    },
    
    # Product/furniture polish (v_-MbZ-W0AbN0)
    {
        "query": "white plastic bottle with label",
        "expected_videos": ["v_-MbZ-W0AbN0"],
        "category": "object"
    },
    {
        "query": "furniture polish cleaning product",
        "expected_videos": ["v_-MbZ-W0AbN0"],
        "category": "product"
    },
    
    # Desert motorbike scene (v_-cAcA8dO7kA)
    {
        "query": "person riding motorbike in desert landscape",
        "expected_videos": ["v_-cAcA8dO7kA"],
        "category": "action"
    },
    {
        "query": "sandy desert terrain with shrubs",
        "expected_videos": ["v_-cAcA8dO7kA"],
        "category": "scene"
    },
    
    # Kitchen cooking scene (v_-nl4G-00PtA)
    {
        "query": "person working in kitchen at sink",
        "expected_videos": ["v_-nl4G-00PtA"],
        "category": "action"
    },
    {
        "query": "kitchen interior with stainless steel sink",
        "expected_videos": ["v_-nl4G-00PtA"],
        "category": "scene"
    },
    
    # Nighttime baseball (v_-uJnucdW6DY)
    {
        "query": "nighttime baseball game on field",
        "expected_videos": ["v_-uJnucdW6DY"],
        "category": "action"
    },
    {
        "query": "baseball field with artificial lighting",
        "expected_videos": ["v_-uJnucdW6DY"],
        "category": "scene"
    },
    
    # Dark dramatic scene (v_-vnSFKJNB94)
    {
        "query": "shirtless man in dark dramatic setting",
        "expected_videos": ["v_-vnSFKJNB94"],
        "category": "person"
    },
    {
        "query": "dark indoor scene with minimal lighting",
        "expected_videos": ["v_-vnSFKJNB94"],
        "category": "lighting"
    },
    
    # Cross-cutting queries for multiple videos
    {
        "query": "person engaged in physical outdoor activity",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo", "v_-6dz6tBH77I", "v_-D1gdv_gQyw", "v_-cAcA8dO7kA", "v_-uJnucdW6DY"],
        "category": "action"
    },
    {
        "query": "daytime outdoor scene with natural lighting",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo", "v_-6dz6tBH77I", "v_-D1gdv_gQyw", "v_-cAcA8dO7kA"],
        "category": "lighting"
    },
    {
        "query": "man performing work or exercise",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo", "v_-HpCLXdtcas", "v_-D1gdv_gQyw", "v_-nl4G-00PtA"],
        "category": "action"
    },
    {
        "query": "sports or athletic activity",
        "expected_videos": ["v_-6dz6tBH77I", "v_-HpCLXdtcas", "v_-uJnucdW6DY"],
        "category": "action"
    },
    {
        "query": "indoor scene with artificial lighting",
        "expected_videos": ["v_-HpCLXdtcas", "v_-nl4G-00PtA", "v_-vnSFKJNB94"],
        "category": "lighting"
    }
]


def calculate_metrics(results: List[Dict], expected_videos: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """Calculate MRR, NDCG, and Recall@k metrics"""
    metrics = {}
    
    # Initialize all metrics to 0
    metrics['mrr'] = 0.0
    for k in k_values:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    
    # Get video IDs from results
    result_videos = [r.get('video_id', '') for r in results]
    
    # If no results, return zero metrics
    if not result_videos:
        return metrics
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, video in enumerate(result_videos):
        if video in expected_videos:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = mrr
    
    # Recall@k
    for k in k_values:
        if k <= len(result_videos):
            recall_k = len(set(result_videos[:k]) & set(expected_videos)) / len(expected_videos) if expected_videos else 0
            metrics[f'recall@{k}'] = recall_k
    
    # NDCG@k (Normalized Discounted Cumulative Gain)
    for k in k_values:
        if k <= len(result_videos):
            # Binary relevance: 1 if video is expected, 0 otherwise
            relevances = [1 if vid in expected_videos else 0 for vid in result_videos[:k]]
            
            # DCG
            dcg = relevances[0] if relevances else 0
            for i in range(1, len(relevances)):
                dcg += relevances[i] / np.log2(i + 2)
            
            # Ideal DCG (all relevant items at top)
            ideal_relevances = [1] * min(len(expected_videos), k) + [0] * max(0, k - len(expected_videos))
            idcg = ideal_relevances[0] if ideal_relevances else 0
            for i in range(1, len(ideal_relevances)):
                idcg += ideal_relevances[i] / np.log2(i + 2)
            
            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg@{k}'] = ndcg
    
    return metrics


def test_profile_with_queries(profile: str, queries: List[Dict], test_multiple_strategies: bool = False) -> Dict:
    """Test a profile with all queries and return detailed results"""

    # Get config and create search service
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    config = get_config(tenant_id="default", config_manager=config_manager)

    try:
        search_service = SearchService(config, profile, config_manager=config_manager, schema_loader=schema_loader)
    except Exception as e:
        print(f"❌ Failed to create search service for {profile}: {e}")
        return {
            'profile': profile,
            'error': str(e)
        }
    
    # Determine strategies to test
    if test_multiple_strategies:
        if profile == 'video_colpali_smol500_mv_frame':
            strategies_to_test = [
                ('binary_binary', 'Visual Only'),
                ('hybrid_binary_bm25_no_description', 'Hybrid No Desc'),
                ('hybrid_binary_bm25', 'Hybrid + Desc'),
                ('bm25_only', 'Text Only')
            ]
        elif 'global' in profile:
            strategies_to_test = [
                ('binary_binary', 'Binary Visual'),
                ('float_binary', 'Float-Binary Hybrid'),
                ('phased', 'Phased (Binary→Float)'),
                ('float_float', 'Float Visual')
            ]
        else:
            strategies_to_test = [
                ('float_float', 'Float Visual'),
                ('float_binary', 'Float-Binary Hybrid'),
                ('binary_binary', 'Binary Visual'),
                ('phased', 'Phased (Binary→Float)')
            ]
    else:
        # Use default strategy for profile
        strategies_to_test = [(None, 'Default')]

    # Get model name from config (reuse config_manager from above)
    profiles = config.get("video_processing_profiles", {})
    model_name = profiles.get(profile, {}).get("embedding_model", "Unknown")
    
    # Results storage
    profile_results = {
        'profile': profile,
        'model': model_name,
        'strategies': {} if test_multiple_strategies else None,
        'queries': [],
        'aggregate_metrics': {}
    }
    
    for strategy, strategy_name in strategies_to_test:
        strategy_results = {
            'name': strategy_name,
            'queries': [],
            'aggregate_metrics': {}
        }
        
        all_metrics = []
        
        for query_data in queries:
            query = query_data['query']
            expected_videos = query_data['expected_videos']
            
            try:
                # Execute search using SearchService with optional ranking strategy and tenant_id
                tenant_id = f"test-tenant-{datetime.now().strftime('%Y%m%d')}"
                search_results = search_service.search(query, top_k=10, ranking_strategy=strategy, tenant_id=tenant_id)
                
                # Convert SearchResult objects to dicts and extract video IDs
                result_dicts = [r.to_dict() for r in search_results]
                results = []
                for r in result_dicts:
                    video_id = r.get('source_id', r['document_id'].split('_')[0])
                    results.append({
                        'video_id': video_id,
                        'score': r['score'],
                        'document_id': r['document_id']
                    })
                
                # Calculate metrics
                metrics = calculate_metrics(results, expected_videos)
                all_metrics.append(metrics)
                
                # Store query results
                query_result = {
                    'query': query,
                    'expected': expected_videos,
                    'results': [r['video_id'] for r in results[:5]],  # Top 5
                    'metrics': metrics,
                    'top_result_correct': results[0]['video_id'] in expected_videos if results else False
                }
                strategy_results['queries'].append(query_result)
                
            except Exception as e:
                error_msg = str(e)
                # Truncate long embedding arrays in error messages
                if 'query(qtb)' in error_msg and len(error_msg) > 500:
                    # Find the embedding array and truncate it
                    start_idx = error_msg.find('[')
                    end_idx = error_msg.find(']', start_idx)
                    if start_idx != -1 and end_idx != -1 and end_idx - start_idx > 100:
                        embedding_preview = error_msg[start_idx:start_idx+50] + "... (truncated) ..." + error_msg[end_idx-20:end_idx+1]
                        error_msg = error_msg[:start_idx] + embedding_preview + error_msg[end_idx+1:]
                
                print(f"❌ Search failed for query '{query}' with strategy '{strategy}': {error_msg}")
                query_result = {
                    'query': query,
                    'expected': expected_videos,
                    'results': [],
                    'metrics': {'recall@1': 0, 'recall@5': 0, 'recall@10': 0, 'mrr': 0},
                    'top_result_correct': False,
                    'error': error_msg
                }
                strategy_results['queries'].append(query_result)
        
        # Calculate aggregate metrics for this strategy
        if all_metrics:
            # Get all unique metric names from all results
            all_metric_names = set()
            for m in all_metrics:
                all_metric_names.update(m.keys())
            
            for metric_name in all_metric_names:
                # Only calculate if at least one result has this metric
                values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
                if values:
                    strategy_results['aggregate_metrics'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        if test_multiple_strategies:
            profile_results['strategies'][strategy or 'default'] = strategy_results
        else:
            # For single strategy, put results at top level
            profile_results['queries'] = strategy_results['queries']
            profile_results['aggregate_metrics'] = strategy_results['aggregate_metrics']
    
    # For backward compatibility when testing multiple strategies
    if test_multiple_strategies and profile_results['strategies']:
        # Use the first strategy as default for top-level results
        first_strategy = list(profile_results['strategies'].values())[0]
        profile_results['queries'] = first_strategy['queries']
        profile_results['aggregate_metrics'] = first_strategy['aggregate_metrics']
    
    return profile_results


def get_best_strategy_for_profile(profile: str) -> str:
    """Get the best ranking strategy for a profile based on previous results"""
    best_strategies = {
        'video_colpali_smol500_mv_frame': 'binary_binary',  # Changed to visual-only for fair comparison
        'video_colqwen_omni_sv_chunk': 'hybrid_binary_bm25',
        'video_videoprism_lvt_base_sv_global': 'binary_binary',
        'video_videoprism_lvt_large_sv_global': 'binary_binary',
        'single__video_videoprism_large_6s': 'default'  # Use default for video_chunks
    }
    return best_strategies.get(profile, 'float_float')


def needs_text_for_strategy(strategy: str) -> bool:
    """Check if strategy needs text query"""
    return strategy in ['bm25_only', 'hybrid_float_bm25', 'hybrid_binary_bm25', 
                       'hybrid_bm25_float', 'hybrid_bm25_binary',
                       'hybrid_float_bm25_no_description', 'hybrid_binary_bm25_no_description',
                       'hybrid_bm25_float_no_description', 'hybrid_bm25_binary_no_description']


def needs_embeddings_for_strategy(strategy: str) -> bool:
    """Check if strategy needs embeddings"""
    return strategy not in ['bm25_only', 'bm25_no_description']


def create_comprehensive_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create the comprehensive table with all queries and results as requested"""
    
    rows = []
    
    for profile_results in all_results:
        profile = profile_results['profile']
        
        # Skip profiles that failed to initialize
        if 'error' in profile_results:
            continue
            
        for query_result in profile_results['queries']:
            query = query_result['query']
            expected = ', '.join(query_result['expected'])
            results = query_result['results']
            
            # Format results with checkmarks
            formatted_results = []
            for i, video in enumerate(results):
                if video in query_result['expected']:
                    formatted_results.append(f"✅ {video}")
                else:
                    formatted_results.append(f"❌ {video}")
            
            # Create result string
            result_str = ' > '.join(formatted_results[:3])  # Show top 3
            if len(results) > 3:
                result_str += ' ...'
            
            # Add metrics
            metrics = query_result['metrics']
            mrr = metrics.get('mrr', 0)
            recall_5 = metrics.get('recall@5', 0)
            
            rows.append({
                'Profile': profile,
                'Query': query[:50] + '...' if len(query) > 50 else query,
                'Expected': expected,
                'Results (Top 3)': result_str,
                'MRR': f"{mrr:.3f}",
                'R@5': f"{recall_5:.3f}"
            })
    
    return pd.DataFrame(rows)


def create_metrics_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create summary table of metrics across profiles"""
    
    rows = []
    
    for profile_results in all_results:
        profile = profile_results['profile']
        
        # Skip profiles that failed to initialize
        if 'error' in profile_results:
            rows.append({
                'Profile': profile,
                'Model': 'FAILED',
                'MRR': 'ERROR',
                'RECALL@1': 'ERROR',
                'RECALL@5': 'ERROR',
                'RECALL@10': 'ERROR',
                'NDCG@5': 'ERROR',
                'NDCG@10': 'ERROR'
            })
            continue
            
        metrics = profile_results.get('aggregate_metrics', {})
        
        row = {
            'Profile': profile,
            'Model': profile_results.get('model', 'N/A')[:30] + '...' if profile_results.get('model', '') and len(profile_results.get('model', '')) > 30 else profile_results.get('model', 'N/A')
        }
        
        # Add mean metrics
        for metric_name in ['mrr', 'recall@1', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
            if metric_name in metrics:
                row[metric_name.upper()] = f"{metrics[metric_name]['mean']:.3f}"
            else:
                row[metric_name.upper()] = "N/A"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive video query test v2")
    parser.add_argument("--profiles", nargs="+", 
                       default=["video_colpali_smol500_mv_frame", "video_colqwen_omni_sv_chunk", 
                               "video_videoprism_lvt_base_sv_global", "video_videoprism_lvt_large_sv_global",
                               "single__video_videoprism_large_6s"],
                       help="Profiles to test")
    parser.add_argument("--output-format", choices=["table", "html", "csv"], default="table",
                       help="Output format")
    parser.add_argument("--test-multiple-strategies", action="store_true",
                       help="Test multiple ranking strategies for each profile")
    parser.add_argument("--full-query-set", action="store_true",
                       help="Use full 44-query set instead of curated 10-query sample")
    
    args = parser.parse_args()
    
    if args.test_multiple_strategies:
        print("📊 Testing multiple ranking strategies for each profile...")
        print("    This will help identify the optimal strategy for each model\n")

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "outputs" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select query set based on arguments
    if args.full_query_set:
        ALL_TEST_QUERIES = GROUND_TRUTH_QA_QUERIES + VISUAL_TEST_QUERIES
        query_description = f"full set ({len(ALL_TEST_QUERIES)} queries)"
        detailed_breakdown = f"  - {len(GROUND_TRUTH_QA_QUERIES)} ground truth QA queries from dataset\n  - {len(VISUAL_TEST_QUERIES)} visual queries for content matching"
    else:
        ALL_TEST_QUERIES = SAMPLE_TEST_QUERIES
        query_description = f"curated sample ({len(ALL_TEST_QUERIES)} queries)"
        detailed_breakdown = "  - Deterministic sample covering all video types and query categories\n  - Use --full-query-set for complete 44-query evaluation"
    
    print("\n=== COMPREHENSIVE VIDEO QUERY TEST v2 ===")
    print(f"Testing {len(args.profiles)} profiles with {query_description}")
    print(detailed_breakdown)
    print("Metrics: MRR, Recall@k, NDCG@k\n")
    
    all_results = []
    
    # Test each profile
    for profile in args.profiles:
        print(f"\n🔍 Testing profile: {profile}")
        
        try:
            results = test_profile_with_queries(profile, ALL_TEST_QUERIES, 
                                               test_multiple_strategies=args.test_multiple_strategies)
            all_results.append(results)
            print(f"✅ Completed {profile}")
        except Exception as e:
            print(f"❌ Error testing {profile}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comprehensive results table
    print("\n" + "="*120)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*120)
    
    results_df = create_comprehensive_results_table(all_results)
    
    if args.output_format == "table":
        print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    elif args.output_format == "csv":
        csv_path = output_dir / f"comprehensive_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # Create metrics summary
    print("\n" + "="*120)
    print("METRICS SUMMARY")
    print("="*120)
    
    metrics_df = create_metrics_summary_table(all_results)
    print(tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False))
    
    # If testing multiple strategies, show comparison
    if args.test_multiple_strategies:
        print("\n" + "="*120)
        print("STRATEGY COMPARISON BY PROFILE")
        print("="*120)
        
        for result in all_results:
            if 'strategies' in result and len(result['strategies']) > 1:
                print(f"\n### {result['profile']} ({result['model']})")
                strategy_rows = []
                for strategy_key, strategy_data in result['strategies'].items():
                    metrics = strategy_data['aggregate_metrics']
                    strategy_rows.append({
                        'Strategy': strategy_data['name'],
                        'Ranking Profile': strategy_key,
                        'MRR': f"{metrics['mrr']['mean']:.3f}",
                        'Recall@1': f"{metrics.get('recall@1', {}).get('mean', 0):.3f}",
                        'Recall@5': f"{metrics.get('recall@5', {}).get('mean', 0):.3f}",
                        'NDCG@5': f"{metrics.get('ndcg@5', {}).get('mean', 0):.3f}"
                    })
                
                strategy_df = pd.DataFrame(strategy_rows)
                print(tabulate(strategy_df, headers='keys', tablefmt='grid', showindex=False))
                print()
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"comprehensive_v2_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'ground_truth_queries': GROUND_TRUTH_QA_QUERIES,
            'visual_queries': VISUAL_TEST_QUERIES,
            'all_queries': ALL_TEST_QUERIES,
            'results': all_results
        }, f, indent=2)
    print(f"\n📊 Detailed results saved to: {json_path}")


if __name__ == "__main__":
    main()
