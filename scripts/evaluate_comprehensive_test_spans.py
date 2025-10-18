#!/usr/bin/env python3
"""
Evaluate spans from comprehensive test runs

This script:
1. Retrieves spans from comprehensive test runs
2. Treats them as test queries for evaluation
3. Runs both reference-free and golden dataset evaluations
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluators.golden_dataset import create_low_scoring_golden_dataset
from src.evaluation.span_evaluator import SpanEvaluator

from tests.comprehensive_video_query_test_v2 import VISUAL_TEST_QUERIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTestEvaluator(SpanEvaluator):
    """Extended evaluator for comprehensive test spans"""
    
    def get_comprehensive_test_spans(self, hours: int = 24) -> pd.DataFrame:
        """
        Get spans from comprehensive test runs
        
        We'll identify them by looking for spans that match our test queries
        """
        # Try different operation names that might contain search results
        operation_names = [
            "search_service.search",
            "search_agent.search", 
            "video_search_agent.search",
            "AgentOrchestrator.process_message",
            None  # Try without filter
        ]
        
        all_spans = pd.DataFrame()
        for op_name in operation_names:
            logger.info(f"Looking for spans with operation_name='{op_name}'")
            spans = self.get_recent_spans(hours=hours, operation_name=op_name)
            if not spans.empty:
                logger.info(f"Found {len(spans)} spans for operation '{op_name}'")
                all_spans = pd.concat([all_spans, spans], ignore_index=True)
        
        if all_spans.empty:
            logger.info("No spans found, using mock comprehensive test data")
            return self._create_comprehensive_test_spans()
        
        # Remove duplicates
        all_spans = all_spans.drop_duplicates(subset=['span_id'])
        logger.info(f"Total unique spans found: {len(all_spans)}")
        
        # Filter for spans that match our test queries
        test_queries = [q['query'] for q in VISUAL_TEST_QUERIES]
        
        # Look for spans with queries matching our test set
        test_spans = []
        for _, span in all_spans.iterrows():
            attributes = span.get('attributes', {})
            query = attributes.get('query', '')
            
            if query in test_queries:
                # Mark it as a test query
                attributes['is_test_query'] = True
                attributes['dataset_id'] = 'comprehensive_test_v2'
                
                # Find which test query it matches
                for test_q in VISUAL_TEST_QUERIES:
                    if test_q['query'] == query:
                        attributes['expected_videos'] = test_q['expected_videos']
                        attributes['query_category'] = test_q['category']
                        break
                
                span['attributes'] = attributes
                test_spans.append(span)
        
        if test_spans:
            logger.info(f"Found {len(test_spans)} spans matching comprehensive test queries")
            return pd.DataFrame(test_spans)
        else:
            logger.info("No matching test spans found, using mock data")
            return self._create_comprehensive_test_spans()
    
    def _create_comprehensive_test_spans(self) -> pd.DataFrame:
        """Create mock spans based on comprehensive test queries"""
        mock_spans = []
        
        # Create spans for each test query
        for i, query_data in enumerate(VISUAL_TEST_QUERIES[:5]):  # First 5 for demo
            mock_spans.append({
                "span_id": f"test_span_{i:03d}",
                "trace_id": f"test_trace_{i:03d}",
                "operation_name": "search_service.search",
                "attributes": {
                    "query": query_data["query"],
                    "is_test_query": True,
                    "dataset_id": "comprehensive_test_v2",
                    "query_category": query_data["category"],
                    "expected_videos": query_data["expected_videos"],
                    "profile": "frame_based_colpali",
                    "ranking_strategy": "binary_binary"
                },
                "outputs": {
                    "results": [
                        {"video_id": vid, "score": 0.85 - i*0.1}
                        for i, vid in enumerate(query_data["expected_videos"][:2])
                    ] + [
                        {"video_id": f"v_other_{i}", "score": 0.5 - i*0.1}
                        for i in range(3)
                    ]
                }
            })
        
        return pd.DataFrame(mock_spans)
    
    def create_golden_dataset_from_test_queries(self):
        """Create golden dataset from comprehensive test queries"""
        golden_dataset = {}
        
        for query_data in VISUAL_TEST_QUERIES:
            golden_dataset[query_data["query"]] = {
                "expected_videos": query_data["expected_videos"],
                "relevance_scores": {}  # Could add scores if available
            }
        
        # Add the challenging queries too
        golden_dataset.update(create_low_scoring_golden_dataset())
        
        return golden_dataset


async def evaluate_comprehensive_test_spans(hours: int = 24):
    """Main evaluation function"""
    
    logger.info("="*70)
    logger.info("EVALUATING COMPREHENSIVE TEST SPANS")
    logger.info("="*70)
    
    # Create evaluator with golden dataset from test queries
    evaluator = ComprehensiveTestEvaluator()
    
    # Update golden dataset
    evaluator.golden_evaluator.golden_dataset = evaluator.create_golden_dataset_from_test_queries()
    
    # Get comprehensive test spans
    test_spans = evaluator.get_comprehensive_test_spans(hours=hours)
    logger.info(f"\nRetrieved {len(test_spans)} test spans for evaluation")
    
    # Show sample spans
    if not test_spans.empty:
        logger.info("\nSample test spans:")
        for i, (_, span) in enumerate(test_spans.head(3).iterrows()):
            attrs = span.get('attributes', {})
            logger.info(f"  {i+1}. Query: '{attrs.get('query', 'N/A')}'")
            logger.info(f"     Category: {attrs.get('query_category', 'N/A')}")
            logger.info(f"     Expected videos: {attrs.get('expected_videos', [])}")
    
    # Run evaluations
    logger.info("\n" + "-"*50)
    logger.info("Running evaluations...")
    
    evaluation_results = await evaluator.evaluate_spans(
        test_spans,
        evaluator_names=["relevance", "diversity", "temporal_coverage", "golden_dataset", "composite"]
    )
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    
    for eval_name, eval_df in evaluation_results.items():
        if eval_df.empty:
            continue
            
        logger.info(f"\n{eval_name.upper()} Evaluator:")
        logger.info(f"  Evaluated spans: {len(eval_df)}")
        
        # Score statistics
        scores = eval_df['score'].values
        valid_scores = scores[scores >= 0]  # Exclude -1 (not evaluable)
        
        if len(valid_scores) > 0:
            logger.info(f"  Mean score: {valid_scores.mean():.3f}")
            logger.info(f"  Min score: {valid_scores.min():.3f}")
            logger.info(f"  Max score: {valid_scores.max():.3f}")
        
        # Label distribution
        label_counts = eval_df['label'].value_counts()
        logger.info("  Label distribution:")
        for label, count in label_counts.items():
            logger.info(f"    {label}: {count}")
        
        # Show some examples
        if eval_name == "golden_dataset":
            logger.info("\n  Sample evaluations:")
            for i, (_, row) in enumerate(eval_df.head(3).iterrows()):
                logger.info(f"    {i+1}. Score: {row['score']:.3f}, Label: {row['label']}")
                logger.info(f"       Explanation: {row['explanation']}")
    
    # Upload to Phoenix
    logger.info("\n" + "-"*50)
    logger.info("Uploading evaluations to Phoenix...")
    evaluator.upload_evaluations_to_phoenix(evaluation_results)
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    total_spans = len(test_spans)
    golden_results = evaluation_results.get('golden_dataset', pd.DataFrame())
    if not golden_results.empty:
        successful_golden = (golden_results['score'] > 0).sum()
        logger.info(f"Total test spans evaluated: {total_spans}")
        logger.info(f"Successfully matched golden dataset: {successful_golden}/{len(golden_results)}")
        
        # Category breakdown
        logger.info("\nPerformance by query category:")
        for _, span in test_spans.iterrows():
            category = span.get('attributes', {}).get('query_category', 'unknown')
            span_id = span.get('span_id')
            
            # Find evaluation for this span
            span_eval = golden_results[golden_results['span_id'] == span_id]
            if not span_eval.empty:
                score = span_eval.iloc[0]['score']
                logger.info(f"  {category}: {score:.3f}")
    
    logger.info("\n✅ Evaluation complete! Check Phoenix UI for detailed results:")
    logger.info("   http://localhost:6006/projects/UHJvamVjdDox/traces")
    

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate comprehensive test spans"
    )
    
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to look back for test spans"
    )
    
    args = parser.parse_args()
    
    try:
        await evaluate_comprehensive_test_spans(hours=args.hours)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
