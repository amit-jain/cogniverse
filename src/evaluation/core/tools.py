"""
Tools for Inspect AI solvers to interact with external services.
"""

from typing import List, Dict, Any, Optional
import logging

from inspect_ai.tool import tool

logger = logging.getLogger(__name__)


@tool
def video_search_tool():
    """
    Tool that wraps our search service for use in Inspect AI.
    """
    async def run(
        query: str,
        profile: str,
        strategy: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute a search using the Cogniverse search service.
        
        Args:
            query: Search query
            profile: Video processing profile
            strategy: Ranking strategy
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Import here to avoid circular dependencies
            from src.search.search_service import SearchService
            from src.tools.config import get_config
            
            config = get_config()
            
            # Create search service with specified profile
            search_service = SearchService(config, profile)
            
            logger.info(f"Executing search: query='{query[:50]}...', profile={profile}, strategy={strategy}")
            
            # Run the search
            search_results_raw = search_service.search(
                query=query,
                top_k=top_k,
                ranking_strategy=strategy
            )
            
            # Convert results to standardized format
            search_results = []
            for i, result in enumerate(search_results_raw):
                result_dict = result.to_dict()
                
                # Extract video_id
                video_id = result_dict.get('source_id', '')
                if not video_id and 'document_id' in result_dict:
                    # Extract from document_id (e.g., "video_frame_0" -> "video")
                    doc_id = result_dict['document_id']
                    if '_frame_' in doc_id:
                        video_id = doc_id.split('_frame_')[0]
                    else:
                        video_id = doc_id
                
                # Get score (ensure it's not 0)
                score = result_dict.get('score', 0.0)
                if score == 0.0:
                    # Use rank-based score if no score available
                    score = 1.0 / (i + 1)
                
                search_results.append({
                    "video_id": video_id,
                    "score": float(score),
                    "rank": i + 1,
                    "document_id": result_dict.get('document_id', ''),
                    "content": result_dict.get('content', ''),
                    "temporal_info": result_dict.get('temporal_info', {}),
                    "metadata": result_dict.get('metadata', {})
                })
            
            logger.info(f"Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty results on failure
            return []
    
    return run


@tool
def phoenix_query_tool():
    """
    Tool for querying Phoenix for traces and datasets.
    """
    async def run(
        query_type: str,
        **kwargs
    ) -> Any:
        """
        Query Phoenix for various data types.
        
        Args:
            query_type: Type of query ("traces", "datasets", "experiments")
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        import phoenix as px
        
        client = px.Client()
        
        try:
            if query_type == "traces":
                # Query for traces
                df = client.get_spans_dataframe(
                    filter_condition=kwargs.get("filter"),
                    start_time=kwargs.get("start_time"),
                    end_time=kwargs.get("end_time"),
                    limit=kwargs.get("limit", 100)
                )
                return df.to_dict(orient='records')
                
            elif query_type == "datasets":
                # Get dataset information
                dataset_name = kwargs.get("name")
                if dataset_name:
                    dataset = client.get_dataset(dataset_name)
                    return {
                        "name": dataset.name,
                        "num_examples": len(dataset.examples),
                        "examples": [ex.to_dict() for ex in dataset.examples[:10]]
                    }
                return []
                
            elif query_type == "experiments":
                # Query experiments (placeholder)
                logger.warning("Experiment querying not yet implemented")
                return []
                
            else:
                raise ValueError(f"Unknown query type: {query_type}")
                
        except Exception as e:
            logger.error(f"Phoenix query failed: {e}")
            return []
    
    return run