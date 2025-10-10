"""
Example of using the evaluation framework with a document/text search schema.

This demonstrates that the system is schema-agnostic and can work with
any type of search backend, not just video search.
"""

import asyncio
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDocumentBackend:
    """Mock backend for document search to demonstrate schema flexibility."""

    def __init__(self):
        self.schema_name = "document_search"
        self.documents = [
            {
                "doc_id": "doc001",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence...",
                "author": "John Smith",
                "date": "2023-01-15",
                "category": "AI",
            },
            {
                "doc_id": "doc002",
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers...",
                "author": "Jane Doe",
                "date": "2023-02-20",
                "category": "AI",
            },
            {
                "doc_id": "doc003",
                "title": "Natural Language Processing",
                "content": "NLP enables computers to understand human language...",
                "author": "Bob Johnson",
                "date": "2023-03-10",
                "category": "AI",
            },
            {
                "doc_id": "doc004",
                "title": "Computer Vision Applications",
                "content": "Computer vision allows machines to interpret visual information...",
                "author": "Alice Brown",
                "date": "2023-04-05",
                "category": "AI",
            },
            {
                "doc_id": "doc005",
                "title": "Python Programming Guide",
                "content": "Python is a versatile programming language used in many fields...",
                "author": "Charlie Wilson",
                "date": "2023-01-20",
                "category": "Programming",
            },
        ]

    async def get_schema_info(self, schema_name: str) -> dict[str, Any]:
        """Return schema information for document search."""
        return {
            "name": "document_search",
            "fields": {
                "id_fields": ["doc_id"],
                "content_fields": ["content", "title"],
                "metadata_fields": ["author", "category"],
                "temporal_fields": ["date"],
                "text_fields": ["title", "author"],
                "numeric_fields": [],
            },
        }

    async def search(
        self, query_text: str, top_k: int = 10, **kwargs
    ) -> list[dict[str, Any]]:
        """Simple keyword search in documents."""
        results = []
        query_lower = query_text.lower()

        for doc in self.documents:
            # Simple relevance scoring based on keyword matches
            score = 0.0
            if query_lower in doc["title"].lower():
                score += 0.5
            if query_lower in doc["content"].lower():
                score += 0.3
            if query_lower in doc["category"].lower():
                score += 0.2

            if score > 0:
                results.append({**doc, "score": score})

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


async def run_document_search_evaluation():
    """Run evaluation on document search system."""
    from cogniverse_dashboard.evaluation.core.ground_truth import SchemaAwareGroundTruthStrategy
    from cogniverse_dashboard.evaluation.core.schema_analyzer import get_schema_analyzer

    # Document analyzer will be automatically selected based on schema
    # Create mock backend
    backend = MockDocumentBackend()

    # Test queries
    test_queries = [
        {
            "query": "machine learning",
            "expected_docs": ["doc001", "doc002"],  # Documents about ML
        },
        {"query": "python", "expected_docs": ["doc005"]},  # Python programming guide
        {"query": "vision", "expected_docs": ["doc004"]},  # Computer vision
    ]

    # Ground truth extraction
    ground_truth_strategy = SchemaAwareGroundTruthStrategy()

    for test_case in test_queries:
        query = test_case["query"]
        expected = test_case["expected_docs"]

        logger.info(f"\n{'='*50}")
        logger.info(f"Testing query: '{query}'")
        logger.info(f"Expected documents: {expected}")

        # Get schema info
        schema_info = await backend.get_schema_info("document_search")

        # Get analyzer
        analyzer = get_schema_analyzer(schema_info["name"], schema_info["fields"])

        logger.info(f"Using analyzer: {analyzer.__class__.__name__}")

        # Analyze query
        query_analysis = analyzer.analyze_query(query, schema_info["fields"])
        logger.info(f"Query analysis: {query_analysis}")

        # Search
        search_results = await backend.search(query)
        logger.info(f"Found {len(search_results)} results")

        # Extract IDs from results
        retrieved_ids = []
        for result in search_results:
            doc_id = analyzer.extract_item_id(result)
            if doc_id:
                retrieved_ids.append(doc_id)

        logger.info(f"Retrieved documents: {retrieved_ids}")

        # Calculate precision/recall
        if expected:
            retrieved_set = set(retrieved_ids)
            expected_set = set(expected)

            intersection = retrieved_set & expected_set
            precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
            recall = len(intersection) / len(expected_set) if expected_set else 0

            logger.info(f"Precision: {precision:.2f}")
            logger.info(f"Recall: {recall:.2f}")

        # Test ground truth extraction
        trace_data = {
            "query": query,
            "metadata": {"schema": "document_search", "fields": schema_info["fields"]},
        }

        ground_truth_result = await ground_truth_strategy.extract_ground_truth(
            trace_data, backend
        )

        logger.info("Ground truth extraction:")
        logger.info(f"  - Items: {ground_truth_result.get('expected_items', [])}")
        logger.info(f"  - Confidence: {ground_truth_result.get('confidence', 0):.2f}")
        logger.info(f"  - Source: {ground_truth_result.get('source', 'unknown')}")


async def run_image_search_evaluation():
    """Run evaluation on image search system."""
    from cogniverse_dashboard.evaluation.core.schema_analyzer import DefaultSchemaAnalyzer

    logger.info("\n" + "=" * 60)
    logger.info("IMAGE SEARCH EXAMPLE")
    logger.info("=" * 60)

    class MockImageBackend:
        """Mock backend for image search."""

        def __init__(self):
            self.schema_name = "image_search"
            self.images = [
                {
                    "image_id": "img001",
                    "description": "sunset over mountains",
                    "tags": ["nature", "sunset"],
                },
                {
                    "image_id": "img002",
                    "description": "cat playing with yarn",
                    "tags": ["animal", "cat"],
                },
                {
                    "image_id": "img003",
                    "description": "city skyline at night",
                    "tags": ["urban", "night"],
                },
            ]

        async def get_schema_info(self, schema_name: str) -> dict[str, Any]:
            return {
                "name": "image_search",
                "fields": {
                    "id_fields": ["image_id"],
                    "content_fields": ["description"],
                    "metadata_fields": ["tags"],
                    "temporal_fields": [],
                    "text_fields": ["description"],
                    "numeric_fields": [],
                },
            }

        async def search(self, query_text: str, top_k: int = 10, **kwargs):
            results = []
            query_lower = query_text.lower()

            for img in self.images:
                score = 0.0
                if query_lower in img["description"].lower():
                    score += 0.6
                for tag in img.get("tags", []):
                    if query_lower in tag.lower():
                        score += 0.4

                if score > 0:
                    results.append({**img, "score": score})

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    # Test with image backend
    backend = MockImageBackend()
    _ = await backend.get_schema_info("image_search")  # noqa: F841

    # Use default analyzer for images
    analyzer = DefaultSchemaAnalyzer()

    test_query = "sunset"
    logger.info(f"Testing image search with query: '{test_query}'")

    # Search
    results = await backend.search(test_query)
    logger.info(f"Found {len(results)} results")

    for result in results:
        image_id = analyzer.extract_item_id(result)
        logger.info(
            f"  - {image_id}: {result.get('description')} (score: {result.get('score'):.2f})"
        )


if __name__ == "__main__":
    # Run both examples
    asyncio.run(run_document_search_evaluation())
    asyncio.run(run_image_search_evaluation())
