"""
Custom evaluation metrics for video retrieval.
"""

from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_mrr(results: List[str], expected: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items

    Returns:
        MRR score (0 to 1)
    """
    if not expected or not results:
        return 0.0

    for i, item in enumerate(results):
        if item in expected:
            return 1.0 / (i + 1)

    return 0.0


def calculate_ndcg(results: List[str], expected: List[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        NDCG@K score (0 to 1)
    """
    if not expected:
        return 0.0

    # Limit to top k results
    results_k = results[:k]

    # Calculate relevance scores (1 if relevant, 0 otherwise)
    relevances = [1 if item in expected else 0 for item in results_k]

    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)  # i+2 because positions start at 1

    # Calculate ideal DCG
    ideal_relevances = [1] * min(len(expected), k)
    ideal_relevances += [0] * max(0, k - len(expected))

    idcg = 0.0
    for i, rel in enumerate(ideal_relevances[:k]):
        idcg += rel / np.log2(i + 2)

    # Calculate NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_precision_at_k(
    results: List[str], expected: List[str], k: int = 5
) -> float:
    """
    Calculate Precision at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        Precision@K score (0 to 1)
    """
    if not results or k == 0:
        return 0.0

    results_k = results[:k]
    relevant_retrieved = len([item for item in results_k if item in expected])

    return relevant_retrieved / len(results_k)


def calculate_recall_at_k(results: List[str], expected: List[str], k: int = 5) -> float:
    """
    Calculate Recall at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        Recall@K score (0 to 1)
    """
    if not expected:
        return 0.0

    results_k = results[:k]
    relevant_retrieved = len([item for item in expected if item in results_k])

    return relevant_retrieved / len(expected)


def calculate_f1_at_k(results: List[str], expected: List[str], k: int = 5) -> float:
    """
    Calculate F1 score at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        F1@K score (0 to 1)
    """
    precision = calculate_precision_at_k(results, expected, k)
    recall = calculate_recall_at_k(results, expected, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_map(
    results_list: List[List[str]], expected_list: List[List[str]]
) -> float:
    """
    Calculate Mean Average Precision across multiple queries.

    Args:
        results_list: List of result lists (one per query)
        expected_list: List of expected result lists (one per query)

    Returns:
        MAP score (0 to 1)
    """
    if not results_list or not expected_list:
        return 0.0

    if len(results_list) != len(expected_list):
        raise ValueError("Results and expected lists must have same length")

    average_precisions = []

    for results, expected in zip(results_list, expected_list):
        if not expected:
            continue

        # Calculate average precision for this query
        precisions = []
        num_relevant = 0

        for i, item in enumerate(results):
            if item in expected:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precisions.append(precision)

        if precisions:
            ap = sum(precisions) / len(expected)
        else:
            ap = 0.0

        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)


def calculate_metrics_suite(
    results: List[str], expected: List[str], k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calculate a suite of metrics for a single query.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k_values: List of k values for @K metrics

    Returns:
        Dictionary of metric names to scores
    """
    metrics = {
        "mrr": calculate_mrr(results, expected),
        "ndcg": calculate_ndcg(results, expected),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = calculate_precision_at_k(results, expected, k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(results, expected, k)
        metrics[f"f1@{k}"] = calculate_f1_at_k(results, expected, k)

    return metrics
