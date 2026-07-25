"""
Custom evaluation metrics for video retrieval.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _validate_k(k: int) -> None:
    if k < 0:
        raise ValueError("k must be non-negative")


def _binary_relevance_by_rank(results: list[str], expected_ids: set[str]) -> list[int]:
    credited_ids: set[str] = set()
    relevances = []
    for item in results:
        is_new_relevant = item in expected_ids and item not in credited_ids
        relevances.append(int(is_new_relevant))
        if is_new_relevant:
            credited_ids.add(item)
    return relevances


def calculate_mrr(results: list[str], expected: list[str]) -> float:
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


def calculate_ndcg(results: list[str], expected: list[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        NDCG@K score (0 to 1)
    """
    _validate_k(k)
    expected_ids = set(expected)
    if not expected_ids:
        return 0.0

    results_k = results[:k]
    relevances = _binary_relevance_by_rank(results_k, expected_ids)

    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)

    ideal_relevances = [1] * min(len(expected_ids), k)

    idcg = 0.0
    for i, rel in enumerate(ideal_relevances):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_precision_at_k(
    results: list[str], expected: list[str], k: int = 5
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
    _validate_k(k)
    if not results or k == 0:
        return 0.0

    results_k = results[:k]
    relevant_retrieved = sum(_binary_relevance_by_rank(results_k, set(expected)))

    return relevant_retrieved / len(results_k)


def calculate_recall_at_k(results: list[str], expected: list[str], k: int = 5) -> float:
    """
    Calculate Recall at K.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k: Cutoff position

    Returns:
        Recall@K score (0 to 1)
    """
    _validate_k(k)
    expected_ids = set(expected)
    if not expected_ids:
        return 0.0

    results_k = results[:k]
    relevant_retrieved = sum(_binary_relevance_by_rank(results_k, expected_ids))

    return relevant_retrieved / len(expected_ids)


def calculate_f1_at_k(results: list[str], expected: list[str], k: int = 5) -> float:
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
    results_list: list[list[str]], expected_list: list[list[str]]
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

    for results, expected in zip(results_list, expected_list, strict=False):
        if not expected:
            continue

        expected_ids = set(expected)
        precisions = []
        num_relevant = 0
        credited_ids: set[str] = set()

        for i, item in enumerate(results):
            if item in expected_ids and item not in credited_ids:
                credited_ids.add(item)
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precisions.append(precision)

        if precisions:
            ap = sum(precisions) / len(expected_ids)
        else:
            ap = 0.0

        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)


def calculate_metrics_suite(
    results: list[str], expected: list[str], k_values: list[int] = None
) -> dict[str, float]:
    """
    Calculate a suite of metrics for a single query.

    Args:
        results: List of retrieved items
        expected: List of expected/relevant items
        k_values: List of k values for @K metrics

    Returns:
        Dictionary of metric names to scores
    """
    if k_values is None:
        k_values = [1, 5, 10]
    metrics = {
        "mrr": calculate_mrr(results, expected),
        "ndcg": calculate_ndcg(results, expected),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = calculate_precision_at_k(results, expected, k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(results, expected, k)
        metrics[f"f1@{k}"] = calculate_f1_at_k(results, expected, k)

    return metrics
