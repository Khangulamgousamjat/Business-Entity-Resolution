"""
Official metric calculation: Macro-averaged F_0.5 score.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed per Source 1 entity, then averaged across all Source 1 entities in the evaluation set.
Singletons:
- True matches == 0 and Predicted matches == 0 -> 1.0
- True matches == 0 and Predicted matches > 0  -> 0.0
- True matches > 0 and Predicted matches == 0  -> 0.0
"""

from typing import Dict, Set


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton cases
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    intersection = len(true_matches & pred_matches)
    if intersection == 0:
        return 0.0

    precision = intersection / n_pred
    recall = intersection / n_true

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def evaluate_predictions(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """
    Compute macro-averaged F_0.5 across all entities in ground truth.
    """
    total_f05 = 0.0
    n_entities = len(ground_truth)

    if n_entities == 0:
        return 0.0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_f05 += compute_entity_f05(true_set, pred_set)

    return total_f05 / n_entities
