"""
Entity Matcher and Precision-Calibrated Ranker.
Filters candidates using composite similarity features and calibrated F_0.5 decision thresholds.
"""

from typing import Dict, List, Set, Tuple
from features import compute_pair_features
from config import MATCH_THRESHOLD


class EntityMatcher:
    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold

    def score_and_filter(
        self,
        s1_record: Tuple[str, str, str, str],
        candidates: List[str],
        target_dict: Dict[str, Tuple[str, str]]
    ) -> List[str]:
        """
        Scores candidate entities for a Source 1 entity and selects matches above the threshold.
        s1_record: (entity_id, business_name, business_address, country)
        candidates: list of candidate entity_ids from blocking
        target_dict: mapping entity_id -> (clean_name, clean_address)
        """
        _, s1_name, s1_addr, _ = s1_record
        matched_ids = []

        for cand_id in candidates:
            target_data = target_dict.get(cand_id)
            if not target_data:
                continue

            cand_name, cand_addr = target_data
            feat = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr)

            if feat["composite_score"] >= self.threshold:
                matched_ids.append((cand_id, feat["composite_score"]))

        # Sort selected matches by confidence score descending
        matched_ids.sort(key=lambda x: x[1], reverse=True)
        return [cand_id for cand_id, _ in matched_ids]
