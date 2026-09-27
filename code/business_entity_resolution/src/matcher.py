"""
Entity Matcher and Precision-Calibrated Ranker.
Loads the trained LightGBM model and metadata threshold, scoring candidate pairs
with machine-learned weights and precision-tuned F_0.5 decision thresholds.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np

from features import compute_pair_features, FEATURE_NAMES


class EntityMatcher:
    def __init__(self, model_dir: str = None, threshold: float = None):
        if model_dir is None:
            model_dir = str(Path(__file__).resolve().parent / "models")

        model_path = os.path.join(model_dir, "lgbm_matcher.joblib")
        meta_path = os.path.join(model_dir, "metadata.json")

        self.model = None
        self.threshold = threshold or 0.80

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.threshold = threshold or meta.get("optimal_threshold", 0.80)
                print(f"[*] Loaded trained model from {model_path} with threshold {self.threshold:.2f}")
            except Exception as e:
                print(f"[!] Warning: Could not load trained model ({e}). Using composite score.")
                self.model = None

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
        if not candidates:
            return []

        s1_id, s1_name, s1_addr, _ = s1_record

        # Prepare feature batch
        cand_list = []
        feature_batch = []

        for cand_id in candidates:
            target_data = target_dict.get(cand_id)
            if not target_data:
                continue
            cand_name, cand_addr = target_data
            feats = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr, cand_id)
            cand_list.append(cand_id)
            feature_batch.append(feats)

        if not feature_batch:
            return []

        # Predict match probabilities
        if self.model is not None:
            X = np.array(feature_batch)
            probs = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback heuristic composite score
            probs = [
                0.40 * f[1] + 0.30 * f[3] + 0.15 * f[7] + 0.15 * f[13]  # sort, jaro, addr, num
                for f in feature_batch
            ]

        # Filter strictly above threshold
        matched = []
        for cand_id, prob in zip(cand_list, probs):
            if prob >= self.threshold:
                matched.append((cand_id, float(prob)))

        # Sort matches by probability descending
        matched.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in matched]
