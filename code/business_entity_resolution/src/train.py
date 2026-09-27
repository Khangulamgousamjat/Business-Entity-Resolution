#!/usr/bin/env python3
"""
Detailed Model Training and Validation Script for Business Entity Resolution.
1. Samples reference entities from train_source1.tsv and train_ground_truth.tsv.
2. Loads true matched entities from train_source2.tsv and train_source3.tsv.
3. Builds candidate blocking index to generate realistic hard negative pairs.
4. Computes rich pairwise similarity feature vectors.
5. Trains a LightGBM GBDT classifier with early stopping.
6. Evaluates out-of-fold validation set using official Macro F_0.5 metric.
7. Optimizes decision threshold tau* to maximize F_0.5 score.
8. Serializes trained model and metadata into models/ directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import clean_business_name, clean_address
from blocking import InvertedIndexBlocker
from features import compute_pair_features, FEATURE_NAMES
from evaluate import compute_entity_f05, evaluate_predictions


def load_ground_truth_sample(
    gt_path: str,
    n_samples: int
) -> Tuple[Dict[str, Set[str]], Set[str], Set[str]]:
    """Load ground truth mappings and set of required target IDs."""
    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            matched_ids = set(m.strip() for m in matched_str.split(",") if m.strip())
            gt_map[s1_id] = matched_ids

            for mid in matched_ids:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

            if len(gt_map) >= n_samples:
                break

    return gt_map, needed_s2, needed_s3


def load_source_records(
    filepath: str,
    filter_ids: Set[str] = None,
    max_records: int = None
) -> Dict[str, Tuple[str, str, str]]:
    """Load records mapping entity_id -> (business_name, business_address, country)."""
    records = {}
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        country_idx = header.index("country") if "country" in header else 3

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(eid_idx, name_idx, addr_idx, country_idx):
                continue
            eid = parts[eid_idx]
            if filter_ids is None or eid in filter_ids:
                records[eid] = (parts[name_idx], parts[addr_idx], parts[country_idx])

            if max_records and len(records) >= max_records:
                break

    return records


def train_model(train_dir: str, output_dir: str, n_samples: int = 25000, val_ratio: float = 0.2):
    print("=" * 70)
    print("AMAZON ML CHALLENGE: DETAILED ENTITY RESOLUTION MODEL TRAINING")
    print("=" * 70)
    print(f"[*] Training Directory: {train_dir}")
    print(f"[*] Output Model Dir:   {output_dir}")
    print(f"[*] Reference Samples:  {n_samples:,}")
    print(f"[*] Validation Ratio:   {val_ratio:.2f}")

    os.makedirs(output_dir, exist_ok=True)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # 1. Load Ground Truth subset
    print("\n[Step 1/6] Loading Ground Truth sample...")
    gt_map, needed_s2, needed_s3 = load_ground_truth_sample(gt_file, n_samples)
    print(f"    Loaded {len(gt_map):,} Source 1 ground truth mappings.")
    print(f"    Required true matches: {len(needed_s2):,} S2 IDs, {len(needed_s3):,} S3 IDs.")

    # 2. Load Source 1 records
    print("\n[Step 2/6] Loading Source 1 records...")
    s1_records = load_source_records(s1_file, filter_ids=set(gt_map.keys()))
    print(f"    Loaded {len(s1_records):,} Source 1 records.")

    # 3. Load Target Source records (True matches + background pool for realistic negatives)
    print("\n[Step 3/6] Loading Source 2 & 3 target records...")
    s2_records = load_source_records(s2_file, filter_ids=needed_s2)
    s3_records = load_source_records(s3_file, filter_ids=needed_s3)

    # Load additional background target records for hard negative mining
    extra_target_limit = 40000
    extra_s2 = load_source_records(s2_file, max_records=extra_target_limit)
    extra_s3 = load_source_records(s3_file, max_records=extra_target_limit)
    s2_records.update(extra_s2)
    s3_records.update(extra_s3)
    target_records = {**s2_records, **s3_records}
    print(f"    Total target pool available: {len(target_records):,} records.")

    # 4. Build Candidate Blocking Index & Generate Hard Negatives
    print("\n[Step 4/6] Building blocking index and mining hard negative pairs...")
    blocker = InvertedIndexBlocker(max_token_freq=2500, top_k_candidates=10)
    target_tuples = [(eid, data[0], data[1], data[2]) for eid, data in target_records.items()]
    blocker.fit_target_source(target_tuples)

    # Split S1 IDs into Train and Validation sets
    all_s1_ids = list(s1_records.keys())
    np.random.seed(42)
    np.random.shuffle(all_s1_ids)
    n_val = int(len(all_s1_ids) * val_ratio)
    val_s1_ids = set(all_s1_ids[:n_val])
    train_s1_ids = set(all_s1_ids[n_val:])

    print(f"    Train Reference Entities: {len(train_s1_ids):,}")
    print(f"    Val Reference Entities:   {len(val_s1_ids):,}")

    X_train, y_train = [], []
    X_val, y_val = [], []
    val_candidates_map = {}

    for s1_id in all_s1_ids:
        is_val = s1_id in val_s1_ids
        s1_name, s1_addr, s1_country = s1_records[s1_id]
        true_matches = gt_map.get(s1_id, set())

        # Retrieve blocking candidates
        cand_ids = blocker.get_candidates(s1_id, s1_name, s1_addr, s1_country)
        all_cands = set(cand_ids) | (true_matches if not is_val else set())

        if is_val:
            val_candidates_map[s1_id] = cand_ids

        # Pairwise features for training
        for cand_id in all_cands:
            if cand_id not in target_records:
                continue
            cand_name, cand_addr, _ = target_records[cand_id]
            feats = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr, cand_id)
            label = 1 if cand_id in true_matches else 0

            if is_val:
                X_val.append(feats)
                y_val.append(label)
            else:
                X_train.append(feats)
                y_train.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    print(f"    Train Pairs: {len(y_train):,} (Pos: {np.sum(y_train == 1):,}, Neg: {np.sum(y_train == 0):,})")
    print(f"    Val Pairs:   {len(y_val):,} (Pos: {np.sum(y_val == 1):,}, Neg: {np.sum(y_val == 0):,})")

    # 5. Train GBDT Matcher (LightGBM)
    print("\n[Step 5/6] Training LightGBM Classifier...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss"
    )
    print("    Model training completed successfully.")

    # Feature Importance
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    print("\n    Top Feature Importances:")
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:8]:
        print(f"      - {feat:20s}: {imp}")

    # 6. Optimize Macro F_0.5 Threshold on Validation Set
    print("\n[Step 6/6] Calibrating Decision Threshold for Macro F_0.5...")
    val_probs = model.predict_proba(X_val)[:, 1]

    # Pre-index validation probabilities back to entities
    idx = 0
    val_pair_probs = {}
    for s1_id in val_s1_ids:
        cands = val_candidates_map.get(s1_id, [])
        val_pair_probs[s1_id] = []
        for cand_id in cands:
            if cand_id in target_records:
                prob = val_probs[idx]
                val_pair_probs[s1_id].append((cand_id, prob))
                idx += 1

    best_thresh = 0.5
    best_f05 = 0.0

    # Grid search over precision-heavy threshold range
    for thresh in np.arange(0.50, 0.96, 0.02):
        thresh = round(float(thresh), 2)
        val_preds = {}
        for s1_id, cand_prob_list in val_pair_probs.items():
            matched = set(cid for cid, p in cand_prob_list if p >= thresh)
            val_preds[s1_id] = matched

        val_gt = {s1: gt_map.get(s1, set()) for s1 in val_s1_ids}
        score = evaluate_predictions(val_gt, val_preds)

        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    print(f"\n    [+] Optimal F_0.5 Threshold: {best_thresh:.2f}")
    print(f"    [+] Validation Macro F_0.5: {best_f05:.4f}")

    # Save Model Artifacts
    model_path = os.path.join(output_dir, "lgbm_matcher.joblib")
    joblib.dump(model, model_path)
    print(f"\n[OK] Saved trained model to: {model_path}")

    metadata = {
        "model_type": "LightGBM GBDT Classifier",
        "optimal_threshold": best_thresh,
        "validation_macro_f05": round(best_f05, 5),
        "n_train_pairs": int(len(y_train)),
        "n_val_pairs": int(len(y_val)),
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved model metadata to: {meta_path}")

    return best_thresh, best_f05


def main():
    parser = argparse.ArgumentParser(description="Train Business Entity Resolution Model.")
    parser.add_argument("--train-dir", default="../../train", help="Path to training directory.")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained model artifacts.")
    parser.add_argument("--n-samples", type=int, default=15000, help="Number of S1 entities to train on.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation holdout ratio.")
    args = parser.parse_args()

    train_model(args.train_dir, args.output_dir, args.n_samples, args.val_ratio)


if __name__ == "__main__":
    main()
