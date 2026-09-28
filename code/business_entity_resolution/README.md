# Business Entity Resolution Module

This directory contains the core modular architecture, feature engineering utilities, blocking algorithms, model training routines, and inference pipeline for large-scale **Business Entity Resolution (ER)**.

## Architecture & Directory Structure

```
business_entity_resolution/
├── README.md              # Module overview and reproduction guide
├── requirements.txt       # Dependencies
└── src/
    ├── __init__.py        # Package initialization
    ├── config.py          # Global path configurations and tunable hyperparameters
    ├── preprocess.py      # Address & business name normalization, legal suffix harmonization
    ├── blocking.py        # Candidate generation & inverted index blocking
    ├── features.py        # Multi-modal string distance and similarity feature extraction
    ├── train.py           # Model training and F_0.5 decision threshold calibration
    ├── matcher.py         # LightGBM matcher with calibrated thresholding
    ├── evaluate.py        # Macro-averaged F_0.5 evaluation metric implementation
    ├── pipeline.py        # End-to-end batch execution pipeline
    └── models/
        ├── lgbm_matcher.joblib # Serialized LightGBM GBDT model
        └── metadata.json       # Optimal decision threshold and feature importance weights
```

## Setup & Installation

1. Create and activate a Python 3.10+ virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

### 1. (Optional) Model Training & Threshold Calibration
To retrain the matching model from scratch on reference entity pairs:
```bash
python src/train.py \
  --train-dir ../../train \
  --output-dir src/models \
  --n-samples 25000 \
  --val-ratio 0.2
```

### 2. End-to-End Inference Execution
Run the full candidate blocking and matching pipeline across test entities:
```bash
python src/pipeline.py \
  --test-dir ../../test \
  --output-dir ../../output \
  --model-dir src/models
```

Generated outputs:
- `output/candidate_pairs.tsv`: All candidates generated during the blocking stage.
- `output/matching_results.tsv`: Final precision-calibrated predicted entity matches.

### 3. Output Schema & Integrity Validation
Verify format compliance, relational constraints, and singleton representations:
```bash
python ../../utils/validate_output.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../test \
  --check-ids
```
