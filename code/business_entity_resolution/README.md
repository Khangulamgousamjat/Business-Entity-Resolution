# Business Entity Resolution Pipeline

This folder contains the complete, self-contained, reproducible pipeline for the Amazon ML Challenge: Business Entity Resolution.

## Directory Structure

```
business_entity_resolution/
├── README.md              # Reproduction instructions
├── requirements.txt       # Pinned dependencies
└── src/
    ├── __init__.py
    ├── config.py          # Paths and tunable hyperparameters
    ├── preprocess.py      # Address & business name normalization
    ├── blocking.py        # Candidate generation & inverted index
    ├── features.py        # String distance and similarity features
    ├── train.py           # Model training and threshold calibration
    ├── matcher.py         # LightGBM matcher with calibrated threshold
    ├── evaluate.py        # Official macro F_0.5 evaluation logic
    ├── pipeline.py        # End-to-end inference runner script
    └── models/
        ├── lgbm_matcher.joblib # Trained LightGBM GBDT model
        └── metadata.json       # Optimal threshold and feature importances
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

## Reproducing Results

### 1. (Optional) Retrain the Matching Model
The repository already includes the pre-trained LightGBM model weights and threshold under `src/models/`. If you wish to retrain from scratch:
```bash
python src/train.py \
  --train-dir ../../train \
  --output-dir src/models \
  --n-samples 25000 \
  --val-ratio 0.2
```

### 2. Run End-to-End Inference Pipeline
Run the full candidate blocking and matching pipeline to generate both required submission TSV files:
```bash
python src/pipeline.py \
  --test-dir ../../test \
  --output-dir ../../output \
  --model-dir src/models
```

This generates:
- `output/candidate_pairs.tsv`: All candidates generated during the blocking stage.
- `output/matching_results.tsv`: Final precision-calibrated predicted matches.

### 3. Validate Submission
Run the official validator script to verify strict format compliance:
```bash
python ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../test \
  --check-ids
```
