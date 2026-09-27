# Business Entity Resolution Pipeline

This repository contains the end-to-end, runnable solution for the Amazon ML Challenge: Business Entity Resolution.

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
    ├── matcher.py         # Classifier / calibrated ranker
    ├── evaluate.py        # Official macro F_0.5 evaluation logic
    └── pipeline.py        # End-to-end runner script
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

### 1. Run Full End-to-End Pipeline
```bash
python src/pipeline.py \
  --test-dir ../../test \
  --output-dir ../../output
```

This will generate:
- `output/candidate_pairs.tsv`: All candidates generated during the blocking stage.
- `output/matching_results.tsv`: Final precision-calibrated predicted matches.

### 2. Validate Submission
Run the official validator script:
```bash
python ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../test
```
