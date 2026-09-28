# Business Entity Resolution Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: LightGBM](https://img.shields.io/badge/Model-LightGBM%20GBDT-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Architecture: Multi-Stage ER](https://img.shields.io/badge/Architecture-Inverted--Index%20%2B%20GBDT-blue.svg)](ARCHITECTURE.md)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](#)

A high-performance, precision-calibrated Machine Learning pipeline for large-scale **Business Entity Resolution (ER)** and record deduplication across disparate, noisy corporate databases.

Designed to resolve non-deterministic entity representations across heterogeneous data sources without universal identifiers, scaling efficiently to millions of records while operating within strict memory bounds.

---

## 📌 Problem Formulation

In modern enterprise architectures, customer, supplier, and company identity records are distributed across independent data silos (e.g. ERP systems, CRM platforms, supply chain registries). These datasets exhibit severe real-world challenges:
- **Missing Global Keys**: Records share no universal primary or foreign keys.
- **Typographical & Phonetic Noise**: Divergent spellings, transliterations, and abbreviations.
- **Legal Entity Variations**: Inconsistent suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`).
- **Address Heterogeneity**: Inconsistent ordering, landmark-based descriptions, and formatting differences.
- **Cardinality ($0, 1, \dots, N$)**: A reference entity may link to zero counterpart records (a *singleton*), one exact record, or multiple operational entities.

---

## 🏗️ Architecture & Pipeline Flow

The resolution engine employs a decoupled, multi-stage architecture engineered for high recall candidate generation and precision-calibrated pairwise matching:

```
[ Multi-Source Ingestion (S1, S2, S3) ]
                   │
                   ▼
[ Stage 1: Domain-Aware Preprocessing & Normalization ]
  • Unicode NFKD normalization & diacritic stripping
  • Jurisdictional legal suffix standardization (US, India, Europe)
  • Address abbreviation expansion & invariant numeric token extraction
                   │
                   ▼
[ Stage 2: Multi-Key Inverted-Index Blocking Engine ]
  • Exact geographic sharding & country partitioning
  • Inverted index on high-IDF name tokens and character shingles
  • High-recall candidate pruning (> 99.9% search space reduction)
  • Generates candidate_pairs.tsv
                   │
                   ▼
[ Stage 3: Multi-Modal Feature Engineering ]
  • 19-dimensional pairwise feature extraction:
    - Orthographic: Levenshtein, Jaro-Winkler, Token Sort & Token Set ratios
    - Substring: Character 3-gram TF-IDF vector space cosine similarity
    - Address Alignment: Token Jaccard, character n-gram similarity, length delta
    - Numeric Invariants: PIN/ZIP code and street number congruence / contradiction flags
                   │
                   ▼
[ Stage 4: Precision-Calibrated GBDT Matching & Ranking ]
  • LightGBM Gradient Boosted Decision Tree with balanced class weighting
  • Macro-averaged F_0.5 decision boundary optimization (Precision weighted 2x)
  • Singleton guardrails & post-filtering
  • Generates matching_results.tsv
                   │
                   ▼
[ Stage 5: Output Validation & Relational Lineage Auditing ]
  • Schema verification & candidate-match subset guarantees via utils/validate_output.py
```

For comprehensive mathematical formulations and component breakdowns, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📊 Evaluation Metric ($F_{0.5}$)

Entity resolution in business applications carries asymmetric risk: merging two distinct corporations (false positive) introduces catastrophic data corruption, whereas omitting a weak connection (false negative) is far less harmful.

The pipeline is calibrated against **Macro-averaged $F_{0.5}$**, weighting precision **$2\times$ over recall**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Singleton Mechanics:
- **Correct Singleton** ($\text{GT} = \emptyset, \text{Pred} = \emptyset$): Receives full score ($1.0$).
- **False Positive on Singleton** ($\text{GT} = \emptyset, \text{Pred} \ne \emptyset$): Instantly drops entity score to $0.0$.
- **Missed Match** ($\text{GT} \ne \emptyset, \text{Pred} = \emptyset$): Score is $0.0$.

Our decision boundary is strictly optimized against this metric to prevent low-confidence matches from degrading the macro score.

---

## 📂 Repository Structure

```
├── ARCHITECTURE.md                  # Comprehensive architectural specification & methodology
├── LICENSE                          # MIT License
├── README.md                        # Primary project documentation
├── ultra_pipeline.py                # High-throughput batch pipeline (handles millions of records)
├── utils/
│   └── validate_output.py           # Zero-dependency output & schema integrity validator
├── output/
│   ├── candidate_pairs.tsv          # Generated candidate pairs from blocking stage
│   └── matching_results.tsv         # Final precision-calibrated predicted matches
├── max_models/
│   └── max_metadata.json            # Model parameters, threshold calibration & feature weights
└── code/
    └── business_entity_resolution/
        ├── README.md                # Modular reproduction guide
        ├── requirements.txt         # Pinned production dependencies
        └── src/
            ├── __init__.py          # Package initialization
            ├── config.py            # Global paths and model hyperparameters
            ├── preprocess.py        # Text cleaning & legal suffix harmonization
            ├── blocking.py          # Candidate generation & inverted index
            ├── features.py          # 19-dimensional feature engineering
            ├── train.py             # Model training & threshold calibration routine
            ├── matcher.py           # LightGBM scoring & candidate thresholding
            ├── evaluate.py          # Macro-averaged F_0.5 evaluation implementation
            ├── pipeline.py          # End-to-end execution runner
            └── models/
                ├── lgbm_matcher.joblib # Serialized model binary
                └── metadata.json       # Baseline model metrics & threshold configuration
```

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/Khangulamgousamjat/Business-Entity-Resolution.git
cd Business-Entity-Resolution

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run High-Scale Resolution Pipeline

To run the end-to-end blocking, feature extraction, and calibrated matching pipeline:

```bash
python ultra_pipeline.py \
  --train-dir train \
  --test-dir test \
  --output-dir output \
  --model-dir max_models
```

Options:
- `--train-dir`: Directory containing ground truth and training source files.
- `--test-dir`: Directory containing evaluation source files (`test_source1.tsv`, etc.).
- `--output-dir`: Output directory for `matching_results.tsv` and `candidate_pairs.tsv`.
- `--skip-train`: Skip training if a pre-trained model checkpoint exists in `--model-dir`.

### 3. Validate Output Integrity

Verify output schema conformity, encoding, singleton formatting, and candidate subset guarantees:

```bash
python utils/validate_output.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir test
```

---

## 🔬 Feature Engineering Overview

The matching engine extracts 19 orthogonal features per candidate pair:

| Category | Features |
| :--- | :--- |
| **Name Orthography** | Levenshtein Ratio, Jaro-Winkler Distance, Token Sort Ratio, Token Set Ratio |
| **Name Semantic / Substring** | Character 3-Gram TF-IDF Cosine Similarity, Name Token Jaccard, Token Length Delta |
| **Address Alignment** | Address Levenshtein Ratio, Token Sort Ratio, Token Jaccard Overlap |
| **Address Substring** | Character 3-Gram Cosine Similarity, Address Length Delta, Missing Address Indicator |
| **Numeric Invariants** | Numeric Token Jaccard, Matching Digits Count, Numeric Contradiction Indicator |
| **Metadata** | Source Origin Indicator ($S_2$ vs $S_3$) |

---

## 📈 Scalability & Performance

- **Memory Bounded Execution**: Country sharding and streaming file reading maintain peak memory consumption strictly under 1.5 GB RAM, even when processing millions of business records.
- **Sub-Second Candidate Retrieval**: Multi-key inverted indexing yields $>99.9\%$ candidate reduction ratio, reducing trillions of possible pairwise comparisons to high-confidence candidate neighborhoods.
- **Asymmetric Calibration**: Optimal threshold tuning maximizes macro $F_{0.5}$ while preserving singleton accuracy across diverse jurisdictional distributions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
