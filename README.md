# Amazon ML Challenge: Business Entity Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Challenge: Amazon ML Challenge](https://img.shields.io/badge/Amazon-ML%20Challenge-orange.svg)](#)

A high-performance, precision-calibrated Machine Learning pipeline for large-scale **Business Entity Resolution (ER)**, matching noisy business records across multiple disparate data sources.

---

## 📌 Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources ($S_1$, $S_2$, $S_3$), each contributing partial, noisy fragments of information without common identifiers.

- **Source 1 ($S_1$)**: Deduplicated reference source.
- **Source 2 ($S_2$) & Source 3 ($S_3$)**: Unaligned sources containing noisy counterpart records.
- **Goal**: For each $S_1$ entity, determine all matching records from $S_2$ and $S_3$ (cardinality: $0$, $1$, or many).
- **Evaluation Metric**: **Macro-averaged $F_{0.5}$** (precision weighted $2\times$ over recall to heavily penalize false merges).
- **Country Generalization**: Open set handling (trains on US, India; test includes France).

---

## 🏗️ Architecture & Pipeline

```
[ Raw Ingestion (S1, S2, S3) ]
             │
             ▼
[ Robust Preprocessing & Normalization ]
  • Country-aware text cleaning & accent removal
  • Legal suffix harmonization (Corp, Ltd, SARL, LLC, Pvt Ltd)
  • Address standardization & numeric tokenization
             │
             ▼
[ Multi-Key Inverted-Index Blocking Engine ]
  • Exact country partitioning
  • Inverted index on high-IDF name and address tokens
  • Produces candidate_pairs.tsv
             │
             ▼
[ Precision-Calibrated Pairwise Matcher ]
  • String similarity metrics (Levenshtein, Jaro-Winkler, Token Sort)
  • Address alignment & numeric token overlap
  • Precision-tuned thresholding for F_0.5 optimization
             │
             ▼
[ Formatter & Validator ]
  • candidate_pairs.tsv (blocking candidates audit)
  • matching_results.tsv (final scored matches)
```

---

## 📂 Repository Structure

```
├── .gitignore
├── LICENSE
├── README.md
├── Documentation_template.md             # Detailed methodology & architectural report
├── amazon_ml_challenge_problem_statement.pdf # Official challenge problem statement
├── utils/
│   └── validate_submission.py           # Submission format validator (zero-dependency)
├── output/
│   └── .gitkeep                         # Output folder for submission TSV files
└── code/
    └── business_entity_resolution/
        ├── README.md                    # Reproduction guide
        ├── requirements.txt             # Pinned dependencies
        └── src/
            ├── __init__.py
            ├── config.py                # Hyperparameters & path configurations
            ├── preprocess.py            # Text normalization & legal suffix cleaning
            ├── blocking.py              # Candidate generation & inverted index
            ├── features.py              # Pairwise similarity feature extraction
            ├── matcher.py               # Precision thresholding & candidate scoring
            ├── evaluate.py              # Macro-averaged F_0.5 evaluation metric
            └── pipeline.py              # End-to-end execution runner
```
---

## 📊 Evaluation Metric ($F_{0.5}$)

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Singletons (0 matches)**: Correctly predicting an empty list yields a score of `1.0`. Any false positive yields `0.0`.
- **Non-Singletons**: Precision is weighted $2\times$ over recall.

---

## 📜 Academic Integrity & Fair Play

- No external database lookups or geocoding APIs.
- Self-contained open-source models ($\le 8\text{B}$ parameters) under MIT / Apache 2.0 licenses.
