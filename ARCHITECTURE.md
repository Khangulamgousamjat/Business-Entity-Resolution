# Business Entity Resolution — Architectural Specification & Methodology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Multi-Stage ER](https://img.shields.io/badge/Architecture-Multi--Stage%20ER-brightgreen.svg)](#)

## 1. Executive Summary & Problem Formulation

In modern enterprise architectures, customer, vendor, and business identity data originates across disparate, heterogeneous systems (e.g., ERP systems, CRM platforms, supplier master databases, national corporate registries). These records lack universal primary keys, feature substantial typographical and phonetic divergence, contain missing fields, and use variable legal entity structures.

**Entity Resolution (ER)** is the task of identifying and linking records across different sources that correspond to the same real-world entity.

### Problem Characteristics
- **Reference Source ($S_1$)**: Deduplicated reference entity registry.
- **Unstructured / Counterpart Sources ($S_2, S_3$)**: Unaligned, noisy feeds containing potential counterpart records.
- **Cardinality**: $0, 1$, or multiple counterpart records matching a given $S_1$ entity. Entities with $0$ matches are defined as **singletons**.
- **Geographic Generalization**: Robustness to multi-country regimes and unseen geographic distributions (e.g., US, India, France).
- **Asymmetric Business Risk**: In entity resolution, merging two distinct businesses (false positive) introduces catastrophic data corruption, whereas omitting a weak link (false negative) is much less harmful. Consequently, performance is evaluated using **Macro-averaged $F_{0.5}$**, where **Precision is weighted $2\times$ over Recall**.

---

## 2. End-to-End System Architecture

The system implements a four-stage decoupled pipeline designed for extreme precision, high recall, and bounded memory utilization:

```
[ Multi-Source Ingestion: S1, S2, S3 ]
                   │
                   ▼
[ Stage 1: Domain-Aware Preprocessing & Normalization ]
  ├── Character Canonicalization (Unicode NFKD, Diacritic Removal)
  ├── Multilingual Legal Entity Suffix Normalization (LLC, Pvt Ltd, SARL, etc.)
  ├── Address Standardization & Street/Avenue Abbreviation Expansion
  └── Invariant Numeric Token Extraction (PIN/ZIP codes, street numbers)
                   │
                   ▼
[ Stage 2: Multi-Key Inverted-Index Blocking Engine ]
  ├── Strict Country Partitioning (Zero cross-country noise)
  ├── High-IDF Name Anchor Inverted Index
  ├── Address Anchor & Geographic Token Co-occurrence
  └── Candidate Pruning (> 99.9% search space reduction -> candidate_pairs.tsv)
                   │
                   ▼
[ Stage 3: Multi-Modal Feature Extraction ]
  ├── Orthographic & String Distance Metrics (Levenshtein, Jaro-Winkler)
  ├── Permutation-Invariant Token Set & Token Sort Similarities
  ├── Character 3-Gram TF-IDF Vector Space Cosine Similarity
  ├── Address Alignment & Numeric Invariant Jaccard Overlap
  └── Source-Specific Discrepancy & Token Length Features
                   │
                   ▼
[ Stage 4: Precision-Calibrated GBDT Matching & Decision Boundary ]
  ├── LightGBM Gradient Boosted Decision Tree Classifier
  ├── Asymmetric F_0.5 Threshold Calibration
  └── Singleton Guardrail & Post-Processing Filter -> matching_results.tsv
```

---

## 3. Preprocessing & Data Normalization

Raw text in real-world business directories contains non-standard casing, punctuation anomalies, legal entity abbreviations, and diacritics. Stage 1 executes deterministic, language-aware normalization:

### 3.1 Legal Suffix Canonicalization
Corporate entity suffixes vary widely across jurisdictions. Failure to normalize suffixes causes false negatives (e.g., "Google Inc." vs "Google LLC") or false positive matches driven solely by shared suffixes.

- **Anglo-American**: `inc`, `incorporated`, `corp`, `corporation`, `ltd`, `limited`, `co`, `company`, `llc`, `pllc`.
- **Indian**: `pvt ltd`, `private limited`, `llp`, `enterprises`, `traders`, `industries`.
- **Continental European / French**: `sarl`, `sas`, `sasu`, `eurl`, `sa`, `sci`.

The normalization engine separates the core distinctive entity stem from the legal suffix, allowing similarity metrics to evaluate both the distinctive stem and the jurisdictional descriptor.

### 3.2 Address Normalization & Numeric Anchors
Addresses exhibit severe formatting differences (e.g., "123 Main St., Apt 4B" vs "Apt 4B, 123 Main Street").
- **Abbreviation Mapping**: Expands/harmonizes street types (`st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `fl` $\rightarrow$ `floor`, `bldg` $\rightarrow$ `building`).
- **Numeric Extraction**: Street numbers, unit/suite numbers, and postal codes (e.g., 5-digit US ZIP, 6-digit Indian PIN) are extracted as discrete numeric invariant sets. Two entities cannot match if their primary postal numbers conflict.

---

## 4. Multi-Key Inverted-Index Blocking Engine

Comparing every reference entity in $S_1$ ($N \approx 2.2 \times 10^6$) against all candidates in $S_2$ and $S_3$ ($M \approx 4.2 \times 10^6$) requires $O(N \times M) \approx 9.2 \times 10^{12}$ pairwise evaluations, which is computationally intractable.

The blocking engine achieves $>99.9\%$ candidate reduction ratio while retaining $>99\%$ candidate recall:

1. **Exact Country Sharding**: Entities are strictly partitioned by country. Cross-country comparisons are bypassed entirely.
2. **Inverted Token Indexing**: High-IDF distinctive name tokens (length $\ge 3$, non-stopwords) are mapped to entity candidate lists.
3. **Compound Key Generation**:
   - First 2 tokens of the business stem.
   - Longest distinctive token in the business name.
   - Distinctive numeric anchor (e.g., postal code) combined with initial character shingles.
4. **Candidate Pruning & Bounding**: Top-$K$ candidate pool per reference entity (default $K=15$) to constrain memory and guarantee predictable downstream inference runtime. All candidate pairs are persisted into `candidate_pairs.tsv` for lineage auditing.

---

## 5. Multi-Modal Feature Engineering

For each candidate pair $(e_{S1}, e_{cand})$, a 19-dimensional feature vector is generated spanning orthographic, phonetic, token-set, and structural modalities:

| Feature Name | Category | Description | Formula / Mechanism |
| :--- | :--- | :--- | :--- |
| `name_ratio` | Orthographic | Normalized Levenshtein similarity | $1 - \frac{\text{Lev}(s_1, s_2)}{\max(|s_1|, |s_2|)}$ |
| `name_jaro_winkler` | Prefix / Typo | Jaro-Winkler distance | Prioritizes common prefixes and typo resilience |
| `name_token_sort` | Permutation | Token Sort Ratio | Levenshtein over alphabetically sorted tokens |
| `name_token_set` | Subset | Token Set Ratio | Intersection vs. remainder token similarity |
| `name_token_jaccard` | Set-theoretic | Jaccard similarity of name words | $\frac{\|T_1 \cap T_2\|}{\|T_1 \cup T_2\|}$ |
| `name_ngram_sim` | Substring | Character 3-gram cosine similarity | Sublinear TF-IDF character shingle overlap |
| `name_len_diff` | Structural | Absolute token length differential | $\| |s_1| - |s_2| \|$ |
| `addr_ratio` | Address | Address Levenshtein similarity | Normalized sequence ratio on address string |
| `addr_token_sort` | Address | Address Token Sort Ratio | Order-invariant address component alignment |
| `addr_token_jaccard` | Address | Jaccard overlap on address tokens | Shared address keywords (street, city, area) |
| `addr_ngram_sim` | Address | Character 3-gram address cosine | High-frequency character shingle match |
| `addr_len_diff` | Address | Absolute address length differential | Identifies mismatched premises/buildings |
| `addr_missing` | Indicator | Missing address indicator flag | Binary flag ($1$ if either address is null) |
| `num_jaccard` | Numeric | Numeric token Jaccard overlap | Overlap ratio of extracted digits/codes |
| `num_match_count` | Numeric | Count of identical numeric anchors | Direct count of matching postal/building numbers |
| `num_conflict` | Numeric | Numeric contradiction indicator | Binary flag ($1$ if conflicting postal/street numbers) |
| `target_is_s3` | Metadata | Target source origin | Binary flag ($0$ for $S_2$, $1$ for $S_3$) |

---

## 6. Model Architecture & Decision Calibration

### 6.1 Gradient Boosted Decision Tree (LightGBM)
The pairwise matching engine uses a LightGBM GBDT classifier tuned for fast parallel inference and high tabular discrimination:
- **Number of Estimators**: 300
- **Maximum Tree Depth**: 6
- **Learning Rate**: 0.05
- **Objective**: Binary Logloss with balanced class weights.

### 6.2 Asymmetric Loss & Optimal Decision Boundary ($\tau^*$)

The evaluation criterion is **Macro-averaged $F_{0.5}$**:

$$F_{0.5} = \frac{(1 + \beta^2) \times \text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} \quad (\text{with } \beta = 0.5)$$

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

#### Singleton Sensitivity
- **Ground Truth $\emptyset$, Prediction $\emptyset$**: $F_{0.5} = 1.0$ (Correct singleton identification).
- **Ground Truth $\emptyset$, Prediction $\ne \emptyset$**: $F_{0.5} = 0.0$ (Catastrophic false positive).
- **Ground Truth $\ne \emptyset$, Prediction $\emptyset$**: $F_{0.5} = 0.0$ (Missed match).

Because singletons constitute a significant portion of real-world entities, and false-positive merges reduce an entity's score immediately to $0.0$, a standard classification threshold ($\tau = 0.5$) is often suboptimal.

Through fine-grained threshold sweeps on holdout validation data, the decision boundary $\tau^*$ is calibrated to maximize the expected macro $F_{0.5}$ score across all reference entities.

---

## 7. Lineage Verification & Output Auditing

The system enforces strict structural constraints on all generated outputs:
1. **Candidate Coverage Guarantee**: Every predicted match in `matching_results.tsv` is strictly audited to ensure it exists within `candidate_pairs.tsv`.
2. **1-to-1 Reference Mapping**: Every input $S_1$ entity ID appears exactly once in both output files.
3. **Deterministic Serialization**: Formatted as strict UTF-8 tab-separated values without BOM or trailing delimiter discrepancies.
