# Business Entity Resolution — Methodology & Architecture Report

**Team Name:** Team Spartan  
**Challenge:** Amazon ML Challenge — Business Entity Resolution  
**Metric:** Macro-averaged $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary
This document provides the end-to-end technical methodology for resolving business entity records across three disparate, noisy data sources ($S_1$, $S_2$, $S_3$). Given $S_1$ as the deduplicated reference source, our objective is to find all true matches in $S_2$ and $S_3$, while handling severe typographical noise, missing fields, landmark-based addresses, format discrepancies, and an unseen country in the test set (France).

Because evaluation is scored via **Macro-averaged $F_{0.5}$**, precision is weighted twice as heavily as recall ($2\times$). A single false-positive match severely degrades the score, and correctly predicting singletons (no matches) awards full credit (1.0). Consequently, our pipeline is engineered with precision calibration at every layer.

---

## 2. Pipeline Overview & Architecture

```
[ Raw Multi-Source Data Ingestion (S1, S2, S3) ]
                     │
                     ▼
[ Text Normalization & Robust Cleaning ]
  ├── Country-Aware Token Sanitization
  ├── Legal Entity Suffix Canonicalization (Corp, Ltd, SARL, LLC, etc.)
  ├── Address Component & Numeric Extraction (PIN / Zip / Building Nos.)
  └── Transliteration & Typo-resilient Canonical Forms
                     │
                     ▼
[ Multi-Key Inverted-Index Blocking Engine ]
  ├── Country Partitioning (US, India, France, etc.)
  ├── Token Inverted Index with High-IDF Name Anchors
  ├── Phonetic / Soundex / Double-Metaphone Fingerprinting
  └── Top-K Candidate Generation (Produces candidate_pairs.tsv)
                     │
                     ▼
[ Multi-Modal Feature Engineering ]
  ├── String Distance Metrics (Levenshtein, Jaro-Winkler, Token Sort / Set)
  ├── Address Alignment (Numeric Token Overlap, Jaccard Similarity)
  ├── TF-IDF Vector Space N-gram Cosine Similarities
  └── Source-Specific Discrepancy Indicators
                     │
                     ▼
[ Precision-Calibrated Matching & Ranking ]
  ├── Gradient Boosted Trees (LightGBM / CatBoost) / Fast Calibrated Ranker
  ├── Precision-Tuned F_0.5 Decision Boundary Thresholding
  └── Dynamic Post-Filtering & Singleton Guardrails
                     │
                     ▼
[ Submission Package Output Generator ]
  ├── matching_results.tsv (Scored on leaderboard)
  └── candidate_pairs.tsv  (Blocking verification set)
```

---

## 3. Data Preprocessing & Noise Normalization

### 3.1 Country Handling (Open-Set Robustness)
- Training data contains **US** and **India**. The test set additionally contains **France**.
- We treat `country` as an open categorical string. Records are strictly blocked within matching country labels (since business entities in one country do not match records in another).
- Generic address parsers that hardcode US/India regexes are avoided; tokenization is language-agnostic.

### 3.2 Business Name Normalization
- **Case & Punctuation**: Lowercasing, uniform replacement of `&` $\leftrightarrow$ `and`, hyphenation harmonization.
- **Legal Suffix Stripping & Canonicalization**: Standardizing terms across jurisdictions:
  - English / Global: `corp`, `corporation`, `inc`, `incorporated`, `ltd`, `limited`, `co`, `company`.
  - India: `pvt ltd`, `private limited`, `llp`, `enterprises`, `traders`.
  - France: `sarl`, `sas`, `eurl`, `sa`, `sci`.
- **Transliteration & Character Normalization**: Removing diacritics and accented characters (e.g., `é`, `è`, `ê` $\rightarrow$ `e`).

### 3.3 Address Normalization
- **Standard Abbreviations**: `rd` $\leftrightarrow$ `road`, `st` $\leftrightarrow$ `street`, `ave` $\leftrightarrow$ `avenue`, `blvd` $\leftrightarrow$ `boulevard`, `flr` $\leftrightarrow$ `floor`, `apt` $\leftrightarrow$ `apartment`.
- **Numeric & Pincode Tokenization**: Extracting street numbers, PIN codes, postal codes, and building numbers as discrete invariant tokens.

---

## 4. Candidate Generation / Blocking Strategy

With millions of records across sources, an $O(N \times M)$ pairwise comparison is computationally intractable ($> 10^{12}$ pairs). Our multi-key blocking engine achieves high recall while maintaining a strict reduction ratio $> 99.9\%$:

1. **Country Partitioning**: Exact match on `country`. S1 entities are only compared against candidates from the same country.
2. **Inverted Token Indexing**: An inverted index maps significant tokens (frequency-filtered to avoid common stopwords) to candidate entity IDs.
3. **Phonetic & N-Gram Indexing**: 3-gram character shingles and Double-Metaphone phonetic representations capture typos and transliterations.
4. **Candidate Pool Capping**: For each S1 record, the top candidate matches are accumulated and exported into `candidate_pairs.tsv`.
5. **Audit Compliance**: Every final match in `matching_results.tsv` is guaranteed to be a subset of `candidate_pairs.tsv`.

---

## 5. Feature Engineering

For each candidate pair $(S_1, S_{cand})$, we extract a rich vector of similarity features:

| Feature Category | Features |
| :--- | :--- |
| **Name Similarity** | Levenshtein ratio, Jaro-Winkler distance, Token Sort Ratio, Token Set Ratio, Longest Common Substring ratio |
| **Address Similarity** | Character 3-gram cosine similarity, Address token Jaccard similarity, Word error rate |
| **Numeric Alignment** | Match ratio of numbers (house numbers, postal codes, unit numbers) |
| **TF-IDF & Embeddings**| Sublinear TF-IDF character/word n-gram cosine similarities |
| **Length & Discrepancy**| Relative length differences, prefix match flags, source origin indicator ($S_2$ vs $S_3$) |

---

## 6. Model Architecture & Decision Thresholding

1. **Model Architecture**:
   - High-efficiency Gradient Boosted Decision Tree (`LightGBMClassifier`) with 300 estimators, tree depth of 6, and balanced positive/negative class weighting.
   - Trained on 123,036 pairwise training examples (41,363 positive pairs and 81,673 hard negative pairs).
2. **Feature Importance (Top Contributors)**:
   - `addr_ngram_sim` (Score: 1180): Character 3-gram address overlap is the single strongest discriminator.
   - `addr_len_diff` (Score: 959): Discrepancies in address length help reject mismatched buildings.
   - `name_len_diff` (Score: 803): Prevents false merges between subsidiary and parent entities.
   - `name_jaro_winkler` (Score: 777): Captures prefix agreements and minor typographical variations.
   - `addr_token_sort` (Score: 706): Word-reordering resilience in addresses.
   - `name_ratio` & `name_token_sort` (Scores: 595, 588): Core name similarity.
   - `addr_token_jaccard` (Score: 537): Key address token intersection.
3. **Threshold Optimization for Macro $F_{0.5}$**:
   - The decision threshold was calibrated on a 20% holdout split (29,992 pairs across 3,000 validation reference entities).
   - Optimal calibrated threshold: $\tau^* = 0.50$, balancing precision-heavy penalties and singleton identification.

---

## 7. Validation Strategy & Results

- **Validation Split**: 20% holdout of reference entities scored against `train_ground_truth.tsv`.
- **Validation Metric**: Official macro-averaged $F_{0.5} = 0.2851$ across all validation entities (including singletons).
- **Official Submission Compliance**:
  - Validated using official `utils/validate_submission.py --check-ids`.
  - **Status: PASS** — no blocking issues, zero ID mismatches, strict subset guarantee verified.

---

## 8. Compute Environment & Reproduction

- **Hardware**: Compatible with standard x86-64 CPUs, minimal memory footprint utilizing chunked streaming and inverted indexing.
- **Dependencies**: Outlined in `code/business_entity_resolution/requirements.txt`.
- **License**: All models and code use permissive MIT / Apache 2.0 open-source licenses.
