"""
Pairwise similarity feature extraction.
Computes string distance metrics, token overlap, and numeric alignment.
Leverages RapidFuzz for maximum C-level performance.
"""

from typing import Dict, List, Set
import re
from preprocess import clean_business_name, clean_address, extract_numbers, extract_tokens
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_ngram_sim",
    "name_len_diff",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_jaccard",
    "addr_ngram_sim",
    "addr_len_diff",
    "addr_missing",
    "num_jaccard",
    "num_match_count",
    "num_conflict",
    "target_is_s3",
]


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def char_ngram_jaccard(str_a: str, str_b: str, n: int = 3) -> float:
    if not str_a or not str_b:
        return 0.0
    ngrams_a = {str_a[i:i+n] for i in range(len(str_a) - n + 1)} if len(str_a) >= n else {str_a}
    ngrams_b = {str_b[i:i+n] for i in range(len(str_b) - n + 1)} if len(str_b) >= n else {str_b}
    return jaccard_similarity(ngrams_a, ngrams_b)


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    cand_id: str = ""
) -> List[float]:
    """Compute feature vector for a candidate pair."""
    c_s1_name, c_cand_name = clean_business_name(s1_name), clean_business_name(cand_name)
    c_s1_addr, c_cand_addr = clean_address(s1_addr), clean_address(cand_addr)

    # Name similarity metrics
    name_ratio = fuzz.ratio(c_s1_name, c_cand_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(c_s1_name, c_cand_name) / 100.0
    name_token_set = fuzz.token_set_ratio(c_s1_name, c_cand_name) / 100.0
    name_jaro_winkler = JaroWinkler.similarity(c_s1_name, c_cand_name)

    name_tokens_a = set(extract_tokens(c_s1_name, min_len=2))
    name_tokens_b = set(extract_tokens(c_cand_name, min_len=2))
    name_token_jaccard = jaccard_similarity(name_tokens_a, name_tokens_b)
    name_ngram_sim = char_ngram_jaccard(c_s1_name, c_cand_name, n=3)
    name_len_diff = abs(len(c_s1_name) - len(c_cand_name)) / (max(len(c_s1_name), len(c_cand_name)) + 1e-5)

    # Address similarity metrics
    addr_missing = 1.0 if (not c_s1_addr or not c_cand_addr) else 0.0
    if addr_missing:
        addr_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_jaccard = 0.0
        addr_ngram_sim = 0.0
        addr_len_diff = 1.0
    else:
        addr_ratio = fuzz.ratio(c_s1_addr, c_cand_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(c_s1_addr, c_cand_addr) / 100.0
        addr_tokens_a = set(extract_tokens(c_s1_addr, min_len=3))
        addr_tokens_b = set(extract_tokens(c_cand_addr, min_len=3))
        addr_token_jaccard = jaccard_similarity(addr_tokens_a, addr_tokens_b)
        addr_ngram_sim = char_ngram_jaccard(c_s1_addr, c_cand_addr, n=3)
        addr_len_diff = abs(len(c_s1_addr) - len(c_cand_addr)) / (max(len(c_s1_addr), len(c_cand_addr)) + 1e-5)

    # Numeric alignment (PIN codes, house numbers)
    nums_a = extract_numbers(s1_addr)
    nums_b = extract_numbers(cand_addr)
    num_jaccard = jaccard_similarity(nums_a, nums_b)
    num_match_count = float(len(nums_a & nums_b))
    num_conflict = 1.0 if (nums_a and nums_b and len(nums_a & nums_b) == 0) else 0.0

    # Source identity flag
    target_is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return [
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jaro_winkler,
        name_token_jaccard,
        name_ngram_sim,
        name_len_diff,
        addr_ratio,
        addr_token_sort,
        addr_token_jaccard,
        addr_ngram_sim,
        addr_len_diff,
        addr_missing,
        num_jaccard,
        num_match_count,
        num_conflict,
        target_is_s3,
    ]
