"""
Pairwise similarity feature extraction.
Computes string similarity, token overlap, and numeric alignment metrics.
Includes standard library fallbacks if rapidfuzz is not yet installed.
"""

from typing import Dict
from difflib import SequenceMatcher
from preprocess import clean_business_name, clean_address, extract_numbers, extract_tokens

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def jaccard_similarity(set_a: set, set_b: set) -> float:
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


def compute_pair_features(name_a: str, addr_a: str, name_b: str, addr_b: str) -> Dict[str, float]:
    """Compute vector of pairwise similarity metrics."""
    c_name_a, c_name_b = clean_business_name(name_a), clean_business_name(name_b)
    c_addr_a, c_addr_b = clean_address(addr_a), clean_address(addr_b)

    # Name similarities
    if RAPIDFUZZ_AVAILABLE:
        name_ratio = fuzz.ratio(c_name_a, c_name_b) / 100.0
        name_token_sort = fuzz.token_sort_ratio(c_name_a, c_name_b) / 100.0
        name_token_set = fuzz.token_set_ratio(c_name_a, c_name_b) / 100.0
        name_jaro_winkler = JaroWinkler.similarity(c_name_a, c_name_b)
        addr_ratio = fuzz.ratio(c_addr_a, c_addr_b) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(c_addr_a, c_addr_b) / 100.0
    else:
        name_ratio = SequenceMatcher(None, c_name_a, c_name_b).ratio()
        tokens_a_sorted = " ".join(sorted(c_name_a.split()))
        tokens_b_sorted = " ".join(sorted(c_name_b.split()))
        name_token_sort = SequenceMatcher(None, tokens_a_sorted, tokens_b_sorted).ratio()
        name_token_set = name_token_sort
        name_jaro_winkler = name_ratio
        addr_ratio = SequenceMatcher(None, c_addr_a, c_addr_b).ratio()
        addr_token_sort = addr_ratio

    # Token overlaps
    name_tokens_a = set(extract_tokens(c_name_a, min_len=2))
    name_tokens_b = set(extract_tokens(c_name_b, min_len=2))
    name_token_jaccard = jaccard_similarity(name_tokens_a, name_tokens_b)

    addr_tokens_a = set(extract_tokens(c_addr_a, min_len=3))
    addr_tokens_b = set(extract_tokens(c_addr_b, min_len=3))
    addr_token_jaccard = jaccard_similarity(addr_tokens_a, addr_tokens_b)

    # Numeric overlaps (PIN / house numbers)
    nums_a = extract_numbers(addr_a)
    nums_b = extract_numbers(addr_b)
    num_jaccard = jaccard_similarity(nums_a, nums_b)
    num_match_count = float(len(nums_a & nums_b))

    # N-gram similarities
    name_ngram_sim = char_ngram_jaccard(c_name_a, c_name_b, n=3)
    addr_ngram_sim = char_ngram_jaccard(c_addr_a, c_addr_b, n=3)

    # Composite match confidence score (calibrated for high precision)
    composite_score = (
        0.35 * name_token_sort
        + 0.25 * name_jaro_winkler
        + 0.15 * addr_token_jaccard
        + 0.15 * addr_ratio
        + 0.10 * num_jaccard
    )

    return {
        "name_ratio": name_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jaro_winkler": name_jaro_winkler,
        "name_token_jaccard": name_token_jaccard,
        "name_ngram_sim": name_ngram_sim,
        "addr_ratio": addr_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_ngram_sim": addr_ngram_sim,
        "num_jaccard": num_jaccard,
        "num_match_count": num_match_count,
        "composite_score": composite_score,
    }
