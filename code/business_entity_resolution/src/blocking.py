"""
Multi-Key Inverted Index Blocking Engine.
Partitions by country, builds inverted index on informative name and address tokens.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from preprocess import clean_business_name, clean_address, extract_tokens


class InvertedIndexBlocker:
    def __init__(self, max_token_freq: int = 1500, top_k_candidates: int = 20):
        self.max_token_freq = max_token_freq
        self.top_k = top_k_candidates
        # Partitioned index: country -> token -> list of target entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.target_records: Dict[str, Tuple[str, str]] = {}

    def fit_target_source(self, records: List[Tuple[str, str, str, str]]):
        """
        Build inverted index over target records (Source 2 and Source 3).
        records: list of (entity_id, business_name, business_address, country)
        """
        for eid, name, addr, country in records:
            country = (country or "UNKNOWN").strip().upper()
            c_name = clean_business_name(name)
            c_addr = clean_address(addr)
            self.target_records[eid] = (c_name, c_addr)

            # Extract distinct tokens for indexing
            name_tokens = set(extract_tokens(c_name, min_len=3))
            addr_tokens = set(extract_tokens(c_addr, min_len=4))
            all_tokens = name_tokens | addr_tokens

            for token in all_tokens:
                self.token_counts[country][token] += 1
                self.index[country][token].append(eid)

    def get_candidates(self, s1_id: str, name: str, addr: str, country: str) -> List[str]:
        """
        Retrieve candidate matching IDs for a given Source 1 entity.
        Scores candidate by token intersection weighted by inverse frequency.
        """
        country = (country or "UNKNOWN").strip().upper()
        country_index = self.index.get(country)
        if not country_index:
            return []

        c_name = clean_business_name(name)
        c_addr = clean_address(addr)

        name_tokens = extract_tokens(c_name, min_len=3)
        addr_tokens = extract_tokens(c_addr, min_len=4)

        candidate_scores: Dict[str, float] = defaultdict(float)

        # Name tokens provide higher blocking signal
        for token in set(name_tokens):
            freq = self.token_counts[country].get(token, 0)
            if 0 < freq <= self.max_token_freq:
                idf_weight = 1.0 / (freq ** 0.5)
                for cand_id in country_index[token]:
                    candidate_scores[cand_id] += idf_weight * 3.0

        # Address tokens provide secondary confirmation
        for token in set(addr_tokens):
            freq = self.token_counts[country].get(token, 0)
            if 0 < freq <= self.max_token_freq:
                idf_weight = 1.0 / (freq ** 0.5)
                for cand_id in country_index[token]:
                    candidate_scores[cand_id] += idf_weight * 1.0

        if not candidate_scores:
            return []

        # Sort candidate entities by score descending
        sorted_candidates = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)
        return sorted_candidates[:self.top_k]
