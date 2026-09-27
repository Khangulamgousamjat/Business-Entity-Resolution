#!/usr/bin/env python3
"""
End-to-End Execution Pipeline for Amazon ML Challenge: Business Entity Resolution.
Ingests test data, runs blocking, extracts features, performs precision scoring,
and generates both candidate_pairs.tsv and matching_results.tsv.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import InvertedIndexBlocker
from matcher import EntityMatcher
from config import MATCH_THRESHOLD, CANDIDATE_POOL_SIZE


def read_source_tsv(filepath: str) -> List[Tuple[str, str, str, str]]:
    """Read source TSV file safely line-by-line."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        country_idx = header.index("country") if "country" in header else 3

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > max(eid_idx, name_idx, addr_idx, country_idx):
                records.append((
                    parts[eid_idx],
                    parts[name_idx],
                    parts[addr_idx],
                    parts[country_idx]
                ))
    return records


def run_pipeline(test_dir: str, output_dir: str, sample_limit: int = None):
    print(f"[*] Starting Business Entity Resolution pipeline...")
    print(f"    Test Directory:   {test_dir}")
    print(f"    Output Directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("[*] Loading Source 2 and Source 3 target records...")
    target_records = []
    if os.path.exists(s2_path):
        target_records.extend(read_source_tsv(s2_path))
    if os.path.exists(s3_path):
        target_records.extend(read_source_tsv(s3_path))
    print(f"    Loaded {len(target_records)} total target candidate pool records.")

    print("[*] Initializing and building candidate blocking index...")
    blocker = InvertedIndexBlocker(max_token_freq=2000, top_k_candidates=CANDIDATE_POOL_SIZE)
    blocker.fit_target_source(target_records)
    print("    Blocking index built successfully.")

    matcher = EntityMatcher(threshold=MATCH_THRESHOLD)

    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"[*] Processing Source 1 reference entities...")
    with open(s1_path, "r", encoding="utf-8") as f_in, \
         open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:

        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        header = f_in.readline().rstrip("\r\n").split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        country_idx = header.index("country") if "country" in header else 3

        count = 0
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or len(parts) <= max(eid_idx, name_idx, addr_idx, country_idx):
                continue

            s1_id = parts[eid_idx]
            name = parts[name_idx]
            addr = parts[addr_idx]
            country = parts[country_idx]

            # 1. Candidate blocking
            candidates = blocker.get_candidates(s1_id, name, addr, country)

            # 2. Pairwise scoring & precision filtering
            matches = matcher.score_and_filter((s1_id, name, addr, country), candidates, blocker.target_records)

            # Ensure matches is a strict subset of candidates
            assert set(matches).issubset(set(candidates))

            # Write rows (single tab separator, comma-separated IDs, empty for singletons)
            f_cand.write(f"{s1_id}\t{','.join(candidates)}\n")
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")

            count += 1
            if count % 100000 == 0:
                print(f"    Processed {count:,} entities...")

            if sample_limit and count >= sample_limit:
                print(f"    Reached sample limit of {sample_limit} records.")
                break

    print(f"[✓] Pipeline complete! Processed {count} Source 1 entities.")
    print(f"    Saved: {matching_file}")
    print(f"    Saved: {candidate_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline.")
    parser.add_argument("--test-dir", default="../../test", help="Path to test files directory.")
    parser.add_argument("--output-dir", default="../../output", help="Path to save output TSVs.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Process first N entities (for testing).")
    args = parser.parse_args()

    run_pipeline(args.test_dir, args.output_dir, args.sample_limit)


if __name__ == "__main__":
    main()
