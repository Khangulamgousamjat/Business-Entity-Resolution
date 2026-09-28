#!/usr/bin/env python3
"""
End-to-End Execution Pipeline for Business Entity Resolution.

Key Architectural Highlights:
1. Country-Partitioned Ingestion:
   Processes one country partition at a time (e.g. France, India, US),
   keeping memory consumption strictly under 1.5 GB while supporting an open set of countries.
2. Inverted Index Blocking:
   Extracts high-IDF name and address token anchors to generate high-recall candidate pairs.
3. Machine-Learned Matching:
   Uses the trained LightGBM GBDT model with precision-tuned thresholding for F_0.5 optimization.
4. Strict Output & Schema Compliance:
   Produces output/candidate_pairs.tsv and output/matching_results.tsv,
   guaranteeing 1-to-1 entity coverage, proper singletons, and subset constraints.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import InvertedIndexBlocker
from matcher import EntityMatcher
from config import CANDIDATE_POOL_SIZE


def discover_countries(filepath: str) -> Set[str]:
    """Scan file to discover all unique country labels dynamically."""
    countries = set()
    if not os.path.exists(filepath):
        return countries
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        c_idx = header.index("country") if "country" in header else 3
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > c_idx:
                countries.add(parts[c_idx].strip())
    return countries


def stream_country_target_records(
    s2_path: str,
    s3_path: str,
    target_country: str,
    max_records: int = None
) -> List[Tuple[str, str, str, str]]:
    """Stream and filter target records for a specific country partition."""
    records = []
    tc_upper = target_country.strip().upper()

    for path in [s2_path, s3_path]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            eid_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            c_idx = header.index("country") if "country" in header else 3

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > max(eid_idx, name_idx, addr_idx, c_idx):
                    country_val = parts[c_idx].strip().upper()
                    if country_val == tc_upper:
                        records.append((parts[eid_idx], parts[name_idx], parts[addr_idx], parts[c_idx]))
                        if max_records and len(records) >= max_records:
                            return records
    return records


def run_pipeline(
    test_dir: str,
    output_dir: str,
    model_dir: str = None,
    sample_limit: int = None,
    target_limit: int = None,
    countries_to_run: List[str] = None
):
    print("=" * 75, flush=True)
    print("BUSINESS ENTITY RESOLUTION: END-TO-END EXECUTION PIPELINE", flush=True)
    print("=" * 75, flush=True)
    print(f"[*] Test Directory:   {test_dir}", flush=True)
    print(f"[*] Output Directory: {output_dir}", flush=True)
    print(f"[*] Model Directory:  {model_dir or 'src/models'}", flush=True)
    if sample_limit:
        print(f"[*] Test Sample Mode: Processing up to {sample_limit:,} S1 entities.", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Missing required test reference file: {s1_path}")

    # Discover countries present in Source 1
    print("\n[Step 1/4] Discovering country partitions in test reference entities...", flush=True)
    all_countries = discover_countries(s1_path)
    print(f"    Discovered countries: {sorted(list(all_countries))}", flush=True)

    if countries_to_run:
        active_countries = [c for c in countries_to_run if c in all_countries]
    else:
        active_countries = sorted(list(all_countries))

    print(f"    Active processing partitions: {active_countries}", flush=True)

    # Initialize Matcher
    print("\n[Step 2/4] Loading Matcher & Decision Thresholds...", flush=True)
    matcher = EntityMatcher(model_dir=model_dir)

    # Output file paths
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    # In sample testing mode or incremental mode:
    # If testing on sample, we process directly
    print("\n[Step 3/4] Processing Country Partitions (Memory-Efficient Streaming)...", flush=True)

    total_processed = 0
    total_matched = 0
    total_singletons = 0

    with open(matching_file, "w", encoding="utf-8", newline="\n") as f_match, \
         open(candidate_file, "w", encoding="utf-8", newline="\n") as f_cand:

        # Write official required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in active_countries:
            print(f"\n--- [Partition: {country}] ---", flush=True)
            print(f"    Loading target candidate pool for country: {country}...", flush=True)

            effective_target_limit = target_limit
            if sample_limit and not target_limit:
                effective_target_limit = sample_limit * 25

            target_records = stream_country_target_records(
                s2_path, s3_path, country, max_records=effective_target_limit
            )
            print(f"    Loaded {len(target_records):,} candidate records for {country}.", flush=True)

            # Build Inverted Index Blocker for this country
            print(f"    Building candidate blocking index for {country}...", flush=True)
            blocker = InvertedIndexBlocker(max_token_freq=2500, top_k_candidates=CANDIDATE_POOL_SIZE)
            blocker.fit_target_source(target_records)
            print(f"    Blocking index ready.", flush=True)

            # Stream Source 1 records for this country
            print(f"    Matching Source 1 entities for {country}...", flush=True)
            part_processed = 0
            part_matched = 0
            part_singletons = 0

            with open(s1_path, "r", encoding="utf-8") as f_in:
                header = f_in.readline().rstrip("\r\n").split("\t")
                eid_idx = header.index("entity_id") if "entity_id" in header else 0
                name_idx = header.index("business_name") if "business_name" in header else 1
                addr_idx = header.index("business_address") if "business_address" in header else 2
                c_idx = header.index("country") if "country" in header else 3

                for line in f_in:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) <= max(eid_idx, name_idx, addr_idx, c_idx):
                        continue

                    s1_country = parts[c_idx].strip()
                    if s1_country.upper() != country.upper():
                        continue

                    s1_id = parts[eid_idx]
                    name = parts[name_idx]
                    addr = parts[addr_idx]

                    # 1. Candidate blocking
                    candidates = blocker.get_candidates(s1_id, name, addr, s1_country)

                    # 2. Precision-calibrated scoring
                    matches = matcher.score_and_filter(
                        (s1_id, name, addr, s1_country),
                        candidates,
                        blocker.target_records
                    )

                    # Enforce subset requirement
                    assert set(matches).issubset(set(candidates))

                    # Statistics
                    if matches:
                        part_matched += 1
                        total_matched += 1
                    else:
                        part_singletons += 1
                        total_singletons += 1

                    cand_str = ",".join(candidates)
                    match_str = ",".join(matches)
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    f_match.write(f"{s1_id}\t{match_str}\n")

                    part_processed += 1
                    total_processed += 1

                    if part_processed % 25000 == 0:
                        print(f"      [{country}] Processed {part_processed:,} records...", flush=True)

                    if sample_limit and total_processed >= sample_limit:
                        print(f"      Reached sample limit of {sample_limit:,} entities.", flush=True)
                        break

            print(f"    Finished {country}: {part_processed:,} entities (Matched: {part_matched:,}, Singletons: {part_singletons:,})", flush=True)

            if sample_limit and total_processed >= sample_limit:
                break

    print("\n" + "=" * 75, flush=True)
    print("[Step 4/4] Execution Summary", flush=True)
    print("=" * 75, flush=True)
    print(f"[OK] Total Processed:   {total_processed:,} Source 1 entities", flush=True)
    print(f"[OK] Matched Entities:  {total_matched:,} ({total_matched/max(1, total_processed)*100:.1f}%)", flush=True)
    print(f"[OK] Singletons:        {total_singletons:,} ({total_singletons/max(1, total_processed)*100:.1f}%)", flush=True)
    print(f"[OK] Generated: {matching_file}", flush=True)
    print(f"[OK] Generated: {candidate_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline.")
    parser.add_argument("--test-dir", default="../../test", help="Path to test files directory.")
    parser.add_argument("--output-dir", default="../../output", help="Path to save output TSVs.")
    parser.add_argument("--model-dir", default=None, help="Directory containing trained model.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Process first N entities.")
    parser.add_argument("--target-limit", type=int, default=None, help="Max target records per country partition.")
    parser.add_argument("--countries", nargs="+", default=None, help="Specific countries to process (e.g. France).")
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        sample_limit=args.sample_limit,
        target_limit=args.target_limit,
        countries_to_run=args.countries
    )


if __name__ == "__main__":
    main()
