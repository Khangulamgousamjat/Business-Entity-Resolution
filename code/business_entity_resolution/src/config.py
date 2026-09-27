"""
Configuration and constants for Business Entity Resolution.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# Model hyperparameters
CANDIDATE_POOL_SIZE = 15
MATCH_THRESHOLD = 0.82  # Precision-heavy threshold for F_0.5
MIN_TOKEN_LEN = 3

# Random seed for reproducibility
SEED = 42
