"""Central configuration. Every threshold, budget, weight, and the model name
live here. No magic numbers anywhere else in the codebase (CLAUDE.md hard rule).

Tune the system by editing this file, not by hunting through the source.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

SAMPLE_CSV = DATA_DIR / "sample.csv"
RAW_TRAIN_CSV = DATA_DIR / "raw" / "train.csv"
WATCHLIST_FILE = DATA_DIR / "watchlist.txt"

VERDICTS_FILE = OUTPUTS_DIR / "verdicts.jsonl"
CASSETTE_FILE = OUTPUTS_DIR / "cassette.jsonl"

# --- Sampling (one-time, builds data/sample.csv) -------------------------
# Keep fraud at its natural low rate; sample is stratified so the rare
# positive class is represented at roughly the population proportion.
SAMPLE_SIZE = 50_000
SAMPLE_SEED = 42

# --- Outer-loop triage thresholds ---------------------------------------
# below LOW            -> auto-clear, no LLM
# LOW <= score < HIGH  -> dispatch to inner loop for investigation
# score >= HIGH        -> auto-flag, still investigated for an explanation
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.80

# --- Rule weights (outer loop) ------------------------------------------
# rule_score is a weighted sum of normalized signal contributions, clipped
# to 0..1. Weights are the knobs for the deterministic detector.
RULE_WEIGHTS = {
    "amount_anomaly": 0.30,      # amount far above the account's own mean
    "velocity_1h": 0.20,         # bursty spending in the last hour
    "velocity_24h": 0.10,        # elevated daily count
    "impossible_travel": 0.25,   # geographically impossible vs prior txn
    "high_risk_category": 0.10,  # merchant category with elevated fraud rate
    "odd_hour": 0.05,            # transaction in the small hours
    "watchlist": 1.00,           # known-fraud card: saturates the score
}

# Signal normalization constants (deterministic, cheap).
AMOUNT_ANOMALY_FULL_Z = 4.0      # z-score that maps to full amount contribution
VELOCITY_1H_FULL = 5             # txns/hour that maps to full 1h contribution
VELOCITY_24H_FULL = 20           # txns/day that maps to full 24h contribution
IMPOSSIBLE_TRAVEL_KMH = 900.0    # implied speed above this is "impossible"
ODD_HOUR_START = 1               # inclusive, local hour
ODD_HOUR_END = 5                 # inclusive, local hour
HIGH_RISK_CATEGORIES = {
    "shopping_net",
    "misc_net",
    "grocery_pos",
}

# --- Velocity windows ----------------------------------------------------
VELOCITY_WINDOW_1H_SECONDS = 3600
VELOCITY_WINDOW_24H_SECONDS = 86_400

# --- Inner-loop agent budget --------------------------------------------
MAX_ITERATIONS = 5               # hard cap on ReAct turns; never loop unbounded
TOKEN_BUDGET_PER_INVESTIGATION = 8_000  # cumulative output-token ceiling
MAX_TOKENS_PER_CALL = 1_024      # per single model call
JSON_PARSE_RETRIES = 1           # retry once on bad JSON, then default to review_queue

# --- Model selection -----------------------------------------------------
# Default to Haiku for minimal spend; Sonnet selectable for harder cases.
MODEL_DEFAULT = "claude-haiku-4-5-20251001"
MODEL_STRONG = "claude-sonnet-4-6"
MODEL = MODEL_DEFAULT

# --- Tool defaults -------------------------------------------------------
ACCOUNT_HISTORY_LOOKBACK_DAYS = 30

# --- Agent action / label mapping ---------------------------------------
# Used by the mock agent and as the fallback when the agent gives up.
DEFAULT_FALLBACK_ACTION = "review_queue"
DEFAULT_FALLBACK_LABEL = "suspicious"
