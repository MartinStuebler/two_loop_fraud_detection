"""Load and normalize the Sparkov dataset.

Two responsibilities:

1. build_sample(): a one-time job that turns the full raw Sparkov CSV into the
   committed data/sample.csv. It samples whole accounts (not random rows) so
   each card keeps its full transaction history. The velocity, account-baseline,
   and impossible-travel logic are meaningless without intact per-card history,
   so row-level sampling would quietly break the detector. Fraud is left at its
   natural low rate.

2. load(): read data/sample.csv and rename the Sparkov columns to the canonical
   names used everywhere else in the codebase (SPEC.md section 4).
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config

# Sparkov raw column -> canonical name. Geo is split into explicit lat/long
# numeric columns (one per endpoint) so haversine is a direct vectorized op.
COLUMN_RENAME = {
    "trans_num": "txn_id",
    "trans_date_trans_time": "timestamp",
    "cc_num": "account_id",
    "amt": "amount",
    "merchant": "merchant_id",
    "category": "merchant_category",
    "lat": "cardholder_lat",
    "long": "cardholder_long",
    "merch_lat": "merchant_lat",
    "merch_long": "merchant_long",
    "city_pop": "city_pop",
    "unix_time": "unix_time",
    "is_fraud": "is_fraud",
}

# Columns kept in the committed sample (the raw Sparkov names above).
KEEP_RAW_COLUMNS = list(COLUMN_RENAME.keys())


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to canonical columns and fix dtypes."""
    df = df.rename(columns=COLUMN_RENAME)
    # account_id as string: cc_num is a long integer and must not be treated as
    # a float (precision loss) or summed as a number.
    df["account_id"] = df["account_id"].astype(str)
    df["txn_id"] = df["txn_id"].astype(str)
    df["merchant_id"] = df["merchant_id"].astype(str)
    df["merchant_category"] = df["merchant_category"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["unix_time"] = df["unix_time"].astype("int64")
    # Chronological order: the orchestrator does a single forward pass and the
    # rolling per-account counters assume time-ordered input.
    df = df.sort_values("unix_time").reset_index(drop=True)
    return df


def load() -> pd.DataFrame:
    """Load the committed sample as a canonical dataframe."""
    if not config.SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"{config.SAMPLE_CSV} not found. Build it once with: "
            f"python -m src.data_loader --build-sample"
        )
    df = pd.read_csv(config.SAMPLE_CSV)
    return _canonicalize(df)


def build_sample() -> pd.DataFrame:
    """One-time: sample whole accounts from the raw CSV into data/sample.csv.

    Returns the canonical dataframe of the sample (also written to disk in raw
    Sparkov column form so load() can re-normalize it).
    """
    if not config.RAW_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {config.RAW_TRAIN_CSV}. Download it from "
            "the Hugging Face mirror pointe77/credit-card-transaction first."
        )

    raw = pd.read_csv(config.RAW_TRAIN_CSV, usecols=KEEP_RAW_COLUMNS)

    # Shuffle account ids deterministically, then take whole accounts until we
    # reach the target row count. Fraud is concentrated in time, not in a few
    # accounts, so whole-account sampling preserves the natural fraud rate well.
    account_sizes = raw.groupby("cc_num").size()
    shuffled_accounts = account_sizes.sample(
        frac=1.0, random_state=config.SAMPLE_SEED
    )
    cumulative = shuffled_accounts.cumsum()
    chosen_accounts = cumulative[cumulative <= config.SAMPLE_SIZE].index
    if len(chosen_accounts) == 0:
        # First account alone already exceeds the target; keep it anyway.
        chosen_accounts = shuffled_accounts.index[:1]

    sample = raw[raw["cc_num"].isin(chosen_accounts)].copy()
    # Persist in raw column form; load() applies the canonical rename.
    config.SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(config.SAMPLE_CSV, index=False)

    canonical = _canonicalize(sample)
    fraud_rate = canonical["is_fraud"].mean()
    print(
        f"Wrote {config.SAMPLE_CSV} : {len(canonical)} rows, "
        f"{canonical['account_id'].nunique()} accounts, "
        f"fraud rate {fraud_rate:.4%}"
    )
    return canonical


def seed_watchlist(df: pd.DataFrame, n_cards: int = 5) -> None:
    """Seed data/watchlist.txt with a few cc_num that appear in known-fraud rows.

    Gives check_watchlist real hits within the sample. One cc_num per line.
    """
    fraud_cards = (
        df.loc[df["is_fraud"] == 1, "account_id"].drop_duplicates().head(n_cards)
    )
    config.WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Known-fraud cc_num, one per line. Seeded from the sample."]
    lines += [str(c) for c in fraud_cards]
    config.WATCHLIST_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {config.WATCHLIST_FILE} : {len(fraud_cards)} cards")


if __name__ == "__main__":
    if "--build-sample" in sys.argv:
        canonical_df = build_sample()
        seed_watchlist(canonical_df)
    else:
        loaded = load()
        print(f"Loaded {len(loaded)} rows with columns: {list(loaded.columns)}")
