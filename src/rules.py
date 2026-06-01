"""Outer-loop triage: cheap, deterministic scoring. No LLM here.

This is the loop that runs on 100 percent of transactions, so everything must
be O(1) per txn using only signals we already have. Its whole job is to be cheap
enough to run everywhere and good enough to decide which tiny fraction is worth
spending an LLM agent on. The outer loop exists to protect the inner loop's
budget.

The score is a weighted sum of normalized signal contributions, clipped to
0..1. All weights and normalization constants live in config.py.
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from src import config
from src.schemas import TransactionSignals

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/long points."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def load_watchlist() -> set[str]:
    """Load known-fraud cc_num from the watchlist file as strings."""
    if not config.WATCHLIST_FILE.exists():
        return set()
    cards: set[str] = set()
    for line in config.WATCHLIST_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cards.add(line)
    return cards


class AccountState:
    """Rolling per-account history, holding only PRIOR transactions.

    The orchestrator updates this after scoring each txn, so when rules read it
    the state reflects everything seen before the current transaction and not
    the transaction itself. That ordering is what makes velocity and
    impossible-travel causal rather than leaking the present.
    """

    def __init__(self) -> None:
        self.count = 0
        self.amount_sum = 0.0
        self.amount_sumsq = 0.0
        self.recent_times: deque[int] = deque()  # unix_time of prior txns
        self.last_unix: int | None = None
        self.last_lat: float | None = None
        self.last_long: float | None = None

    def mean(self) -> float:
        return self.amount_sum / self.count if self.count else 0.0

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean()
        var = max(self.amount_sumsq / self.count - mean * mean, 0.0)
        return math.sqrt(var)

    def velocity(self, now_unix: int, window_seconds: int) -> int:
        """Count of prior txns within window_seconds before now_unix."""
        cutoff = now_unix - window_seconds
        return sum(1 for t in self.recent_times if t > cutoff)

    def update(self, unix_time: int, amount: float, lat: float, long: float) -> None:
        self.count += 1
        self.amount_sum += amount
        self.amount_sumsq += amount * amount
        self.recent_times.append(unix_time)
        # Bound memory: drop times older than the widest window we ever query.
        cutoff = unix_time - config.VELOCITY_WINDOW_24H_SECONDS
        while self.recent_times and self.recent_times[0] <= cutoff:
            self.recent_times.popleft()
        self.last_unix = unix_time
        self.last_lat = lat
        self.last_long = long


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def compute_signals(
    row: pd.Series, state: AccountState, watchlist: set[str]
) -> TransactionSignals:
    """Score one transaction against its prior account state.

    Returns TransactionSignals (rule_score, triggered_rules, and the raw signal
    values the inner loop will also see). Does not mutate state.
    """
    w = config.RULE_WEIGHTS
    triggered: list[str] = []
    contributions = 0.0

    now_unix = int(row["unix_time"])
    amount = float(row["amount"])

    # --- amount anomaly vs the account's own history ---
    mean = state.mean()
    std = state.std()
    if std > 0:
        z = (amount - mean) / std
    else:
        z = 0.0
    amount_vs_account_mean = amount / mean if mean > 0 else 1.0
    if z > 0:
        contrib = _clip01(z / config.AMOUNT_ANOMALY_FULL_Z) * w["amount_anomaly"]
        if contrib > 0:
            contributions += contrib
            triggered.append("amount_anomaly")

    # --- velocity ---
    velocity_1h = state.velocity(now_unix, config.VELOCITY_WINDOW_1H_SECONDS)
    velocity_24h = state.velocity(now_unix, config.VELOCITY_WINDOW_24H_SECONDS)
    v1_contrib = _clip01(velocity_1h / config.VELOCITY_1H_FULL) * w["velocity_1h"]
    if v1_contrib > 0:
        contributions += v1_contrib
        triggered.append("velocity_1h")
    v24_contrib = _clip01(velocity_24h / config.VELOCITY_24H_FULL) * w["velocity_24h"]
    if v24_contrib > 0:
        contributions += v24_contrib
        triggered.append("velocity_24h")

    # --- geo: distance for this txn, impossible travel vs prior txn ---
    distance_km = haversine(
        float(row["cardholder_lat"]),
        float(row["cardholder_long"]),
        float(row["merchant_lat"]),
        float(row["merchant_long"]),
    )
    impossible_travel = False
    if state.last_unix is not None and state.last_lat is not None:
        dt_seconds = now_unix - state.last_unix
        step_km = haversine(
            state.last_lat,
            state.last_long,
            float(row["cardholder_lat"]),
            float(row["cardholder_long"]),
        )
        if dt_seconds > 0:
            implied_kmh = step_km / (dt_seconds / 3600.0)
            if implied_kmh > config.IMPOSSIBLE_TRAVEL_KMH:
                impossible_travel = True
        elif step_km > 0:
            # Same timestamp, different place: physically impossible.
            impossible_travel = True
    if impossible_travel:
        contributions += w["impossible_travel"]
        triggered.append("impossible_travel")

    # --- high-risk merchant category ---
    if str(row["merchant_category"]) in config.HIGH_RISK_CATEGORIES:
        contributions += w["high_risk_category"]
        triggered.append("high_risk_category")

    # --- odd hour ---
    hour = pd.Timestamp(row["timestamp"]).hour
    if config.ODD_HOUR_START <= hour <= config.ODD_HOUR_END:
        contributions += w["odd_hour"]
        triggered.append("odd_hour")

    # --- watchlist: a known-fraud card saturates the score ---
    if str(row["account_id"]) in watchlist:
        contributions += w["watchlist"]
        triggered.append("watchlist")

    rule_score = _clip01(contributions)

    return TransactionSignals(
        txn_id=str(row["txn_id"]),
        rule_score=rule_score,
        triggered_rules=triggered,
        velocity_1h=velocity_1h,
        velocity_24h=velocity_24h,
        amount_vs_account_mean=amount_vs_account_mean,
        distance_km=distance_km,
        impossible_travel=impossible_travel,
    )
