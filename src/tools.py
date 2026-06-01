"""Inner-loop tools: plain Python over the dataset, no LLM.

These are the only way the investigation agent learns anything beyond the single
transaction it was handed. Each tool is a deterministic lookup or computation
over the canonical dataframe. They return plain JSON-serializable dicts so their
output can be fed straight back to the model.

A Toolbox is constructed once over the full dataset and shared across all
investigations; per-account and per-merchant views are precomputed so each tool
call is cheap.

Reference-time convention: tools that take a "window" or "lookback" without an
explicit timestamp measure relative to the account's most recent transaction in
the dataset, unless a txn is passed in (geo / baseline), in which case prior
transactions are those strictly before that txn's time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.rules import haversine, load_watchlist


class Toolbox:
    """Indexed views over the dataset that back the six investigation tools."""

    def __init__(self, df: pd.DataFrame, watchlist: set[str] | None = None) -> None:
        self.df = df
        self._by_account = {
            acct: sub.sort_values("unix_time")
            for acct, sub in df.groupby("account_id")
        }
        # Precompute merchant profiles once: count, dataset fraud rate, category.
        grouped = df.groupby("merchant_id")
        self._merchant_count = grouped.size().to_dict()
        self._merchant_fraud_rate = grouped["is_fraud"].mean().to_dict()
        self._merchant_category = grouped["merchant_category"].first().to_dict()
        self.watchlist = watchlist if watchlist is not None else load_watchlist()

    # --- tool 1 -----------------------------------------------------------
    def get_account_history(
        self, account_id: str, lookback_days: int = config.ACCOUNT_HISTORY_LOOKBACK_DAYS
    ) -> dict[str, Any]:
        """Count, mean amount, top categories, and first-seen for an account."""
        sub = self._by_account.get(str(account_id))
        if sub is None or sub.empty:
            return {"account_id": str(account_id), "count": 0, "found": False}
        latest = int(sub["unix_time"].max())
        cutoff = latest - lookback_days * 86_400
        window = sub[sub["unix_time"] >= cutoff]
        top = window["merchant_category"].value_counts().head(3)
        return {
            "account_id": str(account_id),
            "found": True,
            "lookback_days": lookback_days,
            "count": int(len(window)),
            "mean_amount": round(float(window["amount"].mean()), 2),
            "top_categories": list(top.index),
            "first_seen": str(window["timestamp"].min()),
        }

    # --- tool 2 -----------------------------------------------------------
    def get_velocity(self, account_id: str, window: int) -> dict[str, Any]:
        """Number of txns for the account within `window` seconds of its latest txn."""
        sub = self._by_account.get(str(account_id))
        if sub is None or sub.empty:
            return {"account_id": str(account_id), "window_seconds": window, "count": 0}
        latest = int(sub["unix_time"].max())
        cutoff = latest - window
        count = int((sub["unix_time"] > cutoff).sum())
        return {
            "account_id": str(account_id),
            "window_seconds": window,
            "count": count,
        }

    # --- tool 3 -----------------------------------------------------------
    def compare_to_baseline(self, account_id: str, txn: dict[str, Any]) -> dict[str, Any]:
        """Amount deviation and category novelty vs the account's prior txns."""
        sub = self._by_account.get(str(account_id))
        now_unix = int(txn["unix_time"])
        amount = float(txn["amount"])
        category = str(txn["merchant_category"])
        if sub is None or sub.empty:
            return {"account_id": str(account_id), "has_baseline": False}
        prior = sub[sub["unix_time"] < now_unix]
        if prior.empty:
            return {"account_id": str(account_id), "has_baseline": False}
        mean = float(prior["amount"].mean())
        std = float(prior["amount"].std(ddof=0))
        amount_z = (amount - mean) / std if std > 0 else 0.0
        return {
            "account_id": str(account_id),
            "has_baseline": True,
            "prior_count": int(len(prior)),
            "mean_amount": round(mean, 2),
            "amount_z": round(amount_z, 2),
            "amount_vs_mean": round(amount / mean, 2) if mean > 0 else None,
            "category_novel": bool(category not in set(prior["merchant_category"])),
        }

    # --- tool 4 -----------------------------------------------------------
    def get_merchant_profile(self, merchant_id: str) -> dict[str, Any]:
        """Category, dataset fraud rate, and txn count for a merchant."""
        merchant_id = str(merchant_id)
        if merchant_id not in self._merchant_count:
            return {"merchant_id": merchant_id, "found": False}
        return {
            "merchant_id": merchant_id,
            "found": True,
            "category": self._merchant_category[merchant_id],
            "fraud_rate": round(float(self._merchant_fraud_rate[merchant_id]), 4),
            "txn_count": int(self._merchant_count[merchant_id]),
        }

    # --- tool 5 -----------------------------------------------------------
    def get_geo_risk(self, txn: dict[str, Any]) -> dict[str, Any]:
        """Haversine distance and impossible-travel vs the card's prior txn."""
        distance_km = haversine(
            float(txn["cardholder_lat"]),
            float(txn["cardholder_long"]),
            float(txn["merchant_lat"]),
            float(txn["merchant_long"]),
        )
        result: dict[str, Any] = {
            "distance_km": round(distance_km, 2),
            "impossible_travel": False,
        }
        sub = self._by_account.get(str(txn["account_id"]))
        if sub is None or sub.empty:
            return result
        now_unix = int(txn["unix_time"])
        prior = sub[sub["unix_time"] < now_unix]
        if prior.empty:
            return result
        last = prior.iloc[-1]
        step_km = haversine(
            float(last["cardholder_lat"]),
            float(last["cardholder_long"]),
            float(txn["cardholder_lat"]),
            float(txn["cardholder_long"]),
        )
        dt_seconds = now_unix - int(last["unix_time"])
        if dt_seconds > 0:
            implied_kmh = step_km / (dt_seconds / 3600.0)
            result["implied_kmh"] = round(implied_kmh, 1)
            result["impossible_travel"] = bool(implied_kmh > config.IMPOSSIBLE_TRAVEL_KMH)
        elif step_km > 0:
            result["impossible_travel"] = True
        result["step_km_from_prior"] = round(step_km, 2)
        return result

    # --- tool 6 -----------------------------------------------------------
    def check_watchlist(self, entities: list[str]) -> dict[str, Any]:
        """Membership test of the given entities against the watchlist."""
        matches = [str(e) for e in entities if str(e) in self.watchlist]
        return {"matches": matches, "on_watchlist": len(matches) > 0}
