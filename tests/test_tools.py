"""One known input/output case per inner-loop tool, over a tiny synthetic
dataset so every assertion is deterministic. No real data, no API."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tools import Toolbox


def _df() -> pd.DataFrame:
    # Account A: three small stable grocery txns, then one large electronics
    # txn from a far-away location. Account B: one electronics txn.
    rows = [
        # txn_id, unix, acct, amount, merchant, category, c_lat, c_long, m_lat, m_long, fraud
        ("A1", 0, "A", 10.0, "M1", "shopping_net", 40.0, -74.0, 40.0, -74.0, 0),
        ("A2", 3600, "A", 12.0, "M1", "shopping_net", 40.0, -74.0, 40.0, -74.0, 0),
        ("A3", 7200, "A", 11.0, "M1", "shopping_net", 40.0, -74.0, 40.0, -74.0, 0),
        ("A4", 10800, "A", 500.0, "M2", "electronics", 48.0, 2.0, 48.0, 2.0, 1),
        ("B1", 0, "B", 20.0, "M2", "electronics", 34.0, -118.0, 34.0, -118.0, 0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "txn_id", "unix_time", "account_id", "amount", "merchant_id",
            "merchant_category", "cardholder_lat", "cardholder_long",
            "merchant_lat", "merchant_long", "is_fraud",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["unix_time"], unit="s")
    df["city_pop"] = 100000
    return df


def _txn_a4() -> dict:
    return {
        "txn_id": "A4",
        "unix_time": 10800,
        "account_id": "A",
        "amount": 500.0,
        "merchant_id": "M2",
        "merchant_category": "electronics",
        "cardholder_lat": 48.0,
        "cardholder_long": 2.0,
        "merchant_lat": 48.0,
        "merchant_long": 2.0,
    }


def test_get_account_history():
    tb = Toolbox(_df(), watchlist=set())
    h = tb.get_account_history("A", lookback_days=3650)
    assert h["found"] is True
    assert h["count"] == 4
    assert h["mean_amount"] == pytest.approx((10 + 12 + 11 + 500) / 4)
    assert h["top_categories"][0] == "shopping_net"  # 3 of 4 txns


def test_get_velocity():
    tb = Toolbox(_df(), watchlist=set())
    # latest unix is 10800; window 10800 -> cutoff 0 -> A2, A3, A4 (A1 at 0 excluded).
    v = tb.get_velocity("A", window=10800)
    assert v["count"] == 3


def test_compare_to_baseline():
    tb = Toolbox(_df(), watchlist=set())
    b = tb.compare_to_baseline("A", _txn_a4())
    assert b["has_baseline"] is True
    assert b["prior_count"] == 3
    assert b["mean_amount"] == pytest.approx(11.0)
    assert b["amount_z"] > 10          # 500 vs mean 11 with tiny std
    assert b["category_novel"] is True  # electronics not seen before for A


def test_get_merchant_profile():
    tb = Toolbox(_df(), watchlist=set())
    p = tb.get_merchant_profile("M2")
    assert p["found"] is True
    assert p["txn_count"] == 2
    assert p["fraud_rate"] == pytest.approx(0.5)  # 1 of 2 txns fraud
    assert p["category"] == "electronics"


def test_get_geo_risk():
    tb = Toolbox(_df(), watchlist=set())
    g = tb.get_geo_risk(_txn_a4())
    assert g["distance_km"] == pytest.approx(0.0, abs=1.0)  # cardholder == merchant
    # Prior txn A3 was in New York; A4 is in Europe one hour later.
    assert g["impossible_travel"] is True


def test_check_watchlist():
    tb = Toolbox(_df(), watchlist={"A"})
    w = tb.check_watchlist(["A", "Z"])
    assert w["on_watchlist"] is True
    assert w["matches"] == ["A"]
    assert tb.check_watchlist(["Z"])["on_watchlist"] is False
