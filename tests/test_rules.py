"""Deterministic outer-loop tests: one known case per rule. No data, no API."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.rules import AccountState, compute_signals, haversine

BASE_UNIX = 1_600_000_000  # fixed reference time for all cases


def make_row(
    *,
    txn_id="t1",
    account_id="acct",
    amount=10.0,
    unix_time=BASE_UNIX,
    category="entertainment",  # not high-risk, keeps cases isolated
    hour=12,                   # daytime, not odd-hour
    cardholder=(40.0, -70.0),
    merchant=(40.0, -70.0),    # same point: distance 0, no geo noise
) -> pd.Series:
    ts = pd.Timestamp("2020-09-13") + pd.Timedelta(hours=hour)
    return pd.Series(
        {
            "txn_id": txn_id,
            "account_id": account_id,
            "amount": amount,
            "unix_time": unix_time,
            "merchant_category": category,
            "timestamp": ts,
            "cardholder_lat": cardholder[0],
            "cardholder_long": cardholder[1],
            "merchant_lat": merchant[0],
            "merchant_long": merchant[1],
            "is_fraud": 0,
        }
    )


def test_haversine_known_distance():
    # New York to Los Angeles is roughly 3940 km.
    d = haversine(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 < d < 4000


def test_clean_transaction_scores_zero():
    state = AccountState()
    sig = compute_signals(make_row(), state, watchlist=set())
    assert sig.rule_score == 0.0
    assert sig.triggered_rules == []
    assert sig.impossible_travel is False


def test_amount_anomaly_triggers():
    state = AccountState()
    # Prior history with a small, stable mean, spaced far apart so velocity is 0.
    for i, amt in enumerate((9.0, 10.0, 11.0)):
        state.update(BASE_UNIX + i * 3 * 86_400, amt, 40.0, -70.0)
    now = BASE_UNIX + 10 * 86_400
    sig = compute_signals(make_row(amount=1000.0, unix_time=now), state, set())
    assert "amount_anomaly" in sig.triggered_rules
    assert sig.amount_vs_account_mean > 50
    assert sig.rule_score > 0


def test_velocity_1h_triggers():
    state = AccountState()
    # Five prior txns within the last hour.
    for i in range(5):
        state.update(BASE_UNIX + i * 60, 10.0, 40.0, -70.0)
    now = BASE_UNIX + 5 * 60
    sig = compute_signals(make_row(unix_time=now), state, set())
    assert sig.velocity_1h == 5
    assert "velocity_1h" in sig.triggered_rules


def test_velocity_24h_triggers():
    state = AccountState()
    # Ten prior txns within the last day but spread beyond one hour.
    for i in range(10):
        state.update(BASE_UNIX + i * 3600, 10.0, 40.0, -70.0)
    now = BASE_UNIX + 10 * 3600
    sig = compute_signals(make_row(unix_time=now), state, set())
    assert sig.velocity_24h == 10
    assert "velocity_24h" in sig.triggered_rules


def test_impossible_travel_triggers():
    state = AccountState()
    # Prior txn on the US east coast.
    state.update(BASE_UNIX, 10.0, 40.0, -74.0)
    # One minute later, a cardholder location in Europe: physically impossible.
    now = BASE_UNIX + 60
    sig = compute_signals(
        make_row(unix_time=now, cardholder=(48.0, 2.0), merchant=(48.0, 2.0)),
        state,
        set(),
    )
    assert sig.impossible_travel is True
    assert "impossible_travel" in sig.triggered_rules


def test_high_risk_category_triggers():
    state = AccountState()
    risky = next(iter(config.HIGH_RISK_CATEGORIES))
    sig = compute_signals(make_row(category=risky), state, set())
    assert "high_risk_category" in sig.triggered_rules


def test_odd_hour_triggers():
    state = AccountState()
    sig = compute_signals(make_row(hour=3), state, set())  # 3am
    assert "odd_hour" in sig.triggered_rules


def test_watchlist_saturates_score():
    state = AccountState()
    sig = compute_signals(make_row(account_id="bad_card"), state, {"bad_card"})
    assert "watchlist" in sig.triggered_rules
    assert sig.rule_score == 1.0  # watchlist weight saturates and clips to 1.0


def test_distance_km_computed():
    state = AccountState()
    sig = compute_signals(
        make_row(cardholder=(40.7128, -74.0060), merchant=(34.0522, -118.2437)),
        state,
        set(),
    )
    assert 3900 < sig.distance_km < 4000


def test_rule_score_never_exceeds_one():
    state = AccountState()
    # Stack several signals at once on a watchlisted card.
    for i in range(5):
        state.update(BASE_UNIX + i * 60, 10.0, 40.0, -74.0)
    now = BASE_UNIX + 5 * 60
    risky = next(iter(config.HIGH_RISK_CATEGORIES))
    sig = compute_signals(
        make_row(
            account_id="bad_card",
            amount=5000.0,
            unix_time=now,
            category=risky,
            hour=3,
            cardholder=(48.0, 2.0),
            merchant=(48.0, 2.0),
        ),
        state,
        {"bad_card"},
    )
    assert sig.rule_score == pytest.approx(1.0)
