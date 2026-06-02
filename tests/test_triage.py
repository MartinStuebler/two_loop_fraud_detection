"""Triage agent tests. The real path is always driven by a fake client: this
suite makes zero live API calls (CLAUDE.md hard rule)."""

from __future__ import annotations

import pandas as pd

from src import config
from src.schemas import TransactionSignals
from src.triage_agent import (
    MockTriageAgent,
    RealTriageAgent,
    ReplayTriageAgent,
    _parse_decision,
)


def _signals(rule_score: float) -> TransactionSignals:
    return TransactionSignals(
        txn_id="T1",
        rule_score=rule_score,
        triggered_rules=[],
        velocity_1h=0,
        velocity_24h=0,
        amount_vs_account_mean=1.0,
        distance_km=0.0,
        impossible_travel=False,
    )


def _row() -> pd.Series:
    return pd.Series(
        {
            "txn_id": "T1", "account_id": "A", "amount": 100.0, "merchant_id": "M",
            "merchant_category": "shopping_net", "unix_time": 1_000,
        }
    )


# --- mock ----------------------------------------------------------------
def test_mock_triage_investigates_at_or_above_baseline():
    out = MockTriageAgent().triage(_row(), _signals(config.LOW_THRESHOLD))
    assert out["investigate"] is True
    assert isinstance(out["reason"], str) and out["reason"]


def test_mock_triage_skips_below_baseline():
    out = MockTriageAgent().triage(_row(), _signals(config.LOW_THRESHOLD - 0.01))
    assert out["investigate"] is False


# --- fake Anthropic client for the real triage path ----------------------
class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.usage = type("U", (), {"output_tokens": 5})()


class _OneCallClient:
    """Returns a valid triage decision on the first call. Counts calls so the
    test can assert triage is a single call, not a ReAct loop."""

    def __init__(self) -> None:
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        return _Resp('{"investigate": true, "reason": "Net merchant, worth a look."}')


class _BadJsonClient:
    """Never returns valid JSON, to exercise the bounded retry and safe default."""

    def __init__(self) -> None:
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        return _Resp("sorry, I cannot comply")


def test_real_triage_single_call_parses_decision():
    client = _OneCallClient()
    out = RealTriageAgent(client=client, record=False).triage(_row(), _signals(0.5))
    assert client.calls == 1  # triage is one call, not a loop
    assert out == {"investigate": True, "reason": "Net merchant, worth a look."}


def test_real_triage_bad_json_is_bounded_and_defaults_to_investigate():
    client = _BadJsonClient()
    out = RealTriageAgent(client=client, record=False).triage(_row(), _signals(0.5))
    # Initial attempt plus JSON_PARSE_RETRIES, never unbounded.
    assert client.calls == config.JSON_PARSE_RETRIES + 1
    assert out["investigate"] is True  # safe default: never silently clear


# --- replay --------------------------------------------------------------
def test_replay_falls_back_to_mock_when_cassette_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRIAGE_CASSETTE_FILE", tmp_path / "nope.jsonl")
    out = ReplayTriageAgent().triage(_row(), _signals(config.LOW_THRESHOLD))
    assert out["investigate"] is True  # mock fallback decision


# --- parsing -------------------------------------------------------------
def test_parse_decision_rejects_non_boolean_investigate():
    assert _parse_decision('{"investigate": "yes", "reason": "x"}') is None
    assert _parse_decision('{"reason": "no key"}') is None
