"""Orchestrator and agent tests. The agent is always mocked or driven by a fake
client: this suite makes zero live API calls (CLAUDE.md hard rule)."""

from __future__ import annotations

import pandas as pd

from src import config
from src.agent import MockAgent, RealAgent, _extract_json
from src.orchestrator import run_screening
from src.schemas import TransactionSignals, TxnRecord
from src.tools import Toolbox


# --- a small synthetic dataset that produces at least one flagged txn ----
def _df() -> pd.DataFrame:
    rows = [
        # account Z: three small txns spaced > 24h apart, then a huge one.
        ("Z1", 0, "Z", 9.0),
        ("Z2", 3 * 86_400, "Z", 10.0),
        ("Z3", 6 * 86_400, "Z", 11.0),
        ("Z4", 9 * 86_400, "Z", 10_000.0),  # amount anomaly -> flagged
    ]
    df = pd.DataFrame(rows, columns=["txn_id", "unix_time", "account_id", "amount"])
    df["timestamp"] = pd.to_datetime(df["unix_time"], unit="s") + pd.Timedelta(hours=12)
    df["merchant_id"] = "Mz"
    df["merchant_category"] = "entertainment"  # not high-risk
    df["cardholder_lat"] = 40.0
    df["cardholder_long"] = -74.0
    df["merchant_lat"] = 40.0
    df["merchant_long"] = -74.0
    df["city_pop"] = 100000
    df["is_fraud"] = [0, 0, 0, 1]
    return df


def _signals() -> TransactionSignals:
    return TransactionSignals(
        txn_id="Z4",
        rule_score=0.5,
        triggered_rules=["amount_anomaly"],
        velocity_1h=0,
        velocity_24h=0,
        amount_vs_account_mean=1000.0,
        distance_km=0.0,
        impossible_travel=False,
    )


# --- orchestrator with the mock agent ------------------------------------
def test_orchestrator_writes_valid_records_for_every_row():
    records = run_screening(_df(), investigator=MockAgent(), rules_only=False)
    assert len(records) == 4
    for rec in records:
        # Re-validate against the schema and confirm final_score is populated.
        validated = TxnRecord.model_validate(rec.model_dump())
        assert validated.final_score is not None
        assert validated.llm_called is False  # mock never calls the API
    # At least one row should have been investigated by the inner loop.
    assert any(r.investigated for r in records)
    # Cleared rows are never investigated.
    for r in records:
        if r.rule_score < config.LOW_THRESHOLD:
            assert r.investigated is False


def test_rules_only_investigates_nothing():
    records = run_screening(_df(), investigator=MockAgent(), rules_only=True)
    assert all(r.investigated is False for r in records)
    assert all(r.final_score == r.rule_score for r in records)


# --- mock agent smoke test -----------------------------------------------
def test_mock_agent_returns_valid_verdict():
    verdict = MockAgent().investigate(pd.Series({"txn_id": "Z4"}), _signals())
    assert verdict.txn_id == "Z4"
    assert 0.0 <= verdict.risk_score <= 1.0
    assert verdict.recommended_action in ("clear", "review_queue", "block")
    assert verdict.iterations >= 1


# --- fake Anthropic client to exercise the real loop without a network ---
class _Usage:
    output_tokens = 10


class _ToolUseBlock:
    type = "tool_use"
    name = "get_geo_risk"
    input: dict = {}
    id = "tu_1"


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _AlwaysToolClient:
    """Never finalizes: every call asks for another tool use."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        return _Resp([_ToolUseBlock()], "tool_use")


class _FinalJsonClient:
    """Returns a valid final JSON verdict on the first call."""

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        payload = (
            '{"label": "fraud", "risk_score": 0.91, "confidence": 0.8, '
            '"recommended_action": "block", "rationale": "Large anomalous amount."}'
        )
        return _Resp([_TextBlock(payload)], "end_turn")


def _row() -> pd.Series:
    return pd.Series(
        {
            "txn_id": "Z4", "account_id": "Z", "amount": 10000.0, "merchant_id": "Mz",
            "merchant_category": "entertainment", "unix_time": 9 * 86_400,
            "cardholder_lat": 40.0, "cardholder_long": -74.0,
            "merchant_lat": 40.0, "merchant_long": -74.0,
        }
    )


def test_real_agent_respects_max_iterations():
    client = _AlwaysToolClient()
    agent = RealAgent(Toolbox(_df(), watchlist=set()), client=client, record=False)
    verdict = agent.investigate(_row(), _signals())
    # The loop must stop at MAX_ITERATIONS and never call the client beyond it.
    assert client.calls == config.MAX_ITERATIONS
    assert verdict.iterations == config.MAX_ITERATIONS
    # Hitting the cap without confidence defaults to the review queue.
    assert verdict.recommended_action == "review_queue"


def test_real_agent_parses_final_json():
    client = _FinalJsonClient()
    agent = RealAgent(Toolbox(_df(), watchlist=set()), client=client, record=False)
    verdict = agent.investigate(_row(), _signals())
    assert verdict.label == "fraud"
    assert verdict.risk_score == 0.91
    assert verdict.recommended_action == "block"


def test_extract_json_tolerates_code_fence():
    data = _extract_json('```json\n{"a": 1}\n```')
    assert data == {"a": 1}
