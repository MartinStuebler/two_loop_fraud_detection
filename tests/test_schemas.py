"""Schema validation and round-trip tests. No data, no API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import TransactionSignals, TxnRecord, Verdict


def test_transaction_signals_round_trip():
    sig = TransactionSignals(
        txn_id="t1",
        rule_score=0.42,
        triggered_rules=["amount_anomaly", "velocity_1h"],
        velocity_1h=3,
        velocity_24h=10,
        amount_vs_account_mean=2.5,
        distance_km=12.3,
        impossible_travel=False,
    )
    restored = TransactionSignals.model_validate_json(sig.model_dump_json())
    assert restored == sig


def test_verdict_round_trip():
    v = Verdict(
        txn_id="t1",
        label="fraud",
        risk_score=0.9,
        confidence=0.8,
        recommended_action="block",
        rationale="High amount and impossible travel.",
        tools_used=["get_geo_risk"],
        iterations=2,
    )
    restored = Verdict.model_validate_json(v.model_dump_json())
    assert restored == v


def test_txn_record_round_trip():
    rec = TxnRecord(
        txn_id="t1",
        is_fraud=0,
        rule_score=0.1,
        investigated=False,
        agent_risk_score=None,
        final_score=0.1,
        final_label="legitimate",
        recommended_action="clear",
        llm_called=False,
    )
    restored = TxnRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_risk_score_must_be_in_range():
    with pytest.raises(ValidationError):
        Verdict(
            txn_id="t1",
            label="fraud",
            risk_score=1.5,  # out of 0..1
            confidence=0.5,
            recommended_action="block",
            rationale="bad",
            tools_used=[],
            iterations=1,
        )


def test_verdict_label_enum_enforced():
    with pytest.raises(ValidationError):
        Verdict(
            txn_id="t1",
            label="definitely_fraud",  # not in the literal set
            risk_score=0.5,
            confidence=0.5,
            recommended_action="block",
            rationale="bad",
            tools_used=[],
            iterations=1,
        )


def test_txn_record_final_score_required():
    with pytest.raises(ValidationError):
        TxnRecord(
            txn_id="t1",
            is_fraud=0,
            rule_score=0.1,
            investigated=False,
            agent_risk_score=None,
            # final_score missing
            final_label="legitimate",
            recommended_action="clear",
            llm_called=False,
        )
