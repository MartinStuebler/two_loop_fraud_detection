"""Pydantic v2 schemas: the data contract between the two loops.

These three models are the only structured shapes that cross boundaries:

- TransactionSignals: outer loop -> inner loop. Cheap deterministic features.
- Verdict: inner loop -> orchestrator. The agent's structured judgement.
- TxnRecord: orchestrator -> outputs/verdicts.jsonl. One line per txn, the
  single artifact the eval harness reads.

See SPEC.md section 5. Field names and types here are the contract; do not
drift from them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TransactionSignals(BaseModel):
    """Cheap, deterministic features the outer loop hands to the inner loop."""

    txn_id: str
    rule_score: float = Field(ge=0.0, le=1.0)
    triggered_rules: list[str] = Field(default_factory=list)
    velocity_1h: int = Field(ge=0)
    velocity_24h: int = Field(ge=0)
    amount_vs_account_mean: float
    distance_km: float = Field(ge=0.0)
    impossible_travel: bool


class Verdict(BaseModel):
    """The inner-loop agent's structured output.

    risk_score is continuous and REQUIRED: the eval harness sweeps a decision
    threshold across it, so it must be a real number in 0..1, not a bucket.
    """

    txn_id: str
    label: Literal["legitimate", "suspicious", "fraud"]
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: Literal["clear", "review_queue", "block"]
    rationale: str
    tools_used: list[str] = Field(default_factory=list)
    iterations: int = Field(ge=0)


class TxnRecord(BaseModel):
    """One JSON line per transaction in outputs/verdicts.jsonl.

    final_score is populated for EVERY transaction: it is agent_risk_score when
    the txn was investigated, otherwise the outer-loop rule_score. The eval
    harness relies on final_score and rule_score being present on every row.
    """

    txn_id: str
    is_fraud: int = Field(ge=0, le=1)
    rule_score: float = Field(ge=0.0, le=1.0)
    investigated: bool
    agent_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    final_label: str
    recommended_action: str
    llm_called: bool
