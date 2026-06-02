"""Two-loop orchestrator.

Baseline (always): score every txn deterministically with the rule engine,
maintain rolling per-account state, and write rule_score onto every record. The
rule score is no longer the gate that decides who gets investigated; it is the
deterministic baseline carried alongside the final score.

Gate (who gets investigated): selectable.
  "rules" : the deterministic rule-score band gate (cheap, runs nowhere extra).
  "agent" : the triage LLM agent decides per transaction.
The gate only chooses who is investigated; rule_score is computed and written
for every transaction either way. With the agent gate, triage runs on every txn,
so triage calls scale with total volume while investigation calls scale only
with the flagged subset. The two are tracked separately for cost accounting.

Second loop (investigation, only the flagged subset): those transactions are
dispatched to the investigation LLM agent.

Writes exactly one TxnRecord per transaction to outputs/verdicts.jsonl. That
file is the only handoff to the eval harness. Every run ends with an estimated
cost line computed from call counts, so it is correct even in mock mode.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src import config
from src.rules import AccountState, compute_signals, load_watchlist
from src.schemas import TransactionSignals, TxnRecord, Verdict


class Investigator(Protocol):
    """Anything the inner loop can be: mock, replay, record, or live agent."""

    def investigate(self, row: pd.Series, signals: TransactionSignals) -> Verdict:
        ...


class Triage(Protocol):
    """The first loop: decides per txn whether to investigate. Returns a dict
    with keys investigate (bool) and reason (str)."""

    def triage(self, row: pd.Series, signals: TransactionSignals) -> dict:
        ...


def _triage_band(rule_score: float) -> str:
    """Map a rule score to one of the three triage bands."""
    if rule_score < config.LOW_THRESHOLD:
        return "clear"          # auto-clear, no LLM
    if rule_score >= config.HIGH_THRESHOLD:
        return "auto_flag"      # auto-flag, still investigated for an explanation
    return "investigate"        # the mid band: genuinely uncertain, send to agent


def _baseline_record(
    row: pd.Series, signals: TransactionSignals, llm_called: bool = False
) -> TxnRecord:
    """Build a TxnRecord for a txn the inner loop did not investigate.

    final_score is the baseline rule_score; label/action follow the rule-score
    band. llm_called is True when triage still spent an API call on this txn
    (record/live) even though it routed away from a full investigation.
    """
    band = _triage_band(signals.rule_score)
    if band == "clear":
        label, action = "legitimate", "clear"
    elif band == "auto_flag":
        label, action = "fraud", "block"
    else:
        label, action = "suspicious", "review_queue"
    return TxnRecord(
        txn_id=signals.txn_id,
        is_fraud=int(row["is_fraud"]),
        rule_score=signals.rule_score,
        investigated=False,
        agent_risk_score=None,
        final_score=signals.rule_score,
        final_label=label,
        recommended_action=action,
        llm_called=llm_called,
    )


def _investigated_record(
    row: pd.Series, signals: TransactionSignals, verdict: Verdict, llm_called: bool
) -> TxnRecord:
    """Build a TxnRecord when the inner loop investigated the txn.

    final_score becomes the agent's continuous risk_score.
    """
    return TxnRecord(
        txn_id=signals.txn_id,
        is_fraud=int(row["is_fraud"]),
        rule_score=signals.rule_score,
        investigated=True,
        agent_risk_score=verdict.risk_score,
        final_score=verdict.risk_score,
        final_label=verdict.label,
        recommended_action=verdict.recommended_action,
        llm_called=llm_called,
    )


def run_screening(
    df: pd.DataFrame,
    investigator: Investigator | None = None,
    triage: Triage | None = None,
    gate: str = "rules",
    rules_only: bool = False,
    real_api: bool = False,
) -> list[TxnRecord]:
    """Run the full pipeline over df and write outputs/verdicts.jsonl.

    investigator: the inner loop. If None or rules_only is True, the inner loop
        is disabled and every record comes from the baseline alone.
    triage: the first-loop triage agent. Required when gate is "agent".
    gate: "rules" (deterministic rule-score band) or "agent" (triage LLM).
        Ignored when rules_only is True. rule_score is written either way.
    real_api: True when the run actually hits the API (record/live). Drives the
        per-record llm_called flag. The estimated cost line is independent of
        this: it is computed from logical call counts, so it works in mock too.
    """
    watchlist = load_watchlist()
    states: dict[str, AccountState] = {}
    records: list[TxnRecord] = []

    # Separate cost-accounting counters: triage scales with volume (agent gate),
    # investigation scales only with the flagged subset.
    n_investigated = 0
    triage_calls = 0
    investigation_calls = 0

    for _, row in df.iterrows():
        account_id = str(row["account_id"])
        state = states.setdefault(account_id, AccountState())

        # Score against PRIOR state, then update state with this txn.
        signals = compute_signals(row, state, watchlist)

        # Decide who gets investigated. rule_score is the baseline regardless.
        triaged = False
        if rules_only or investigator is None:
            investigate = False
        elif gate == "agent":
            if triage is None:
                raise ValueError("gate='agent' requires a triage agent")
            triage_calls += 1
            triaged = True
            investigate = bool(triage.triage(row, signals)["investigate"])
        else:  # gate == "rules"
            investigate = _triage_band(signals.rule_score) != "clear"

        if investigate:
            investigation_calls += 1
            verdict = investigator.investigate(row, signals)
            record = _investigated_record(row, signals, verdict, llm_called=real_api)
            n_investigated += 1
        else:
            # A triaged-but-cleared txn still spent a triage call (record/live).
            record = _baseline_record(row, signals, llm_called=real_api and triaged)

        records.append(record)
        state.update(
            int(row["unix_time"]),
            float(row["amount"]),
            float(row["cardholder_lat"]),
            float(row["cardholder_long"]),
        )

    _write_verdicts(records)
    _print_summary(records, n_investigated, gate, rules_only)
    _print_cost_estimate(gate, triage_calls, investigation_calls, rules_only)
    return records


def _write_verdicts(records: list[TxnRecord]) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.VERDICTS_FILE.open("w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def _print_summary(
    records: list[TxnRecord],
    n_investigated: int,
    gate: str,
    rules_only: bool,
) -> None:
    total = len(records)
    n_llm = sum(1 for r in records if r.llm_called)
    baseline_flagged = sum(1 for r in records if r.rule_score >= config.LOW_THRESHOLD)
    label = "rules-only" if rules_only else f"gate={gate}"
    print(
        f"[{label}] wrote {total} records to {config.VERDICTS_FILE.name} | "
        f"investigated {n_investigated} ({n_investigated / total:.2%}) | "
        f"rule-baseline flagged {baseline_flagged} ({baseline_flagged / total:.2%}) | "
        f"llm_called {n_llm} ({n_llm / total:.2%} of volume)"
    )


def _print_cost_estimate(
    gate: str,
    triage_calls: int,
    investigation_calls: int,
    rules_only: bool,
) -> None:
    """Estimated dollar cost from logical call counts. Computed the same way in
    every mode, so a free mock run reports what the equivalent real run would
    cost. See the EST_* constants in config.py."""
    est = (
        triage_calls * config.EST_TRIAGE_COST
        + investigation_calls * config.EST_INVESTIGATION_COST
    )
    gate_label = "none" if rules_only else gate
    print(
        f"[estimated] gate={gate_label} | triage_calls {triage_calls} | "
        f"investigation_calls {investigation_calls} | est_cost ${est:.2f}"
    )
