"""Two-loop orchestrator.

Outer loop (always): score every txn deterministically, maintain rolling
per-account state, and triage into three bands using the config thresholds.

Inner loop (only the flagged fraction): when an agent is provided and we are not
in rules-only mode, dispatch the flagged transactions to the LLM agent. Running
the agent on every txn is the thing we cannot afford, so the outer loop's triage
exists precisely to keep the inner loop's call volume small. That is the budget
contract of the whole system.

Writes exactly one TxnRecord per transaction to outputs/verdicts.jsonl. That
file is the only handoff to the eval harness.
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


def _triage_band(rule_score: float) -> str:
    """Map a rule score to one of the three triage bands."""
    if rule_score < config.LOW_THRESHOLD:
        return "clear"          # auto-clear, no LLM
    if rule_score >= config.HIGH_THRESHOLD:
        return "auto_flag"      # auto-flag, still investigated for an explanation
    return "investigate"        # the mid band: genuinely uncertain, send to agent


def _rules_only_record(row: pd.Series, signals: TransactionSignals) -> TxnRecord:
    """Build a TxnRecord from the outer loop alone (inner loop disabled).

    final_score is the rule_score; label/action follow the triage band.
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
        llm_called=False,
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
    rules_only: bool = False,
    llm_called_for_investigations: bool = False,
) -> list[TxnRecord]:
    """Run the full pipeline over df and write outputs/verdicts.jsonl.

    investigator: the inner loop. If None or rules_only is True, the inner loop
        is disabled and every record comes from the outer loop alone.
    llm_called_for_investigations: whether dispatching to the investigator costs
        an API call (True for record/live, False for mock/replay). Drives the
        llm_called flag used for cost accounting.
    """
    watchlist = load_watchlist()
    states: dict[str, AccountState] = {}
    records: list[TxnRecord] = []

    # Session summary counters.
    n_cleared = n_investigated = n_auto_flagged = 0

    for _, row in df.iterrows():
        account_id = str(row["account_id"])
        state = states.setdefault(account_id, AccountState())

        # Score against PRIOR state, then update state with this txn.
        signals = compute_signals(row, state, watchlist)
        band = _triage_band(signals.rule_score)

        use_inner_loop = investigator is not None and not rules_only and band != "clear"
        if use_inner_loop:
            verdict = investigator.investigate(row, signals)
            record = _investigated_record(
                row, signals, verdict, llm_called_for_investigations
            )
            n_investigated += 1
            if band == "auto_flag":
                n_auto_flagged += 1
        else:
            record = _rules_only_record(row, signals)
            if band == "clear":
                n_cleared += 1
            elif band == "auto_flag":
                n_auto_flagged += 1

        records.append(record)
        state.update(
            int(row["unix_time"]),
            float(row["amount"]),
            float(row["cardholder_lat"]),
            float(row["cardholder_long"]),
        )

    _write_verdicts(records)
    _print_summary(records, n_cleared, n_investigated, n_auto_flagged, rules_only)
    return records


def _write_verdicts(records: list[TxnRecord]) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.VERDICTS_FILE.open("w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def _print_summary(
    records: list[TxnRecord],
    n_cleared: int,
    n_investigated: int,
    n_auto_flagged: int,
    rules_only: bool,
) -> None:
    total = len(records)
    n_llm = sum(1 for r in records if r.llm_called)
    flagged = sum(1 for r in records if r.rule_score >= config.LOW_THRESHOLD)
    mode = "rules-only" if rules_only else "two-loop"
    print(
        f"[{mode}] wrote {total} records to {config.VERDICTS_FILE.name} | "
        f"outer-loop flagged {flagged} ({flagged / total:.2%}) | "
        f"investigated {n_investigated} | auto-flagged {n_auto_flagged} | "
        f"llm_called {n_llm} ({n_llm / total:.2%} of volume)"
    )
