"""First loop: the triage agent.

A cheap LLM gate that runs on every transaction and decides, in a single short
call, whether the transaction is worth handing to the slower investigation
agent. It replaces the old rule-score band gate: the deterministic rule_score is
still computed for every transaction, but it is now only the baseline written to
each record, not the thing that decides who gets investigated.

Four execution modes mirror src/agent.py (SPEC section 9):
  mock   : deterministic stand-in decision from the baseline rule score. Zero API calls. Default.
  replay : read a recorded decision from the triage cassette by txn_id. Zero API calls.
  record : real LLM triage on each txn, appending decisions to the triage cassette.
  live   : real LLM triage on each txn. No recording.

Every mode returns the same shape: {"investigate": bool, "reason": str}. The
real call is a single message (no tools, no ReAct loop), so triage stays cheap
relative to a full investigation. On a parse failure it defaults to investigate
so a possibly-fraudulent txn is never silently cleared.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src import config
from src.agent import _extract_json, _text_of
from src.schemas import TransactionSignals


# --- mock ----------------------------------------------------------------
class MockTriageAgent:
    """Deterministic stand-in. Investigates when the baseline rule score reaches
    LOW_THRESHOLD, so the full pipeline runs for free and the mock decision
    matches the band gate this triage agent replaces."""

    def triage(self, row: pd.Series, signals: TransactionSignals) -> dict:
        investigate = signals.rule_score >= config.LOW_THRESHOLD
        if investigate:
            reason = (
                f"Mock triage: rule_score {signals.rule_score:.2f} at or above the "
                f"{config.LOW_THRESHOLD:.2f} baseline; sending to investigation."
            )
        else:
            reason = (
                f"Mock triage: rule_score {signals.rule_score:.2f} below the "
                f"{config.LOW_THRESHOLD:.2f} baseline; no investigation needed."
            )
        return {"investigate": investigate, "reason": reason}


# --- replay --------------------------------------------------------------
class ReplayTriageAgent:
    """Replays decisions recorded in the triage cassette, keyed by txn_id. Free.

    Falls back to the mock decision for any txn not present in the cassette, so a
    partial recording still produces a complete run.
    """

    def __init__(self) -> None:
        self._cassette: dict[str, dict] = {}
        if config.TRIAGE_CASSETTE_FILE.exists():
            for line in config.TRIAGE_CASSETTE_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._cassette[str(rec["txn_id"])] = {
                    "investigate": bool(rec["investigate"]),
                    "reason": str(rec["reason"]),
                }
        self._fallback = MockTriageAgent()

    def triage(self, row: pd.Series, signals: TransactionSignals) -> dict:
        cached = self._cassette.get(signals.txn_id)
        if cached is not None:
            return cached
        return self._fallback.triage(row, signals)


# --- real agent (record / live) -----------------------------------------
def _load_prompt(path) -> str:
    """Read an externalized system prompt from disk. Loaded once at import.

    Raises a clear error if the prompt file is missing so a misconfigured
    checkout fails loudly rather than running with an empty prompt.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. The triage system prompt is "
            f"externalized under prompts/; restore it to run triage."
        )
    return path.read_text().strip()


TRIAGE_SYSTEM_PROMPT = _load_prompt(config.TRIAGE_PROMPT_FILE)


class RealTriageAgent:
    """Single-call LLM triage backed by the Anthropic API.

    The client is injected so the call path can be exercised by a fake client in
    tests and never make a live call.
    """

    def __init__(self, client: Any, record: bool = False) -> None:
        self.client = client
        self.record = record

    def triage(self, row: pd.Series, signals: TransactionSignals) -> dict:
        txn = {
            "txn_id": str(row["txn_id"]),
            "account_id": str(row["account_id"]),
            "amount": float(row["amount"]),
            "merchant_id": str(row["merchant_id"]),
            "merchant_category": str(row["merchant_category"]),
            "unix_time": int(row["unix_time"]),
        }
        user_msg = (
            "Triage this transaction.\n"
            f"transaction: {json.dumps(txn)}\n"
            f"signals: {signals.model_dump_json()}"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
        decision = self._decide(messages)
        if self.record:
            self._append_cassette(signals.txn_id, decision)
        return decision

    def _decide(self, messages: list[dict[str, Any]]) -> dict:
        """One short call, with at most JSON_PARSE_RETRIES extra attempts on bad
        JSON. Bounded: never loops unbounded. Defaults to investigate on giving
        up, so a possibly-fraudulent txn is never silently cleared."""
        for _ in range(config.JSON_PARSE_RETRIES + 1):
            resp = self.client.messages.create(
                model=config.MODEL,
                max_tokens=config.TRIAGE_MAX_TOKENS,
                system=TRIAGE_SYSTEM_PROMPT,
                messages=messages,
            )
            decision = _parse_decision(_text_of(resp))
            if decision is not None:
                return decision
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY the JSON "
                    "object described, nothing else.",
                }
            )
        return {
            "investigate": True,
            "reason": "Triage could not parse a decision; defaulting to investigate "
            "so the transaction is not silently cleared.",
        }

    def _append_cassette(self, txn_id: str, decision: dict) -> None:
        config.TRIAGE_CASSETTE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with config.TRIAGE_CASSETTE_FILE.open("a") as f:
            f.write(json.dumps({"txn_id": txn_id, **decision}) + "\n")


def _parse_decision(text: str) -> dict | None:
    """Parse the model's JSON triage reply into the decision shape, or None."""
    data = _extract_json(text)
    if data is None or "investigate" not in data:
        return None
    investigate = data["investigate"]
    if not isinstance(investigate, bool):
        return None
    reason = str(data.get("reason", "")).strip() or "No reason given."
    return {"investigate": investigate, "reason": reason}


# --- factory -------------------------------------------------------------
def build_triage_agent(mode: str):
    """Construct the triage agent for the given mode.

    mock and replay never touch the network. record and live build the Anthropic
    client (key strictly from ANTHROPIC_API_KEY). The mode here matches the inner
    loop's mode so a single run is consistently free or consistently real.
    """
    if mode == "mock":
        return MockTriageAgent()
    if mode == "replay":
        return ReplayTriageAgent()
    if mode in ("record", "live"):
        import os

        import anthropic
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. record and live modes need it; "
                "use --mode mock or --mode replay to run for free."
            )
        client = anthropic.Anthropic(api_key=api_key)
        return RealTriageAgent(client=client, record=(mode == "record"))
    raise ValueError(f"unknown mode {mode}")
