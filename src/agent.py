"""Inner loop: the investigation agent.

This is the expensive half. It runs only on the small fraction the outer loop
flags, because running an LLM on every transaction is exactly the cost the
two-loop design exists to avoid.

Four execution modes (SPEC section 9):
  mock   : deterministic stub verdict from the signals. Zero API calls. Default.
  replay : read a recorded verdict from the cassette by txn_id. Zero API calls.
  record : real agent on the flagged subset, appending verdicts to the cassette.
  live   : real agent on everything. Off by default.

The real agent is a bounded ReAct loop: reason, call tools, re-evaluate, repeat
until confident or until MAX_ITERATIONS or the per-investigation token budget is
hit. It never loops unbounded. On giving up it defaults to review_queue. It
always returns a valid Verdict with a continuous risk_score.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src import config
from src.schemas import TransactionSignals, Verdict
from src.tools import Toolbox


# --- shared helpers ------------------------------------------------------
def _band_label_action(score: float) -> tuple[str, str, str]:
    """Map a continuous score to (label, recommended_action, band) by config."""
    if score < config.LOW_THRESHOLD:
        return "legitimate", "clear", "clear"
    if score >= config.HIGH_THRESHOLD:
        return "fraud", "block", "auto_flag"
    return "suspicious", "review_queue", "investigate"


def default_verdict(txn_id: str, iterations: int, tools_used: list[str]) -> Verdict:
    """The safe fallback when the agent cannot reach a confident answer.

    Defaults to the review queue rather than clearing or blocking outright.
    """
    return Verdict(
        txn_id=txn_id,
        label=config.DEFAULT_FALLBACK_LABEL,
        risk_score=config.HIGH_THRESHOLD,  # mid-high: route to a human
        confidence=0.3,
        recommended_action=config.DEFAULT_FALLBACK_ACTION,
        rationale="Investigation hit a budget or parsing limit without a "
        "confident conclusion. Routed to the review queue by default.",
        tools_used=tools_used,
        iterations=iterations,
    )


# --- mock ----------------------------------------------------------------
class MockAgent:
    """Deterministic stub. Adds no intelligence: risk_score passes the rule
    score through so the full pipeline runs for free. Used by every test."""

    def investigate(self, row: pd.Series, signals: TransactionSignals) -> Verdict:
        score = signals.rule_score
        label, action, _ = _band_label_action(score)
        # Confidence grows with distance from the uncertain middle band.
        midpoint = (config.LOW_THRESHOLD + config.HIGH_THRESHOLD) / 2
        confidence = min(1.0, 0.5 + abs(score - midpoint))
        return Verdict(
            txn_id=signals.txn_id,
            label=label,
            risk_score=score,
            confidence=round(confidence, 2),
            recommended_action=action,
            rationale="Mock verdict: deterministic passthrough of the outer-loop "
            "rule score. No API call was made.",
            tools_used=[],
            iterations=1,
        )


# --- replay --------------------------------------------------------------
class ReplayAgent:
    """Replays verdicts recorded in the cassette, keyed by txn_id. Free.

    Falls back to the mock verdict for any txn not present in the cassette, so a
    partial recording still produces a complete run.
    """

    def __init__(self) -> None:
        self._cassette: dict[str, Verdict] = {}
        if config.CASSETTE_FILE.exists():
            for line in config.CASSETTE_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                v = Verdict.model_validate_json(line)
                self._cassette[v.txn_id] = v
        self._fallback = MockAgent()

    def investigate(self, row: pd.Series, signals: TransactionSignals) -> Verdict:
        cached = self._cassette.get(signals.txn_id)
        if cached is not None:
            return cached
        return self._fallback.investigate(row, signals)


# --- real agent (record / live) -----------------------------------------
def _load_prompt(path) -> str:
    """Read an externalized system prompt from disk. Loaded once at import.

    Raises a clear error if the prompt file is missing so a misconfigured
    checkout fails loudly rather than running with an empty prompt.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. The investigation system prompt is "
            f"externalized under prompts/; restore it to run the agent."
        )
    return path.read_text().strip()


SYSTEM_PROMPT = _load_prompt(config.INVESTIGATION_PROMPT_FILE)


def _tool_schemas() -> list[dict[str, Any]]:
    """Anthropic tool-use schemas for the six inner-loop tools."""
    return [
        {
            "name": "get_account_history",
            "description": "Count, mean amount, top categories, and first-seen for an account.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                },
                "required": ["account_id"],
            },
        },
        {
            "name": "get_velocity",
            "description": "Number of txns for an account within a window in seconds.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "window_seconds": {"type": "integer"},
                },
                "required": ["account_id", "window_seconds"],
            },
        },
        {
            "name": "compare_to_baseline",
            "description": "Amount deviation and category novelty for the current txn vs the account's prior txns.",
            "input_schema": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
        {
            "name": "get_merchant_profile",
            "description": "Category, dataset fraud rate, and txn count for a merchant.",
            "input_schema": {
                "type": "object",
                "properties": {"merchant_id": {"type": "string"}},
                "required": ["merchant_id"],
            },
        },
        {
            "name": "get_geo_risk",
            "description": "Cardholder-to-merchant distance and impossible-travel check for the current txn.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "check_watchlist",
            "description": "Test whether the given entities are on the known-fraud watchlist.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["entities"],
            },
        },
    ]


class RealAgent:
    """Bounded ReAct agent backed by the Anthropic API.

    The Anthropic client is injected so the loop can be exercised in tests with
    a fake client and never make a live call.
    """

    def __init__(self, toolbox: Toolbox, client: Any, record: bool = False) -> None:
        self.toolbox = toolbox
        self.client = client
        self.record = record

    def _execute_tool(self, name: str, tool_input: dict, txn: dict) -> dict:
        """Dispatch a tool call, injecting the current txn where a tool needs it."""
        tb = self.toolbox
        if name == "get_account_history":
            return tb.get_account_history(
                tool_input["account_id"],
                tool_input.get("lookback_days", config.ACCOUNT_HISTORY_LOOKBACK_DAYS),
            )
        if name == "get_velocity":
            return tb.get_velocity(tool_input["account_id"], tool_input["window_seconds"])
        if name == "compare_to_baseline":
            return tb.compare_to_baseline(tool_input["account_id"], txn)
        if name == "get_merchant_profile":
            return tb.get_merchant_profile(tool_input["merchant_id"])
        if name == "get_geo_risk":
            return tb.get_geo_risk(txn)
        if name == "check_watchlist":
            return tb.check_watchlist(tool_input.get("entities", []))
        return {"error": f"unknown tool {name}"}

    def investigate(self, row: pd.Series, signals: TransactionSignals) -> Verdict:
        txn = {
            "txn_id": str(row["txn_id"]),
            "account_id": str(row["account_id"]),
            "amount": float(row["amount"]),
            "merchant_id": str(row["merchant_id"]),
            "merchant_category": str(row["merchant_category"]),
            "unix_time": int(row["unix_time"]),
            "cardholder_lat": float(row["cardholder_lat"]),
            "cardholder_long": float(row["cardholder_long"]),
            "merchant_lat": float(row["merchant_lat"]),
            "merchant_long": float(row["merchant_long"]),
        }
        user_msg = (
            "Investigate this transaction.\n"
            f"transaction: {json.dumps(txn)}\n"
            f"signals: {signals.model_dump_json()}"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
        tools = _tool_schemas()
        tools_used: list[str] = []
        spent_tokens = 0
        iterations = 0

        verdict = self._run_loop(messages, tools, txn, tools_used, spent_tokens, iterations)
        if self.record:
            self._append_cassette(verdict)
        return verdict

    def _run_loop(self, messages, tools, txn, tools_used, spent_tokens, iterations) -> Verdict:
        txn_id = txn["txn_id"]
        for iterations in range(1, config.MAX_ITERATIONS + 1):
            if spent_tokens >= config.TOKEN_BUDGET_PER_INVESTIGATION:
                return default_verdict(txn_id, iterations, tools_used)

            resp = self.client.messages.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS_PER_CALL,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            spent_tokens += getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        tools_used.append(block.name)
                        result = self._execute_tool(block.name, dict(block.input), txn)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result),
                            }
                        )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # Final answer: parse JSON, retry once on failure, else default.
            text = _text_of(resp)
            verdict = self._parse_verdict(text, txn_id, tools_used, iterations)
            if verdict is not None:
                return verdict
            retried = self._retry_for_json(messages, tools, txn_id, tools_used, iterations)
            return retried

        # Loop exhausted without a final answer.
        return default_verdict(txn_id, config.MAX_ITERATIONS, tools_used)

    def _retry_for_json(self, messages, tools, txn_id, tools_used, iterations) -> Verdict:
        for _ in range(config.JSON_PARSE_RETRIES):
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY the JSON "
                    "object described, nothing else.",
                }
            )
            resp = self.client.messages.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS_PER_CALL,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            verdict = self._parse_verdict(_text_of(resp), txn_id, tools_used, iterations)
            if verdict is not None:
                return verdict
        return default_verdict(txn_id, iterations, tools_used)

    def _parse_verdict(
        self, text: str, txn_id: str, tools_used: list[str], iterations: int
    ) -> Verdict | None:
        """Parse the model's JSON judgement into a Verdict, or None on failure."""
        data = _extract_json(text)
        if data is None:
            return None
        try:
            return Verdict(
                txn_id=txn_id,
                label=data["label"],
                risk_score=float(data["risk_score"]),
                confidence=float(data["confidence"]),
                recommended_action=data["recommended_action"],
                rationale=str(data["rationale"]),
                tools_used=tools_used,
                iterations=iterations,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _append_cassette(self, verdict: Verdict) -> None:
        config.CASSETTE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with config.CASSETTE_FILE.open("a") as f:
            f.write(verdict.model_dump_json() + "\n")


def _text_of(resp: Any) -> str:
    """Concatenate the text blocks of a response."""
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of model text, tolerating a code fence."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


# --- factory -------------------------------------------------------------
def build_agent(mode: str):
    """Construct the investigator for the given mode.

    mock and replay never touch the network. record and live build the Anthropic
    client (key strictly from ANTHROPIC_API_KEY) and a Toolbox over the full
    dataset so tools see complete account history.
    """
    if mode == "mock":
        return MockAgent()
    if mode == "replay":
        return ReplayAgent()
    if mode in ("record", "live"):
        import os

        import anthropic

        from src.data_loader import load
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. record and live modes need it; "
                "use --mode mock or --mode replay to run for free."
            )
        client = anthropic.Anthropic(api_key=api_key)
        toolbox = Toolbox(load())
        return RealAgent(toolbox, client=client, record=(mode == "record"))
    raise ValueError(f"unknown mode {mode}")
