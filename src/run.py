"""CLI entry point.

Examples:
    python -m src.run --rules-only --limit 1000           # outer loop only, the baseline
    python -m src.run --mode mock --gate rules --limit 5000   # rule-score gate, free
    python -m src.run --mode mock --gate agent --limit 5000   # triage-agent gate, free
    python -m src.run --mode replay                       # reuse a recorded cassette, free
    python -m src.run --record --gate agent --limit 50    # real triage + agent (costs a little)
    python -m src.run --mode live                         # real agent on everything (off by default)

--gate chooses who is investigated: rules (deterministic, default) or agent (LLM
triage on every txn). Only --record / --mode live spend money; mock and replay
make zero API calls. Every run prints an estimated cost line from call counts.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python src/run.py ...` (the SPEC gate form) as well as `python -m src.run`
# by making the repo root importable regardless of how this file is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load


def _build_investigator(mode: str):
    """Construct the inner-loop investigator for the given mode.

    Imported lazily so the rules-only path never depends on the agent module
    or the Anthropic client.
    """
    from src.agent import build_agent

    return build_agent(mode)


def _build_triage(mode: str):
    """Construct the first-loop triage agent for the given mode.

    Imported lazily so the rules-only path never depends on the triage module
    or the Anthropic client.
    """
    from src.triage_agent import build_triage_agent

    return build_triage_agent(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-loop fraud screening")
    parser.add_argument(
        "--mode",
        choices=["mock", "replay", "record", "live"],
        default="mock",
        help="inner-loop execution mode (default: mock, free)",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="disable the inner loop; outer-loop baseline only",
    )
    parser.add_argument(
        "--gate",
        choices=["rules", "agent"],
        default="rules",
        help="who decides which txns are investigated: the deterministic "
        "rule-score band (rules, default) or the triage LLM agent (agent)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="shortcut for --mode record (real agent, writes the cassette)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N transactions",
    )
    args = parser.parse_args()

    mode = "record" if args.record else args.mode

    df = load()
    if args.limit is not None:
        df = df.head(args.limit)

    # Import here to avoid a hard dependency in the rules-only path.
    from src.orchestrator import run_screening

    real_api = mode in ("record", "live")

    if args.rules_only:
        run_screening(df, investigator=None, gate=args.gate, rules_only=True, real_api=real_api)
        return

    investigator = _build_investigator(mode)
    # The triage agent is only built for the agent gate, so the rules gate never
    # needs the triage module or (in record/live) an API key.
    triage = _build_triage(mode) if args.gate == "agent" else None
    run_screening(
        df,
        investigator=investigator,
        triage=triage,
        gate=args.gate,
        real_api=real_api,
    )
    if mode == "record":
        msg = f"Recorded verdicts to {config.CASSETTE_FILE.name}"
        if args.gate == "agent":
            msg += f" and triage decisions to {config.TRIAGE_CASSETTE_FILE.name}"
        print(msg + " for free replay.")


if __name__ == "__main__":
    main()
