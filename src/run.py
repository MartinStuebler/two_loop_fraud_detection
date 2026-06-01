"""CLI entry point.

Examples:
    python -m src.run --rules-only --limit 1000     # outer loop only, the baseline
    python -m src.run --mode mock --limit 5000      # full two-loop, free
    python -m src.run --mode replay                 # reuse a recorded cassette, free
    python -m src.run --record --limit 50           # real agent on flagged subset (costs a little)
    python -m src.run --mode live                   # real agent on everything (off by default)

Only --record / --mode live spend money. mock and replay make zero API calls.
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

    if args.rules_only:
        run_screening(df, investigator=None, rules_only=True)
        return

    investigator = _build_investigator(mode)
    llm_called = mode in ("record", "live")
    run_screening(
        df,
        investigator=investigator,
        rules_only=False,
        llm_called_for_investigations=llm_called,
    )
    if mode == "record":
        print(f"Recorded verdicts to {config.CASSETTE_FILE.name} for free replay.")


if __name__ == "__main__":
    main()
