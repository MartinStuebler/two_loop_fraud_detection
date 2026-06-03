"""Prompt-tuning harness for the triage agent (scaffolding).

Runs the first-loop triage agent over a fixed, labeled gold set and scores its
investigate / clear decisions against the known is_fraud labels, then prints
precision, recall, and F1 and appends one row to tune/tuning_log.csv.

Design rules this file obeys:
  - It never writes the gold files. They are read-only fixtures.
  - It REUSES the scoring functions in eval/evaluate.py (confusion_matrix,
    precision, recall, operating_point); it does not reimplement them and it
    does not modify eval/.
  - The default path makes no live API calls: mode is replay (recorded triage
    decisions) or mock (deterministic from rule_score). record and live exist
    for when a real gold set with full transaction fields arrives later.
  - The tuning memory lives in tune/tuning_log.csv, never back in the prompt.

Scoring trick: a triage decision is binary (investigate or clear). We turn it
into the continuous final_score the eval functions expect: 1.0 when the agent
chose to investigate, 0.0 when it cleared. operating_point at threshold 0.5 then
gives the confusion matrix and F1 of "flag for investigation" against is_fraud.

IMPORTANT about mock and replay: in those modes the candidate prompt text is NOT
actually sent to a model, so the decisions (and therefore the scores) do not
depend on the prompt. The prompt path is still validated and logged so the run
is reproducible, and in record / live the candidate prompt IS exercised. See
tune/README.md.

Usage:
  python tune/tune.py                         # score default prompt on TUNE set, replay mode
  python tune/tune.py --prompt prompts/triage_candidate.md
  python tune/tune.py --mode mock
  python tune/tune.py --test                  # score the held-out TEST set, reported separately, not logged
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable whether run as `python tune/tune.py` or `-m tune.tune`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config
from src.schemas import TransactionSignals
from src.triage_agent import build_triage_agent

# Reuse the owner's scoring functions; do not reimplement or modify them.
from eval.evaluate import confusion_matrix, operating_point, precision, recall

GOLD_TUNE = ROOT / "tune" / "gold_tune.jsonl"
GOLD_TEST = ROOT / "tune" / "gold_test.jsonl"
TUNING_LOG = ROOT / "tune" / "tuning_log.csv"
LOG_COLUMNS = [
    "timestamp",
    "prompt_file",
    "git_commit",
    "tune_precision",
    "tune_recall",
    "tune_f1",
]

# Triage runs only on the band the outer loop hands up, so a clear decision and
# an investigate decision map onto these continuous scores for the eval funcs.
SCORE_INVESTIGATE = 1.0
SCORE_CLEAR = 0.0
DECISION_THRESHOLD = 0.5

# Fields RealTriageAgent reads off the row in record / live mode. The seed gold
# does not carry them; a real gold set must, or record / live will fail loudly.
ROW_FIELDS = ("account_id", "amount", "merchant_id", "merchant_category", "unix_time")


def load_gold(path: Path) -> list[dict]:
    """Read a gold fixture (one labeled record per line). Read only."""
    if not path.exists():
        raise FileNotFoundError(
            f"Gold file not found: {path}. The tuning harness needs the fixed "
            f"gold fixtures under tune/."
        )
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _signals_from_gold(rec: dict) -> TransactionSignals:
    """Build the signals the triage agent reads. rule_score and txn_id are the
    only fields mock and replay actually use; the rest are honest placeholders
    because the seed gold does not carry the full signal vector."""
    return TransactionSignals(
        txn_id=str(rec["txn_id"]),
        rule_score=float(rec["rule_score"]),
        triggered_rules=[],
        velocity_1h=0,
        velocity_24h=0,
        amount_vs_account_mean=0.0,
        distance_km=0.0,
        impossible_travel=False,
    )


def _row_from_gold(rec: dict, mode: str) -> pd.Series:
    """Build the transaction row the triage agent receives. mock and replay
    ignore everything but txn_id; record and live need the raw fields, so we
    fail loudly if a future real run is pointed at a seed-shaped gold set."""
    data = {"txn_id": str(rec["txn_id"])}
    for field in ROW_FIELDS:
        if field in rec:
            data[field] = rec[field]
    if mode in ("record", "live"):
        missing = [f for f in ROW_FIELDS if f not in data]
        if missing:
            raise ValueError(
                f"Gold record {rec['txn_id']} is missing fields needed by "
                f"{mode} mode: {missing}. The seed gold supports only mock and "
                f"replay; supply a real gold set with full transaction fields "
                f"to tune against the live model."
            )
    return pd.Series(data)


def _apply_prompt(prompt_path: Path) -> None:
    """Validate the candidate prompt and make record / live actually use it.

    The triage module loads its system prompt once at import from the configured
    file. To evaluate a candidate prompt without editing prompts/triage.md, we
    read the candidate and override the in-memory prompt. This has no effect in
    mock / replay (they never call the model) but keeps the harness honest for
    record / live."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    import src.triage_agent as triage_module

    triage_module.TRIAGE_SYSTEM_PROMPT = prompt_path.read_text().strip()


def score_prompt(gold: list[dict], prompt_path: Path, mode: str) -> dict:
    """Run triage over the gold set and score decisions against is_fraud by
    reusing eval/evaluate.py. Returns the operating_point result plus counts."""
    _apply_prompt(prompt_path)
    agent = build_triage_agent(mode)

    scored = []
    investigated = 0
    for rec in gold:
        signals = _signals_from_gold(rec)
        row = _row_from_gold(rec, mode)
        decision = agent.triage(row, signals)
        investigate = bool(decision["investigate"])
        investigated += int(investigate)
        # Reshape the binary decision into the final_score the eval funcs read.
        scored.append(
            {
                "txn_id": rec["txn_id"],
                "is_fraud": int(rec["is_fraud"]),
                "final_score": SCORE_INVESTIGATE if investigate else SCORE_CLEAR,
            }
        )

    op = operating_point(scored, DECISION_THRESHOLD)
    op["n"] = len(scored)
    op["n_fraud"] = sum(r["is_fraud"] for r in scored)
    op["n_investigated"] = investigated
    return op


def git_commit() -> str:
    """Short HEAD hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_log(prompt_path: Path, op: dict) -> None:
    """Append one tuning row. The CSV is the only memory; nothing goes back into
    the prompt. Writes a header the first time."""
    new_file = not TUNING_LOG.exists()
    with TUNING_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(LOG_COLUMNS)
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                str(prompt_path),
                git_commit(),
                f"{op['precision']:.4f}",
                f"{op['recall']:.4f}",
                f"{op['f1']:.4f}",
            ]
        )


def _print_result(title: str, op: dict, mode: str) -> None:
    cm = op["cm"]
    print(f"== {title} ==")
    print(f"mode: {mode}   records: {op['n']}   fraud in set: {op['n_fraud']}")
    print(f"investigated: {op['n_investigated']} of {op['n']}")
    print(f"confusion: TP {cm['tp']}  FP {cm['fp']}  TN {cm['tn']}  FN {cm['fn']}")
    print(
        f"precision {op['precision']:.3f}   recall {op['recall']:.3f}   "
        f"f1 {op['f1']:.3f}"
    )
    if op["n_fraud"] == 0:
        print(
            "note: this set contains no fraud, so recall and f1 are degenerate. "
            "The seed gold is a placeholder; see tune/README.md."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Triage prompt-tuning harness.")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=config.TRIAGE_PROMPT_FILE,
        help="path to the candidate triage system prompt (default: configured triage.md)",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "replay", "record", "live"],
        default="replay",
        help="triage execution mode (default: replay, no API calls)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="score the held-out TEST set instead of TUNE. Reported separately, "
        "never written to the tuning log. Run only after tuning.",
    )
    args = parser.parse_args(argv)

    prompt_path = args.prompt.resolve()

    if args.test:
        gold = load_gold(GOLD_TEST)
        op = score_prompt(gold, prompt_path, args.mode)
        _print_result("HELD-OUT TEST (not logged)", op, args.mode)
        print(f"prompt: {prompt_path}")
        return

    gold = load_gold(GOLD_TUNE)
    op = score_prompt(gold, prompt_path, args.mode)
    _print_result("TUNE", op, args.mode)
    append_log(prompt_path, op)
    print(f"prompt: {prompt_path}")
    print(f"logged one row to {TUNING_LOG}")


if __name__ == "__main__":
    main()
