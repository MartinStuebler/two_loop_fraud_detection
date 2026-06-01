"""Evaluation harness: STUB ONLY.

This file is intentionally left as a stub by Claude Code. The evaluation logic is
the project owner's to write (SPEC.md sections 10 and 11). Claude Code provides
only the function signatures and this pointer; it must not implement the metrics,
write eval tests, or generate report.md.

What the owner implements here (see SPEC.md section 11):
  - Read one TxnRecord per line from outputs/verdicts.jsonl.
  - Threshold sweep over final_score: precision and recall at each threshold
    (the precision/recall curve, the explicit FP vs FN tradeoff).
  - Operating point: pick one threshold, report its confusion matrix
    (TP, FP, TN, FN) and F1.
  - Baseline: repeat precision and recall using rule_score alone so the
    two-loop lift over the rules-only system is visible.
  - Cost story: count llm_called == true rows against total volume.
  - Write all of the above to eval/report.md.

The data contract is fixed: every TxnRecord row has final_score, rule_score, and
is_fraud, plus llm_called for cost accounting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

VERDICTS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "verdicts.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def load_records(path: Path = VERDICTS_PATH) -> list[dict[str, Any]]:
    """Read outputs/verdicts.jsonl into a list of TxnRecord dicts."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


def threshold_sweep(records: list[dict[str, Any]], score_key: str = "final_score"):
    """Return precision/recall (and the curve data) across decision thresholds."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


def operating_point(records: list[dict[str, Any]], threshold: float):
    """Return the confusion matrix (TP, FP, TN, FN) and F1 at one threshold."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


def baseline(records: list[dict[str, Any]]):
    """Precision/recall using rule_score alone (the rules-only baseline)."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


def cost_story(records: list[dict[str, Any]]):
    """Count llm_called == true rows against total volume."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


def write_report(path: Path = REPORT_PATH) -> None:
    """Write the full evaluation to eval/report.md."""
    raise NotImplementedError("Owner implements: see SPEC.md section 11.")


if __name__ == "__main__":
    raise SystemExit(
        "eval/evaluate.py is a stub. The owner implements the metrics per "
        "SPEC.md section 11."
    )
