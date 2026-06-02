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

import json
from pathlib import Path
from typing import Any

VERDICTS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "verdicts.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def load_records(path: Path = VERDICTS_PATH) -> list[dict[str, Any]]:
    """Read outputs/verdicts.jsonl into a list of TxnRecord dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def threshold_sweep(records, score_key="final_score"):
    """Compute precision and recall at thresholds from 0.0 to 1.0 in 0.05 steps."""
    results = []
    steps = 21  # 0.00, 0.05, ... 1.00 is 21 points
    for i in range(steps):
        t = i * 0.05
        cm = confusion_matrix(records, t, score_key)
        p = precision(cm)
        r = recall(cm)
        results.append({"threshold": t, "precision": p, "recall": r})
    return results


def confusion_matrix(records,threshold, score_key="final_score"):
    """Count TP, FP, TN, FN at a decision threshold."""
    tp = fp = tn = fn = 0
    for r in records:
        flagged = r[score_key] >= threshold
        is_fraud = r["is_fraud"] == 1
        if flagged and is_fraud:
            tp += 1
        elif flagged and not is_fraud:
            fp += 1
        elif not flagged and is_fraud:
            fn += 1
        else:
            tn +=1
    return {"tp": tp, "fp": fp, "tn": tn, "fn":  fn}
    

def precision(cm):
    """Of everything flagged, what fraction was really fraud? tp / (tp + fp)."""
    tp = cm["tp"]
    fp = cm["fp"]
    denom = tp + fp
    return tp / denom if denom else 0.0
    
def recall(cm):
    """Of all real fraud, what fraction did we catch? tp / (tp + fn)."""
    tp = cm["tp"]
    fn = cm["fn"]
    denom = tp + fn
    return tp / denom if denom else 0.0
    

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
    records = load_records()
    print(f"loaded {len(records)} records")
    for row in threshold_sweep(records):
        t = row["threshold"]
        p = row["precision"]
        r = row["recall"]
        print(f"t={t:.2f} p={p:.2f} r={r:.2f}")
 
