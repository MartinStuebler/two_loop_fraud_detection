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
    cm = confusion_matrix(records, threshold)
    p = precision(cm)
    r = recall(cm)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"threshold": threshold, "cm": cm, "precision": p, "recall": r, "f1": f1}


def baseline(records: list[dict[str, Any]]):
    """Precision/recall using rule_score alone (the rules-only baseline)."""
    return threshold_sweep(records, score_key="rule_score")


def cost_story(records: list[dict[str, Any]]):
    """Count llm_called == true rows against total volume."""
    total = len(records)
    llm_calls = sum(1 for r in records if r["llm_called"])
    pct = llm_calls / total if total else 0.0
    return {"total": total, "llm_calls": llm_calls, "pct_with_llm": pct}

def write_report(path: Path = REPORT_PATH) -> None:
    """Write the full evaluation to eval/report.md."""
    records = load_records()

    two_loop = threshold_sweep(records, score_key="final_score")
    rules = baseline(records)
    op = operating_point(records, threshold=0.5)
    cost = cost_story(records)

    lines = []
    lines.append("# Fraud Screening Evaluation\n")
    lines.append(f"Total transactions: {cost['total']}\n")

    lines.append("## Operating point (threshold 0.50)\n")
    cm = op["cm"]
    lines.append(f"- TP {cm['tp']}  FP {cm['fp']}  TN {cm['tn']}  FN {cm['fn']}")
    lines.append(f"- precision {op['precision']:.3f}  recall {op['recall']:.3f}  f1 {op['f1']:.3f}\n")

    lines.append("## Cost story\n")
    lines.append(f"- transactions that called the LLM: {cost['llm_calls']} ({cost['pct_with_llm']:.2%} of volume)\n")

    lines.append("## Precision / recall curve\n")
    lines.append("| threshold | two-loop P | two-loop R | rules-only P | rules-only R |")
    lines.append("| --- | --- | --- | --- | --- |")
    for tl, rl in zip(two_loop, rules):
        lines.append(f"| {tl['threshold']:.2f} | {tl['precision']:.3f} | {tl['recall']:.3f} | {rl['precision']:.3f} | {rl['recall']:.3f} |")

    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    write_report()
