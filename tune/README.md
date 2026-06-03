# Triage prompt-tuning harness

A small harness for iterating on the first-loop triage prompt against a fixed,
labeled gold set. It runs the triage agent over the gold set, scores its
investigate / clear decisions against known `is_fraud` labels, prints
precision / recall / F1, and records each run in `tuning_log.csv`.

It lives under `tune/`, not `eval/`: project rules forbid writing anything under
`eval/`. The harness only IMPORTS the scoring functions from `eval/evaluate.py`
(`confusion_matrix`, `precision`, `recall`, `operating_point`) and never modifies
them.

## The gold set is a placeholder. Do not trust these numbers.

`gold_tune.jsonl` and `gold_test.jsonl` are seeded from the existing recorded run
(`outputs/verdicts.jsonl`, whose every txn_id is also in the triage cassette) only
so the harness is runnable today. This seed is **far too small and too skewed to
be a real judge of a prompt**:

- 100 records total, with exactly **1 fraud**. Real fraud evaluation needs orders
  of magnitude more positives.
- The deterministic split puts that single fraud in the TUNE set, so the
  **held-out TEST set contains zero fraud**. Recall and F1 on the test set are
  therefore degenerate (0 fraud means recall is undefined and reported as 0).
- The seed records carry only the fields from `verdicts.jsonl` plus the label.
  They do not carry the raw transaction features or the full signal vector, so
  `mock` and `replay` work but `record` / `live` do not run against this seed
  (the harness fails loudly if you try).

Replace both files with a real labeled set before drawing any conclusion. The
fields each gold record must carry are the same fields as `outputs/verdicts.jsonl`
plus the known `is_fraud` label; for `record` / `live` tuning, also include the
raw transaction fields `account_id, amount, merchant_id, merchant_category,
unix_time`.

## Files

| File | What it is |
| --- | --- |
| `gold_tune.jsonl` | TUNE portion. Iterate against this. Read only. |
| `gold_test.jsonl` | Held-out TEST portion. Touch only after tuning. Read only. |
| `tune.py` | The harness. |
| `tuning_log.csv` | Append-only score log. The only tuning memory. |

The gold files are read-only in spirit and on disk (mode 444). Nothing in the
system writes to them at runtime; the harness only reads them. They were seeded
once, by hand, from the recorded run.

### Split

Deterministic and reproducible, independent of Python hash seeding:
`int(txn_id[:8], 16) % 10 < 7` goes to TUNE, the rest to TEST (roughly 70 / 30).
A given txn_id always lands in the same split.

## Scoring

A triage decision is binary: investigate or clear. The harness maps it to the
continuous `final_score` the eval functions expect (1.0 = investigate, 0.0 =
clear) and calls `operating_point` at threshold 0.5. The result is the confusion
matrix and F1 of "flag this transaction for investigation" against `is_fraud`.

## A caveat about mock and replay

`mock` and `replay` make no API calls, which is the default and free path. But in
those modes the **candidate prompt text is never sent to a model**, so the
decisions, and therefore the scores, do not depend on the prompt:

- `mock` decides from `rule_score` alone (investigate when `rule_score >=
  LOW_THRESHOLD`).
- `replay` returns the decision recorded in the triage cassette by txn_id.

The harness still validates and logs the prompt path so runs are reproducible,
and in `record` / `live` the candidate prompt IS exercised (the harness overrides
the in-memory triage system prompt with the file you pass, without editing
`prompts/triage.md`). Real prompt tuning therefore requires `record` or `live`
against a real gold set. Until then, treat mock / replay runs as a wiring check.

## Usage

```bash
# Score the default triage prompt on the TUNE set, replay mode (free). Logs a row.
python tune/tune.py

# Score a candidate prompt instead.
python tune/tune.py --prompt prompts/triage_candidate.md

# Free deterministic mock instead of replay.
python tune/tune.py --mode mock

# Held-out check. Run only after tuning. Reported separately, never logged.
python tune/tune.py --test
```

### `tuning_log.csv` columns

`timestamp, prompt_file, git_commit, tune_precision, tune_recall, tune_f1`

One row per TUNE run. The held-out `--test` run is reported to the console only
and is never written here, so the log stays a clean record of tuning-set scores.
Nothing is ever written back into the prompt.
