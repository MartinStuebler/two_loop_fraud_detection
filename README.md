# Two-Loop Agentic Fraud Screening

Screen individual payment transactions for fraud with two loops: a cheap
deterministic triage loop that runs on every transaction, and an expensive LLM
investigation agent that runs only on the small flagged fraction.

The whole design exists to answer one constraint: you cannot afford to run an
LLM agent on every transaction. So a fast rules-based outer loop decides which
tiny slice is worth the spend, and the agent investigates only those. On the
committed sample the outer loop forwards about 10 percent of transactions to the
inner loop, so roughly 90 percent never cost an API call.

## Architecture

```
            every txn                      flagged fraction (~10%)
   CSV  ->  OUTER LOOP (rules)  --band-->  INNER LOOP (LLM agent)  ->  verdicts.jsonl
            deterministic, cheap           bounded ReAct, tools
```

- **Outer loop (`src/rules.py`, `src/orchestrator.py`).** For each transaction
  it computes a deterministic `rule_score` in 0..1 from cheap signals (amount
  anomaly vs the account's own history, 1h and 24h velocity, haversine distance,
  impossible travel, odd hour, high-risk category, watchlist hit). It maintains
  rolling per-account state in a single forward pass. Three bands:
  below `LOW_THRESHOLD` auto-clears, at or above `HIGH_THRESHOLD` auto-flags,
  and the uncertain middle is sent to the inner loop. Auto-flagged txns are also
  sent to the inner loop for an explanation.
- **Inner loop (`src/agent.py`, `src/tools.py`).** A bounded ReAct agent gets
  one transaction plus its precomputed signals, then calls plain-Python tools
  over the dataset to gather what it still needs. It loops reason then tool then
  re-evaluate until confident or until `MAX_ITERATIONS` or a per-investigation
  token budget is hit. It never loops unbounded. On giving up it defaults to the
  review queue. It always returns a continuous `risk_score`.
- **The contract.** The orchestrator writes exactly one `TxnRecord` per
  transaction to `outputs/verdicts.jsonl`. That file is the only handoff to the
  evaluation harness. `final_score` is populated for every row: the agent's
  `risk_score` when investigated, otherwise the outer-loop `rule_score`.

All thresholds, budgets, weights, and the model name live in `src/config.py`.
There are no magic numbers elsewhere.

## Why Sparkov

The Sparkov synthetic credit-card dataset fits the problem exactly: it has a
ground-truth `is_fraud` label, real merchant categories, and both cardholder and
merchant latitude/longitude. The geo columns are what make the distance and
impossible-travel signals possible, and the labels are what make the evaluation
meaningful. It is synthetic, so there is no real customer data involved. We pull
the open Apache-2.0 mirror `pointe77/credit-card-transaction` from Hugging Face,
then sample whole accounts (not random rows) down to about 50k transactions,
keeping fraud at its natural low rate (~0.55 percent). Sampling whole accounts
matters: the velocity, account-baseline, and impossible-travel logic is
meaningless without each card's intact history, so row-level sampling would
quietly break the detector. Only the sample is committed (`data/sample.csv`); the
338 MB raw file is not.

To rebuild the sample from the raw download:

```bash
python -m src.data_loader --build-sample
```

## Execution modes (minimal spend)

The pipeline runs for free by default. Only `--record` and `--mode live` spend
money. Select the mode with `--mode {mock,replay,record,live}`.

- **mock (default).** The agent returns a deterministic stub verdict derived
  from the signals. The full pipeline runs with zero API calls. Every test uses
  this mode.
- **replay.** The agent reads verdicts from `outputs/cassette.jsonl` keyed by
  `txn_id`. Zero API calls. Use this after one recording run to reproduce the
  agent's output for free.
- **record.** `python src/run.py --record --limit 50` runs the real agent on the
  flagged subset of the first 50 transactions and saves their verdicts to the
  cassette. This is the only routine command that costs anything, a few cents on
  Haiku, because only the flagged subset is investigated.
- **live.** `python src/run.py --mode live` runs the real agent on every flagged
  transaction in the whole sample. Off by default. On the committed sample about
  4,900 transactions would be investigated; at roughly a few thousand tokens
  each on Haiku that is on the order of a few US dollars. Prefer record then
  replay unless you specifically want a full live run.

The real modes read the API key strictly from the `ANTHROPIC_API_KEY`
environment variable. The key is never hardcoded.

### Common commands

```bash
python src/run.py --rules-only --limit 1000   # outer-loop baseline, free
python src/run.py --mode mock                  # full two-loop, free, whole sample
python src/run.py --record --limit 50          # real agent on flagged subset (~cents)
python src/run.py --mode replay                # reuse the cassette, free
pytest                                          # full suite, zero live API calls
```

## Project layout

```
src/        outer loop, inner loop, tools, schemas, config, CLI
data/       committed Sparkov sample and the watchlist
outputs/    generated verdicts.jsonl (the eval handoff) and cassette.jsonl
eval/       the owner's evaluation harness (Claude Code leaves only a stub)
tests/      pytest suite, agent always mocked
```

The evaluation lives under `eval/` and is intentionally a stub here. It reads
`outputs/verdicts.jsonl` and is the part that judges the system: the threshold
sweep over `final_score`, the precision/recall tradeoff, the operating-point
confusion matrix, the rules-only baseline for measuring two-loop lift, and the
cost story. See `SPEC.md` section 11.

## What I would change for production

This is a batch portfolio build, so the production gap is mostly about state,
streaming, and feedback. The rolling per-account counters would move out of an
in-memory pass into a real low-latency store (for example Redis or a feature
store) so the outer loop can score a single live transaction in milliseconds
without replaying history; the merchant and account profiles the tools compute
on the fly would be precomputed and cached rather than recomputed per call. The
deterministic rule weights, which are hand-set here, would be replaced or
augmented by a trained gradient-boosted model calibrated to a target alert
budget, with the LLM agent reserved for the genuinely ambiguous band and for
producing human-readable rationales for analysts. Every verdict and its eventual
chargeback outcome would feed a labeled feedback loop so thresholds, weights, and
the model are retuned on drift rather than frozen, and the agent's tool calls,
token spend, and latency would be logged per investigation for cost and audit
control.
