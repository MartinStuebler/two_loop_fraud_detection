# PRD: Two-Loop Agentic Fraud Screening System

Portfolio build. Goal: screen individual payment transactions for fraud with a triage loop plus an investigation loop, built to be technically strong on its own merits. The eval harness is written by you; Claude Code builds everything else and hands you a clean data contract.

## 0. How to use this document with Claude Code

This file is the **specification**. Do not code straight from it. Sequence:

1. Place this file at the repo root as `SPEC.md`.
2. In Claude Code, enter **plan mode** (Shift+Tab twice).
3. Prompt: "Read SPEC.md. Build only the components marked CLAUDE CODE in section 10. Produce a plan in tasks/todo.md with checkable items, one task per file. Do not write code yet, and never touch eval/."
4. Review and approve the plan.
5. Implement phase by phase (section 12). After each phase, run that phase's gate command and confirm it passes before continuing.

Every acceptance gate is a command that passes or fails, not an interpretive judgment. That is deliberate: checkable criteria hold, interpretive ones drift.

## 1. Goal

Catch single fraudulent transactions using two loops:

- **Outer loop (triage, deterministic):** runs on 100 percent of transactions, cheap, no LLM.
- **Inner loop (investigation, agentic):** an LLM agent that runs only on the small flagged fraction.

The core constraint: you cannot run an LLM agent on every transaction. The outer loop exists to protect the inner loop's budget. State that rationale wherever relevant in comments.

## 2. Non-goals (out of scope)

- No streaming infra. Batch over a CSV.
- No model training. Outer loop is rules plus simple stats; inner loop is a pretrained LLM.
- No web UI. CLI plus the eval report you write.
- No account takeover, laundering, or synthetic-identity detection. Single-transaction fraud only.
- No live payment systems or real customer data.

## 3. Architecture

### 3.1 Outer loop: triage (deterministic)

For each transaction: compute a cheap `rule_score` in 0..1 from deterministic signals only, then apply thresholds (`LOW_THRESHOLD`, `HIGH_THRESHOLD` in config):

- below LOW: auto-clear, record, continue.
- above HIGH: auto-flag, still send to the inner loop for an explanation.
- in between: dispatch to the inner loop for investigation.

Maintains per-account rolling counters (velocity, amount sums) and a session summary. Must run with the inner loop disabled (`--rules-only`) for the baseline.

### 3.2 Inner loop: investigation agent (LLM, bounded ReAct)

Receives one transaction plus its precomputed signals. Loops: reason about what is missing, call tools to gather it, re-evaluate, repeat until confident or until `MAX_ITERATIONS` (default 5) or a per-investigation token budget is hit. On hitting the cap without confidence, default to `review_queue`. Never loop unbounded. Emits a structured `Verdict` (section 5) with a **continuous** `risk_score`.

### 3.3 The interface between loops

Outer loop hands the inner loop a transaction plus `TransactionSignals`. Inner loop hands back a `Verdict`. The orchestrator writes one `TxnRecord` per transaction to `outputs/verdicts.jsonl`. That file is the only thing your eval reads.

## 4. Data: Sparkov

Use the Sparkov synthetic credit-card dataset (has `is_fraud`, real merchant categories, and cardholder plus merchant lat/long, which enables geo and impossible-travel checks). Sample down to a manageable subset (for example 50k rows, fraud kept at its natural low rate) and commit only the sample.

`data_loader.py` normalizes Sparkov columns to canonical names:

```
txn_id            <- trans_num
timestamp         <- trans_date_trans_time
account_id        <- cc_num
amount            <- amt
merchant_id       <- merchant
merchant_category <- category
cardholder_geo    <- (lat, long)
merchant_geo      <- (merch_lat, merch_long)
city_pop          <- city_pop
is_fraud          <- is_fraud
```

## 5. Schemas (Pydantic v2)

```
TransactionSignals:
    txn_id: str
    rule_score: float            # 0..1 deterministic
    triggered_rules: list[str]
    velocity_1h: int
    velocity_24h: int
    amount_vs_account_mean: float
    distance_km: float           # cardholder to merchant
    impossible_travel: bool

Verdict:
    txn_id: str
    label: Literal["legitimate", "suspicious", "fraud"]
    risk_score: float            # 0..1 continuous, REQUIRED for the threshold sweep
    confidence: float            # 0..1
    recommended_action: Literal["clear", "review_queue", "block"]
    rationale: str               # plain language, <= 4 sentences
    tools_used: list[str]
    iterations: int

TxnRecord:                       # one JSON line per txn in outputs/verdicts.jsonl
    txn_id: str
    is_fraud: int                # ground truth 0/1
    rule_score: float            # outer-loop score, present for ALL txns
    investigated: bool
    agent_risk_score: float | None
    final_score: float           # agent_risk_score if investigated else rule_score
    final_label: str
    recommended_action: str
    llm_called: bool             # for cost accounting
```

The agent must return valid `Verdict` JSON. Prompt for JSON only, parse with Pydantic, retry once on failure, then default to `review_queue`.

## 6. Inner-loop tools (plain Python over the dataset, no LLM)

- `get_account_history(account_id, lookback_days)` : count, mean amount, top categories, first-seen.
- `get_velocity(account_id, window)` : txns in the window.
- `compare_to_baseline(account_id, txn)` : amount deviation, category novelty.
- `get_merchant_profile(merchant_id)` : category, dataset fraud rate, txn count.
- `get_geo_risk(txn)` : haversine distance cardholder to merchant, impossible-travel vs the same card's prior txn.
- `check_watchlist(entities)` : membership test against a local file seeded with a few known-fraud `cc_num`s.

## 7. File structure

```
fraud-screening/
  SPEC.md
  CLAUDE.md
  README.md
  requirements.txt
  data/
    sample.csv
    watchlist.txt
  src/
    schemas.py
    data_loader.py
    rules.py
    tools.py
    agent.py            # inner loop, with mock + live modes
    orchestrator.py     # outer loop, writes outputs/verdicts.jsonl
    config.py
    run.py
  outputs/
    verdicts.jsonl      # generated, the handoff to your eval
    cassette.jsonl      # generated by --record, enables free replay
  eval/
    evaluate.py         # YOU WRITE THIS. Claude Code leaves a stub only.
    report.md           # you generate
  tests/
    test_rules.py
    test_tools.py
    test_schemas.py
    test_orchestrator.py  # agent mocked, no live calls
```

## 8. Tech constraints

- Python 3.11+, Pydantic v2, pandas, pytest.
- LLM via Anthropic API, key from `ANTHROPIC_API_KEY`, never hardcoded.
- Default model `claude-haiku-4-5-20251001` for minimal spend; `claude-sonnet-4-6` selectable in config.
- All thresholds, budgets, and the model name live in `config.py`. No magic numbers elsewhere.
- Tests never make live API calls. Mock the agent.
- No m-dashes in any comment, string, or doc.

## 9. Execution modes (minimal spend)

The pipeline must run for free by default. Only `--record` spends money.

- **mock (default):** agent returns a deterministic stub verdict. Full pipeline runs, zero API calls. Used by all tests.
- **replay:** agent reads verdicts from `outputs/cassette.jsonl` keyed by `txn_id`. Zero API calls. Use after one recording run.
- **record:** `python src/run.py --record --limit 50` runs the real agent on the flagged subset of the first 50 transactions, saves their verdicts to the cassette. This is the only command that costs anything, a handful of cents on Haiku.
- **live:** real agent on everything. Off by default; document the likely cost in the README.

`run.py` exposes `--mode {mock,replay,record,live}`, `--rules-only`, and `--limit`.

## 10. Division of labor

**CLAUDE CODE builds:** `schemas.py`, `data_loader.py`, `rules.py`, `tools.py`, `agent.py`, `orchestrator.py`, `config.py`, `run.py`, and `tests/` (except eval tests). It must produce `outputs/verdicts.jsonl` in the exact `TxnRecord` shape.

**YOU build:** everything under `eval/`. Claude Code leaves `eval/evaluate.py` as a stub with the function signatures and a docstring pointing here, nothing more. It must not implement eval logic, write eval tests, or generate `report.md`.

## 11. The eval harness contract (your task)

Your `evaluate.py` reads `outputs/verdicts.jsonl` and produces `eval/report.md`. Inputs and required outputs:

- **Input:** one `TxnRecord` per line. You have `final_score` and `is_fraud` for every row, `rule_score` for every row, and `llm_called` per row.
- **Threshold sweep:** vary a decision threshold across `final_score`, compute precision and recall at each, and produce the precision/recall curve data. This is the explicit FP vs FN tradeoff you asked to surface.
- **Operating point:** pick one threshold, report its confusion matrix (TP, FP, TN, FN) and F1.
- **Baseline:** repeat precision and recall using `rule_score` alone (the rules-only system) so the two-loop lift is visible.
- **Cost story:** count `llm_called == true` rows. That is how many transactions cost money. Report it against total volume.
- **Report:** write all of the above to `eval/report.md`.

Claude Code's job is done when `verdicts.jsonl` is correct. Judging the system is yours, which is the part that shows you understand the tradeoff.

## 12. Phased build plan with gates (Claude Code)

**Phase 1: data and schemas.** Build `data_loader.py`, `schemas.py`, commit the Sparkov sample.
Gate: `python -c "from src.data_loader import load; d=load(); assert {'txn_id','amount','is_fraud'}.issubset(d.columns); print(len(d))"` prints a row count.

**Phase 2: outer loop (rules-only).** Build `rules.py` and a first `orchestrator.py` with the inner loop disabled.
Gate: `python src/run.py --rules-only --limit 1000` writes a `TxnRecord` for every row to `outputs/verdicts.jsonl`. `pytest tests/test_rules.py` passes.

**Phase 3: tools.** Build `tools.py`.
Gate: `pytest tests/test_tools.py` passes, one known input/output case per tool.

**Phase 4: inner-loop agent.** Build `agent.py` with the bounded ReAct loop, mock and replay modes, JSON parsing, and the caps.
Gate: mock-mode single-transaction smoke test returns a valid `Verdict` and respects `MAX_ITERATIONS`. `pytest tests/test_orchestrator.py` passes with the agent mocked.

**Phase 5: full two-loop run.** Wire mid-band dispatch to the agent. Confirm all four execution modes work.
Gate: `python src/run.py --mode mock --limit 5000` completes and `outputs/verdicts.jsonl` validates against `TxnRecord` for every row, with `final_score` populated everywhere. (Eval is yours, not a gate here.)

## 13. Definition of done

- All five phase gates pass.
- `pytest` green, zero live API calls in the suite.
- `outputs/verdicts.jsonl` validates fully against `TxnRecord`.
- `eval/evaluate.py` is a stub only; `eval/` is otherwise untouched by Claude Code.
- README explains the architecture, the Sparkov choice, the four execution modes, and one paragraph on what you would change for production.

## 14. Contents for CLAUDE.md (create separately, keep it lean)

```
# Fraud Screening Project Rules

## Hard rules
- Never write or modify anything under eval/. That code is the owner's.
- Never call the live Anthropic API in tests. Mock the agent.
- Default run mode is mock and must cost nothing. Only --record hits the API.
- Never commit secrets. Key comes from ANTHROPIC_API_KEY.
- All thresholds, budgets, and model name live in config.py. No magic numbers elsewhere.
- The agent must be bounded by MAX_ITERATIONS and a token budget. No unbounded loops.
- The agent must emit a continuous risk_score. orchestrator must write final_score for every txn.
- No m-dashes in code, comments, or docs.

## Workflow
- Plan before coding. Write the plan to tasks/todo.md with checkable items.
- One file per task. Check in after each phase gate before continuing.
- Run the phase gate command and confirm it passes before advancing.

## Architecture reminder
- Outer loop: cheap, deterministic, runs on all transactions.
- Inner loop: expensive LLM agent, runs only on flagged transactions.
- The outer loop exists to protect the inner loop's budget.
```
