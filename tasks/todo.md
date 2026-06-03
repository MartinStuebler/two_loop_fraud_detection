# Build Plan: Two-Loop Agentic Fraud Screening

One task per file, grouped by SPEC phase. Check each box when the file is done and its phase gate passes. Claude Code never touches `eval/` beyond the single stub.

## Phase 0: Environment and scaffolding
- [x] Install Python 3.12 (Homebrew), create `.venv`, install `requirements.txt`
- [x] Create directory tree (`data/`, `src/`, `outputs/`, `eval/`, `tests/`, `tasks/`)
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `tasks/todo.md` (this file)
- [x] Download raw Sparkov CSV from Hugging Face mirror to `data/raw/` (not committed)

## Phase 1: Data and schemas
- [x] `src/schemas.py` (TransactionSignals, Verdict, TxnRecord, Pydantic v2)
- [x] `src/config.py` (all thresholds, budgets, model name, paths, weights)
- [x] `src/data_loader.py` (`load()` + one-time sampler to `data/sample.csv`)
- [x] `data/sample.csv` committed (49,683 rows, 41 accounts, 0.55% fraud)
- [x] `data/watchlist.txt` seeded with 5 known-fraud cc_num
- [x] Gate: load assert prints row count (49683)

## Phase 2: Outer loop (rules-only)
- [x] `src/rules.py` (rule_score, triggered_rules, TransactionSignals build)
- [x] `src/orchestrator.py` (inner loop disabled path, writes verdicts.jsonl)
- [x] `src/run.py` (argparse, --rules-only path)
- [x] `tests/test_schemas.py`
- [x] `tests/test_rules.py`
- [x] Gate: `python src/run.py --rules-only --limit 1000` + `pytest tests/test_rules.py`

## Phase 3: Tools
- [x] `src/tools.py` (six tools, no LLM)
- [x] `tests/test_tools.py` (one case per tool)
- [x] Gate: `pytest tests/test_tools.py`

## Phase 4: Inner-loop agent
- [x] `src/agent.py` (bounded ReAct, mock/replay/record/live, JSON parse + retry, caps)
- [x] `tests/test_orchestrator.py` (agent mocked + fake-client loop tests, no live calls)
- [x] Gate: mock smoke test valid Verdict + MAX_ITERATIONS respected; `pytest tests/test_orchestrator.py`

## Phase 5: Full two-loop run
- [x] `src/orchestrator.py` mid-band + above-HIGH dispatch to agent (done in Phase 2 structure)
- [x] All four modes wired in `src/run.py`
- [x] `eval/evaluate.py` STUB ONLY (signatures + docstring pointing to SPEC section 11)
- [x] `README.md` (architecture, Sparkov choice, four modes, production paragraph)
- [x] Gate: `python src/run.py --mode mock --limit 5000`, all 5000 rows validate against TxnRecord with final_score populated

## Definition of done
- [x] All five phase gates pass
- [x] `pytest` green (29 passed), zero live API calls
- [x] `outputs/verdicts.jsonl` validates fully against TxnRecord
- [x] `eval/evaluate.py` is a stub only; `eval/` otherwise untouched
- [x] No m-dashes anywhere
- [x] README covers architecture, Sparkov, four modes, production paragraph

## Phase 6: Triage agent + selectable gate + cost estimate
- [x] `src/config.py`: add `TRIAGE_CASSETTE_FILE`, `TRIAGE_MAX_TOKENS`,
      `HAIKU_INPUT_PER_MTOK`, `HAIKU_OUTPUT_PER_MTOK`, `EST_TRIAGE_COST`, `EST_INVESTIGATION_COST`
- [x] `src/triage_agent.py`: `triage(row, signals) -> {"investigate": bool, "reason": str}`,
      mock/replay/record/live modes mirroring `src/agent.py`, `build_triage_agent(mode)`
- [x] `src/run.py`: `--gate rules|agent` (default rules); build triage only for the agent gate
- [x] `src/orchestrator.py`: gate selects rule-score band vs triage agent; rule_score
      always computed and written; triage and investigation calls tracked separately
- [x] `src/orchestrator.py`: print an `[estimated]` cost line every run from call counts (works in mock)
- [x] `tests/test_triage.py`: mock decision + fake-client real loop + replay; zero live calls
- [x] Gate: two free mock runs (rules and agent) write 49,683 records and print cost lines; `pytest` green

## Phase 7: Prompt-tuning harness for the triage agent
Scaffolding only; a real gold set arrives later. Lives under `tune/`, NOT `eval/`
(CLAUDE.md forbids writing under `eval/`; harness only IMPORTS from `eval/evaluate.py`).
- [x] `tune/gold_tune.jsonl` + `tune/gold_test.jsonl`: fixed labeled fixtures seeded from
      the existing cassette/verdicts; same fields as verdicts.jsonl incl `is_fraud`.
      Deterministic split by txn_id; nothing in the system writes these at runtime.
- [x] `tune/tune.py`: take a prompt-file path, run triage over the gold TUNE set in
      replay/mock (no live API by default), score by REUSING `confusion_matrix`,
      `precision`, `recall`, `operating_point` from `eval/evaluate.py`; print P/R/F1.
- [x] `tune/tune.py`: append one row to `tune/tuning_log.csv`
      (timestamp, prompt_file, git_commit, tune_precision, tune_recall, tune_f1).
- [x] `tune/tune.py`: `--test` flag scores the held-out TEST set, reported separately,
      NOT written to the tuning log.
- [x] `tune/README.md`: mark the seed as a placeholder, far too small to be a real judge
      (1 fraud / 100, test split has 0 fraud); document split, read-only intent, modes.
- [x] Gate: dry run on the seed tune set prints P/R/F1 and shows the logged row.
