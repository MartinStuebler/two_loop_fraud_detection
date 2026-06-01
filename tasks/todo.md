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
