# Build Plan: Two-Loop Agentic Fraud Screening

One task per file, grouped by SPEC phase. Check each box when the file is done and its phase gate passes. Claude Code never touches `eval/` beyond the single stub.

## Phase 0: Environment and scaffolding
- [x] Install Python 3.12 (Homebrew), create `.venv`, install `requirements.txt`
- [x] Create directory tree (`data/`, `src/`, `outputs/`, `eval/`, `tests/`, `tasks/`)
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `tasks/todo.md` (this file)
- [ ] Download raw Sparkov CSV from Hugging Face mirror to `data/raw/` (not committed)

## Phase 1: Data and schemas
- [ ] `src/schemas.py` (TransactionSignals, Verdict, TxnRecord, Pydantic v2)
- [ ] `src/config.py` (all thresholds, budgets, model name, paths, weights)
- [ ] `src/data_loader.py` (`load()` + one-time sampler to `data/sample.csv`)
- [ ] `data/sample.csv` committed (50k sample, natural fraud rate)
- [ ] `data/watchlist.txt` seeded with a few known-fraud cc_num
- [ ] Gate: load assert prints row count

## Phase 2: Outer loop (rules-only)
- [ ] `src/rules.py` (rule_score, triggered_rules, TransactionSignals build)
- [ ] `src/orchestrator.py` (first cut, inner loop disabled, writes verdicts.jsonl)
- [ ] `src/run.py` (first cut, argparse, --rules-only path)
- [ ] `tests/test_schemas.py`
- [ ] `tests/test_rules.py`
- [ ] Gate: `python src/run.py --rules-only --limit 1000` + `pytest tests/test_rules.py`

## Phase 3: Tools
- [ ] `src/tools.py` (six tools, no LLM)
- [ ] `tests/test_tools.py` (one case per tool)
- [ ] Gate: `pytest tests/test_tools.py`

## Phase 4: Inner-loop agent
- [ ] `src/agent.py` (bounded ReAct, mock/replay/record/live, JSON parse + retry, caps)
- [ ] `tests/test_orchestrator.py` (agent mocked, no live calls)
- [ ] Gate: mock smoke test valid Verdict + MAX_ITERATIONS respected; `pytest tests/test_orchestrator.py`

## Phase 5: Full two-loop run
- [ ] Extend `src/orchestrator.py` (mid-band + above-HIGH dispatch to agent)
- [ ] Wire all four modes in `src/run.py`
- [ ] `eval/evaluate.py` STUB ONLY (signatures + docstring pointing to SPEC section 11)
- [ ] `README.md` (architecture, Sparkov choice, four modes, production paragraph)
- [ ] Gate: `python src/run.py --mode mock --limit 5000`, all rows validate against TxnRecord with final_score populated
