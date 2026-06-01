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
