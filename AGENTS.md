<!-- FOR AI AGENTS - Human readability is a side effect, not a goal -->
<!-- Managed by agent: keep sections and order; edit content, not structure -->

# MediBot Agent Guide

**Precedence:** the **closest `AGENTS.md`** wins. This root file provides project defaults.

## Current learning plan

1. Build and test the backend database-access layer before FastAPI.
2. Route questions with a simple, transparent keyword strategy first.
3. Keep the routing interface replaceable; after the assignment, evaluate semantic and LLM-based routing.
4. Implement SQL RAG only after the SQL-generation, validation, and result-explanation flow has been discussed.
5. Add FastAPI endpoints only after the backend concepts are approved.

## Architecture boundaries

| Area | Responsibility |
| --- | --- |
| `backend/app/core/` | Configuration and authorization policy |
| `backend/app/services/` | Routing and SQL-RAG orchestration; `llm/` holds the shared LLM interface/factory and provider adapters |
| `backend/app/repositories/` | Read-only SQLite access; never call an LLM |
| `backend/app/api/` | Future FastAPI endpoints; do not add until requested |
| `backend/tests/` | Standard-library `unittest` coverage |

## Routing decision

- Phase 1 uses `route_question()` with `KeywordRoutingStrategy`.
- SQL RAG requires both an allowed role and an analytical phrase plus a database-domain term.
- Preserve the `RoutingStrategy` interface when adding semantic or LLM routing.
- Routing does not replace authorization; SQL access remains limited by `can_use_sql_rag()`.

## Data and security

- Database: `backend/data/mediassist.db`.
- Use `execute_read_only_query()` for database access.
- Only `SELECT` queries against the policy-approved tables are allowed.
- Keep credentials in `backend/.env`; never commit or display secrets.
- Select the LLM provider with `REQUIRED_LLM_MODEL_GROUP`; create provider clients only through `create_llm_client()`.
- Keep provider model IDs in the matching provider-prefixed `.env` variables.
- Do not make a real LLM API call without explicit user approval after key rotation.

## Validation

Run the backend unit tests from `MediBot/backend`:

```bash
python3 -m unittest discover -s tests -v
```

## Learning-code comments

- Add a concise `#` comment immediately above every class and function/method.
- Explain the responsibility in plain language; include a Java comparison when it helps explain a Python-specific construct.
- Keep docstrings for Python tooling, but do not use a docstring as the only learner-facing explanation.

## Index of scoped AGENTS.md

<!-- AGENTS-GENERATED:START scope-index -->
<!-- AGENTS-GENERATED:END scope-index -->
