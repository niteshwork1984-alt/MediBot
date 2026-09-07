<!-- FOR AI AGENTS - Human readability is a side effect, not a goal -->
<!-- Managed by agent: keep sections and order; edit content, not structure -->

# MediBot Agent Guide

**Precedence:** the **closest `AGENTS.md`** wins. This root file provides project defaults.

## Current learning plan

1. Build and test the backend database-access layer before FastAPI.
2. Route questions with a simple, transparent keyword strategy first.
3. Keep the routing interface replaceable; after the assignment, evaluate semantic and LLM-based routing.
4. Implement SQL RAG only after the SQL-generation, validation, and result-explanation flow has been discussed.
5. Implement document ingestion as a separate explicit command before retrieval; use Docling, LangChain, and Qdrant hybrid storage.
6. Implement user identity, role verification, and signed session tokens before FastAPI endpoints.
7. Add FastAPI endpoints only after the backend concepts are approved.

## Architecture boundaries

| Area | Responsibility |
| --- | --- |
| `backend/app/core/` | Configuration, authorization policy, password hashing, and signed-session helpers |
| `backend/app/services/` | Routing, authentication, retrieval, and SQL-RAG orchestration; `llm/` holds the shared LLM interface/factory and provider adapters |
| `backend/app/ingestion/` | Docling chunks, LangChain documents/embeddings, and Qdrant index-state operations; never expose ingestion through FastAPI |
| `backend/scripts/` | Explicit batch commands, including document ingestion |
| `backend/app/repositories/` | SQLite data access; `mediassist_repository.py` remains read-only and user authentication access is separate; never call an LLM |
| `backend/app/api/` | Future FastAPI endpoints; do not add until requested |
| `backend/tests/` | Standard-library `unittest` coverage |

## Routing decision

- Phase 1 uses `route_question()` with `KeywordRoutingStrategy`.
- SQL RAG requires both an allowed role and an analytical phrase plus a database-domain term.
- Preserve the `RoutingStrategy` interface when adding semantic or LLM routing.
- Routing does not replace authorization; SQL access remains limited by `can_use_sql_rag()`.

## Data and security

- Database: `backend/data/mediassist.db`.
- Authentication database: `backend/data/medibot_auth.db`, created only by the explicit demo-user bootstrap command.
- Use `execute_read_only_query()` for database access.
- Only `SELECT` queries against the policy-approved tables are allowed.
- Keep credentials in `backend/.env`; never commit or display secrets.
- Select the LLM provider with `REQUIRED_LLM_MODEL_GROUP`; create provider clients only through `create_llm_client()`.
- Keep provider model IDs in the matching provider-prefixed `.env` variables.
- Do not make a real LLM API call without explicit user approval after key rotation.
- Use LangChain `Document` and `QdrantVectorStore` in `HYBRID` mode for document-vector writes; direct `qdrant-client` is limited to collection setup and incremental-state checks.
- Do not implement public signup for this internal assignment; seed only the five demo users through an explicit command, and read an authenticated role from the signed token in future FastAPI endpoints.

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
