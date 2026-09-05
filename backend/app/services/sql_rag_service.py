"""SQL-RAG orchestration: LLM SQL generation, safe SQLite access, and LLM answers."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import LLMModelPurpose
from app.core.permissions import can_use_sql_rag
from app.repositories.mediassist_repository import execute_read_only_query
from app.services.llm.llm_client import LLMClient, create_llm_client


_SQL_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SELECT_START = re.compile(r"\bSELECT\b", re.IGNORECASE)
_SQL_SCHEMA = """claims(
    claim_id TEXT, patient_id TEXT, patient_name TEXT, department TEXT,
    claim_type TEXT, diagnosis_code TEXT, insurer TEXT, claimed_amount REAL,
    approved_amount REAL, status TEXT, submitted_date TEXT, resolved_date TEXT
)
maintenance_tickets(
    ticket_id TEXT, equipment_name TEXT, equipment_id TEXT, category TEXT,
    campus TEXT, issue_type TEXT, fault_code TEXT, raised_by TEXT,
    raised_date TEXT, resolved_date TEXT, status TEXT, resolution_note TEXT
)"""


# This immutable value makes the generated SQL, database rows, and final answer inspectable together.
@dataclass(frozen=True)
class SQLRagResult:
    """The intermediate and final values produced by one SQL-RAG request."""

    sql: str
    rows: list[dict]
    answer: str


# This function extracts a single SQL statement from a model response before repository validation.
def clean_generated_sql(raw_output: str) -> str:
    """Return SQL from a markdown-fenced or prefixed LLM response."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("The LLM must return a non-empty SQL response.")

    fenced_sql = _SQL_FENCE.search(raw_output)
    candidate = fenced_sql.group(1).strip() if fenced_sql else raw_output.strip()
    select_match = _SELECT_START.search(candidate)
    if not select_match:
        raise ValueError("The LLM response does not contain a SELECT statement.")

    statement = candidate[select_match.start() :].strip()
    if ";" in statement:
        statement = statement.split(";", maxsplit=1)[0].strip()

    if not statement:
        raise ValueError("The LLM response does not contain SQL after SELECT.")
    return statement


# This service coordinates the SQL-RAG workflow and enforces the SQL-RAG role policy.
class SQLRagService:
    """Role-aware application service for analytical questions over MediAssist SQLite data."""

    # This constructor accepts dependencies so tests can use fakes instead of Groq or SQLite.
    def __init__(
        self,
        llm_client: LLMClient,
        query_executor: Callable[[str], list[dict]] = execute_read_only_query,
    ) -> None:
        """Store the provider client and the safe read-only database operation."""
        self._llm_client = llm_client
        self._query_executor = query_executor

    # This method runs the three required SQL-RAG steps for one authorized user question.
    def answer_with_details(self, question: str, role: str) -> SQLRagResult:
        """Generate safe SQL, read the database, and generate a natural-language answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if not can_use_sql_rag(role):
            raise PermissionError(f"Role '{role}' is not allowed to use SQL RAG.")

        raw_sql = self._llm_client.generate(
            system_prompt=(
                "You write SQLite SELECT queries for MediAssist. Use only the approved "
                "tables and columns below. Return exactly one SELECT statement and no prose.\n"
                f"Schema:\n{_SQL_SCHEMA}"
            ),
            user_prompt=question,
            purpose=LLMModelPurpose.SQL,
        )
        sql = clean_generated_sql(raw_sql)
        rows = self._query_executor(sql)
        answer = self._llm_client.generate(
            system_prompt=(
                "You answer MediAssist analytical questions using only the supplied SQLite "
                "query results. If the results are empty, say that no matching records were found."
            ),
            user_prompt=(
                f"Question: {question}\n"
                f"SQL query: {sql}\n"
                f"Query results (JSON): {json.dumps(rows, default=str)}"
            ),
            purpose=LLMModelPurpose.ANSWER,
        )
        return SQLRagResult(sql=sql, rows=rows, answer=answer)

    # This convenience method returns only the final string required by the assignment chain.
    def answer(self, question: str, role: str) -> str:
        """Return the natural-language SQL-RAG answer for one authorized role."""
        return self.answer_with_details(question, role).answer


# This plain function is the assignment-required SQL-RAG entry point for an already-authorized call.
def sql_rag_chain(question: str) -> str:
    """Answer a billing SQL-RAG question using the configured provider and database."""
    service = SQLRagService(create_llm_client())
    return service.answer(question, role="billing_executive")
