"""Unit tests for SQL-RAG orchestration without real Groq calls or database reads."""

import unittest

from app.core.config import LLMModelPurpose
from app.services.llm.llm_client import LLMClient
from app.services.sql_rag_service import SQLRagService, clean_generated_sql


# This fake LLM returns prearranged outputs and records each SQL-RAG prompt.
class FakeLLMClient(LLMClient):
    """In-memory LLM implementation for SQL-RAG service tests."""

    # This constructor stores deterministic outputs for sequential generate calls.
    def __init__(self, responses: list[str]) -> None:
        """Create a fake LLM with the next response for each request."""
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    # This method records prompts and returns the next fake response without network I/O.
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: LLMModelPurpose,
    ) -> str:
        """Return one configured response for the requested model purpose."""
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "purpose": purpose,
            }
        )
        return self._responses.pop(0)


# This test class verifies the complete SQL-RAG flow using only local fakes.
class SQLRagServiceTests(unittest.TestCase):
    # This test verifies markdown SQL is cleaned, executed, and explained in the required order.
    def test_answers_question_through_sql_and_result_explanation(self) -> None:
        llm_client = FakeLLMClient(
            [
                "Here is the query:\n```sql\nSELECT COUNT(*) AS claim_count FROM claims\n```",
                "There are 4 claims.",
            ]
        )
        executed_sql: list[str] = []

        # This local executor records SQL and returns a deterministic database result.
        def fake_query_executor(sql: str) -> list[dict]:
            executed_sql.append(sql)
            return [{"claim_count": 4}]

        service = SQLRagService(llm_client, fake_query_executor)
        result = service.answer_with_details("How many claims are there?", "admin")

        self.assertEqual(result.sql, "SELECT COUNT(*) AS claim_count FROM claims")
        self.assertEqual(executed_sql, [result.sql])
        self.assertEqual(result.rows, [{"claim_count": 4}])
        self.assertEqual(result.answer, "There are 4 claims.")
        self.assertEqual(llm_client.calls[0]["purpose"], LLMModelPurpose.SQL)
        self.assertEqual(llm_client.calls[1]["purpose"], LLMModelPurpose.ANSWER)
        self.assertIn('"claim_count": 4', llm_client.calls[1]["user_prompt"])

    # This test verifies unauthorized roles are rejected before any LLM or database call.
    def test_rejects_unauthorized_role_before_generating_sql(self) -> None:
        llm_client = FakeLLMClient(["SELECT * FROM claims"])
        service = SQLRagService(llm_client)

        with self.assertRaisesRegex(PermissionError, "not allowed"):
            service.answer("How many claims are there?", "nurse")

        self.assertEqual(llm_client.calls, [])

    # This test verifies non-empty question text is required before the workflow starts.
    def test_rejects_empty_question(self) -> None:
        service = SQLRagService(FakeLLMClient(["SELECT * FROM claims"]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            service.answer("  ", "admin")


# This test class verifies the boundary between raw model output and repository validation.
class CleanGeneratedSQLTests(unittest.TestCase):
    # This test verifies a fenced SQL response is reduced to the SQL statement only.
    def test_extracts_sql_from_markdown_fence(self) -> None:
        raw_output = "```sqlite\nSELECT status, COUNT(*) FROM claims GROUP BY status;\n```"

        self.assertEqual(
            clean_generated_sql(raw_output),
            "SELECT status, COUNT(*) FROM claims GROUP BY status",
        )

    # This test verifies an LLM response without a SELECT statement is rejected.
    def test_rejects_response_without_select_statement(self) -> None:
        with self.assertRaisesRegex(ValueError, "SELECT"):
            clean_generated_sql("I cannot write a query.")
