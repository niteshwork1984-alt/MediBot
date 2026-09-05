"""Unit tests for the provider interface and Groq client adapter."""

import unittest

from app.core.config import (
    LLMConfiguration,
    LLMModelGroup,
    LLMModelPurpose,
)
from app.services.llm.groq.groq_llm_client import GroqLLMClient
from app.services.llm.llm_client import LLMClient, create_llm_client


# This fake completion object mirrors only the Groq response fields our client reads.
class FakeCompletion:
    """Minimal fake Groq completion response used by unit tests."""

    # This constructor builds the nested choices/message/content response shape.
    def __init__(self, content: str | None) -> None:
        """Store a fake completion message in Groq-compatible attributes."""
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


# This fake completions object records requests instead of sending a Groq API call.
class FakeCompletions:
    """Fake nested Groq SDK component for completion creation."""

    # This constructor prepares a response and a place to capture the latest request.
    def __init__(self, response_content: str | None = "Generated text") -> None:
        """Create a fake completion endpoint with deterministic output."""
        self.response_content = response_content
        self.request: dict[str, object] | None = None

    # This method records Groq request arguments and returns a local fake response.
    def create(self, **kwargs: object) -> FakeCompletion:
        """Capture a request without performing network I/O."""
        self.request = kwargs
        return FakeCompletion(self.response_content)


# This fake SDK client exposes the same nested chat.completions shape as Groq.
class FakeGroqSDK:
    """In-memory replacement for the official Groq SDK client."""

    # This constructor connects the fake nested chat and completions objects.
    def __init__(self, completions: FakeCompletions) -> None:
        """Build the small portion of Groq's SDK shape used by GroqLLMClient."""
        self.chat = type("Chat", (), {"completions": completions})()


# This test class verifies Groq behavior through fakes, without API keys or network calls.
class GroqLLMClientTests(unittest.TestCase):
    # This setup method builds Groq configuration and fake SDK components for each test.
    def setUp(self) -> None:
        self.configuration = LLMConfiguration(
            model_group=LLMModelGroup.GROQ,
            sql_test_model="openai/gpt-oss-20b",
            sql_model="openai/gpt-oss-120b",
            answer_model="openai/gpt-oss-20b",
        )
        self.completions = FakeCompletions()
        self.client = GroqLLMClient(
            self.configuration,
            groq_client=FakeGroqSDK(self.completions),
        )

    # This test verifies the Groq client explicitly implements the shared LLM interface.
    def test_groq_client_implements_llm_interface(self) -> None:
        self.assertIsInstance(self.client, LLMClient)

    # This test verifies SQL generation uses the configured SQL model and deterministic temperature.
    def test_uses_sql_model_for_sql_purpose(self) -> None:
        answer = self.client.generate(
            "Generate SQLite SQL.",
            "How many claims were escalated?",
            LLMModelPurpose.SQL,
        )

        self.assertEqual(answer, "Generated text")
        self.assertEqual(
            self.completions.request["model"], "openai/gpt-oss-120b"
        )
        self.assertEqual(self.completions.request["temperature"], 0)

    # This test verifies answer generation uses the separately configured answer model.
    def test_uses_answer_model_for_answer_purpose(self) -> None:
        self.client.generate("Explain rows.", "Rows: []", LLMModelPurpose.ANSWER)

        self.assertEqual(
            self.completions.request["model"], "openai/gpt-oss-20b"
        )

    # This test verifies an empty provider response is surfaced as a clear runtime error.
    def test_rejects_empty_provider_response(self) -> None:
        empty_client = GroqLLMClient(
            self.configuration,
            groq_client=FakeGroqSDK(FakeCompletions(response_content=None)),
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            empty_client.generate("System", "Question", LLMModelPurpose.SQL)

    # This test verifies the factory explains that an unimplemented provider cannot be used yet.
    def test_factory_rejects_unimplemented_provider(self) -> None:
        openai_configuration = LLMConfiguration(
            model_group=LLMModelGroup.OPENAI,
            sql_test_model="test",
            sql_model="sql",
            answer_model="answer",
        )

        with self.assertRaisesRegex(NotImplementedError, "implementation has not been added"):
            create_llm_client(openai_configuration)
