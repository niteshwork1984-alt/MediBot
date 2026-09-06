"""Unit tests for configuration-driven LLM provider and model selection."""

import unittest
from unittest.mock import patch

from app.core.config import (
    ConfigurationError,
    LLMModelGroup,
    LLMModelPurpose,
    load_hybrid_retrieval_configuration,
    load_llm_configuration,
)


# This test class verifies provider-prefixed settings are selected from configuration.
class LLMConfigurationTests(unittest.TestCase):
    # This helper returns complete in-memory configuration for one provider group.
    def _environment_for(self, group: str) -> dict[str, str]:
        return {
            "REQUIRED_LLM_MODEL_GROUP": group,
            f"{group}_SQL_TEST_MODEL": "test-model",
            f"{group}_SQL_MODEL": "sql-model",
            f"{group}_ANSWER_MODEL": "answer-model",
        }

    # This test verifies Groq settings are loaded when the Groq group is selected.
    def test_loads_groq_prefixed_models(self) -> None:
        configuration = load_llm_configuration(self._environment_for("GROQ"))

        self.assertEqual(configuration.model_group, LLMModelGroup.GROQ)
        self.assertEqual(
            configuration.model_for(LLMModelPurpose.SQL_TEST), "test-model"
        )
        self.assertEqual(configuration.model_for(LLMModelPurpose.SQL), "sql-model")
        self.assertEqual(
            configuration.model_for(LLMModelPurpose.ANSWER), "answer-model"
        )

    # This test verifies OpenAI can use its own model settings before its client exists.
    def test_loads_openai_prefixed_models(self) -> None:
        configuration = load_llm_configuration(self._environment_for("OPENAI"))

        self.assertEqual(configuration.model_group, LLMModelGroup.OPENAI)
        self.assertEqual(configuration.sql_model, "sql-model")

    # This test verifies an unknown provider group produces an actionable error.
    def test_rejects_unknown_model_group(self) -> None:
        environment = self._environment_for("UNSUPPORTED")

        with self.assertRaisesRegex(ConfigurationError, "Unsupported"):
            load_llm_configuration(environment)

    # This test verifies the selected provider must define every model purpose.
    def test_requires_all_selected_provider_model_settings(self) -> None:
        environment = {"REQUIRED_LLM_MODEL_GROUP": "GROQ"}

        with self.assertRaisesRegex(ConfigurationError, "GROQ_SQL_TEST_MODEL"):
            load_llm_configuration(environment)

    # This test verifies process startup environment variables are the source used at runtime.
    def test_loads_models_from_startup_environment(self) -> None:
        environment = self._environment_for("GROQ")
        environment["GROQ_SQL_MODEL"] = "startup-sql-model"

        with patch.dict("app.core.config.os.environ", environment, clear=True):
            configuration = load_llm_configuration()

        self.assertEqual(configuration.sql_model, "startup-sql-model")


# This test class verifies the configuration limits applied to Hybrid RAG retrieval.
class HybridRetrievalConfigurationTests(unittest.TestCase):
    # This test verifies defaults make a small retrieval request while retaining a larger safety ceiling.
    def test_loads_default_retrieval_limits(self) -> None:
        configuration = load_hybrid_retrieval_configuration({})

        self.assertEqual(configuration.default_limit, 5)
        self.assertEqual(configuration.max_limit, 20)

    # This test verifies a default above the safety maximum fails during startup configuration.
    def test_rejects_default_limit_above_maximum(self) -> None:
        environment = {
            "HYBRID_RAG_DEFAULT_LIMIT": "6",
            "HYBRID_RAG_MAX_LIMIT": "5",
        }

        with self.assertRaisesRegex(ConfigurationError, "cannot exceed"):
            load_hybrid_retrieval_configuration(environment)
