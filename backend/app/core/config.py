"""Backend configuration and provider-specific model selection."""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


BACKEND_DIRECTORY = Path(__file__).resolve().parents[2]
DATABASE_PATH = BACKEND_DIRECTORY / "data" / "mediassist.db"

# This loads local development settings only for values absent from the startup environment.
load_dotenv(BACKEND_DIRECTORY / ".env", override=False)

# This default URL points to the local Qdrant container started through docker-compose.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# This default collection name identifies the first document index contract version.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medibot_documents_v1")

# This version changes when chunking or embedding settings require a complete new index.
INDEX_VERSION = os.getenv("INDEX_VERSION", "v1")


# This exception explains a missing or invalid environment configuration value.
class ConfigurationError(ValueError):
    """Raised when required backend configuration is missing or invalid."""


# This enum lists the LLM providers the application can select from configuration.
class LLMModelGroup(StrEnum):
    """Supported provider groups for model selection."""

    GROQ = "GROQ"
    OPENAI = "OPENAI"
    CLAUDE = "CLAUDE"


# This enum identifies why the application needs an LLM model.
class LLMModelPurpose(StrEnum):
    """Supported model purposes in the SQL-RAG workflow."""

    SQL_TEST = "SQL_TEST"
    SQL = "SQL"
    ANSWER = "ANSWER"


# This immutable object groups the selected provider and its three configured model IDs.
@dataclass(frozen=True)
class LLMConfiguration:
    """Resolved LLM provider and model IDs for one application run."""

    model_group: LLMModelGroup
    sql_test_model: str
    sql_model: str
    answer_model: str

    # This method returns the model ID that corresponds to one workflow purpose.
    def model_for(self, purpose: LLMModelPurpose) -> str:
        """Return the configured model for a requested LLM purpose."""
        models = {
            LLMModelPurpose.SQL_TEST: self.sql_test_model,
            LLMModelPurpose.SQL: self.sql_model,
            LLMModelPurpose.ANSWER: self.answer_model,
        }
        return models[purpose]


# This helper reads one required setting and raises a clear error when it is absent.
def _required_setting(name: str, environment: Mapping[str, str]) -> str:
    """Return a required non-empty environment value."""
    value = environment.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required configuration: {name}")
    return value


# This function resolves the selected provider and the matching provider-prefixed models.
def load_llm_configuration(
    environment: Mapping[str, str] | None = None,
) -> LLMConfiguration:
    """Load LLM configuration from the environment or a supplied test mapping."""
    source = os.environ if environment is None else environment
    group_name = _required_setting("REQUIRED_LLM_MODEL_GROUP", source).upper()

    try:
        model_group = LLMModelGroup(group_name)
    except ValueError as error:
        supported_groups = ", ".join(group.value for group in LLMModelGroup)
        raise ConfigurationError(
            f"Unsupported REQUIRED_LLM_MODEL_GROUP: {group_name}. "
            f"Supported values: {supported_groups}."
        ) from error

    prefix = model_group.value
    return LLMConfiguration(
        model_group=model_group,
        sql_test_model=_required_setting(f"{prefix}_SQL_TEST_MODEL", source),
        sql_model=_required_setting(f"{prefix}_SQL_MODEL", source),
        answer_model=_required_setting(f"{prefix}_ANSWER_MODEL", source),
    )


# This function reads the API key belonging to the provider selected in configuration.
def load_provider_api_key(
    model_group: LLMModelGroup,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Load a selected provider's API key without logging its value."""
    source = os.environ if environment is None else environment
    return _required_setting(f"{model_group.value}_API_KEY", source)
