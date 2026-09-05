"""Groq implementation of MediBot's common LLM interface."""

from typing import Any

from app.core.config import (
    ConfigurationError,
    LLMConfiguration,
    LLMModelGroup,
    LLMModelPurpose,
    load_provider_api_key,
)
from app.services.llm.llm_client import LLMClient


# This concrete client adapts MediBot's interface to Groq's Python SDK.
class GroqLLMClient(LLMClient):
    """LLM client implementation for Groq-hosted models."""

    # This constructor receives configuration and accepts an optional fake SDK for tests.
    def __init__(
        self,
        configuration: LLMConfiguration,
        groq_client: Any | None = None,
    ) -> None:
        """Create a Groq client using the selected provider configuration."""
        if configuration.model_group is not LLMModelGroup.GROQ:
            raise ConfigurationError("GroqLLMClient requires REQUIRED_LLM_MODEL_GROUP=GROQ.")

        self._configuration = configuration
        self._client = groq_client or self._create_sdk_client()

    # This private method imports the SDK only when the real Groq client is needed.
    def _create_sdk_client(self) -> Any:
        """Create the official Groq SDK client using the local API key."""
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError(
                "Groq SDK is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error

        return Groq(api_key=load_provider_api_key(LLMModelGroup.GROQ))

    # This method sends prompts to Groq with the model chosen for the requested purpose.
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: LLMModelPurpose,
    ) -> str:
        """Generate a non-empty Groq chat-completion response."""
        completion = self._client.chat.completions.create(
            model=self._configuration.model_for(purpose),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        return content
