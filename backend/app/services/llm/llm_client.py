"""Provider-independent LLM interface and client factory."""

from abc import ABC, abstractmethod

from app.core.config import (
    LLMConfiguration,
    LLMModelGroup,
    LLMModelPurpose,
    load_llm_configuration,
)


# This abstract base class is the Java-like interface shared by all LLM providers.
class LLMClient(ABC):
    """Common interface for Groq, OpenAI, Claude, and future LLM clients."""

    # This abstract method defines one provider-independent text-generation operation.
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: LLMModelPurpose,
    ) -> str:
        """Generate text using the configured model for a workflow purpose."""
        raise NotImplementedError


# This factory selects the provider implementation from REQUIRED_LLM_MODEL_GROUP.
def create_llm_client(configuration: LLMConfiguration | None = None) -> LLMClient:
    """Create the configured provider client or explain an unimplemented provider."""
    selected_configuration = configuration or load_llm_configuration()

    if selected_configuration.model_group is LLMModelGroup.GROQ:
        # This local import prevents a circular dependency between interface and adapter.
        from app.services.llm.groq.groq_llm_client import GroqLLMClient

        return GroqLLMClient(selected_configuration)

    raise NotImplementedError(
        f"{selected_configuration.model_group.value} is configured, but its LLM client "
        "implementation has not been added yet."
    )
