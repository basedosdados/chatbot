from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")

class BasePromptFormatter(ABC, Generic[T]):
    """Generic base class for building system prompts from domain-specific context.

    It is generic over the `T` type, allowing different agents to pass
    different, specific context objects for prompt generation.

    Subclasses must override at least the `build_system_prompt` and/or `abuild_system_prompt`
    methods. They may also override example retrieval hooks for few-shot prompting. All the
    available methods are described below:

    1. `_get_examples(query: str, context: T)`
       - Retrieve example objects for few-shot prompting.
    2. `_aget_examples(query: str, context: T)`
       - Asynchronously return example objects for few-shot prompting.
    3. `build_system_prompt(query: str, context: T)`
       - Compose and return the final system prompt string.
    4. `abuild_system_prompt(query: str, context: T)`
       - Asynchronously compose and return the final system prompt string.

    By default, no examples are returned in the example retrieval hooks.
    """

    def _get_examples(self, context: T) -> Any:
        """Retrieve example objects relevant to a given context.
        Optionally overrides for few-shot prompt building.

        Args:
            context (T): Generic context object to be used during retrieval.

        Returns:
            Any: Any example objects. Defaults to an empty list.
        """
        return []

    async def _aget_examples(self, context: T) -> Any:
        """Asynchronously retrieve example objects relevant to a given context.
        Optionally overrides for few-shot prompt building.

        Args:
            context (T): Generic context object to be used during retrieval.

        Returns:
            Any example objects. Defaults to an empty list.
        """
        return []

    @abstractmethod
    def build_system_prompt(self, context: T) -> str:
        """Compose the system prompt for SQL generation.

        Args:
            context (T): Generic context object to be used during system prompt building.

        Returns:
            str: The system prompt to send to the LLM.
                 To include examples or custom templates, override this method:
                    1. Call self._get_examples(context) to retrieve examples.
                    2. Format and inject them into your prompt template.
        """
        ...

    @abstractmethod
    async def abuild_system_prompt(self, context: T) -> str:
        """Asynchronously compose the system prompt for SQL generation.

        Args:
            context (T): Generic context object to be used during system prompt building.

        Returns:
            str: The system prompt to send to the LLM.
                 To include examples or custom templates, override this method:
                    1. Call self._get_examples(context) to retrieve examples.
                    2. Format and inject them into your prompt template.
        """
        ...
