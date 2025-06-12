from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")

class BasePromptFormatter(ABC, Generic[T]):
    """A generic base class for building system prompts.

    It is generic over the `T` type, allowing different agents to pass
    different, specific context/metadata objects for prompt generation.

    Subclasses should override one or more of the following hooks to customize prompt behavior:

    1. `_get_examples(query: str, metadata: T)`
       - Return a sequence of example objects to include in the prompt.
    2. `_aget_examples(query: str, metadata: T)`
       - Asynchronously return example objects for the prompt.
    3. `build_system_prompt(query: str, metadata: T)`
       - Compose and return the final system prompt string, given the retrieved examples.
    4. `abuild_system_prompt(query: str, metadata: T)`
       - Compose and return the final system prompt string, given the asynchronously retrieved examples.

    By default, no examples are returned, and the base system prompt is used. Override any
    of these methods in your subclass to implement custom few-shot logic or prompt templates.
    """

    def _get_examples(self, context: T) -> Any:
        """Retrieve example objects relevant to a given user message.
        Optionally overrides for few-shot prompt building.

        Args:
            query (str): The user's natural language question.
            metadata (T): A generic object of type `T` to be used during retrieval.

        Returns:
            Any: Any example objects. Default implementation returns an empty list.
        """
        return []

    async def _aget_examples(self, context: T) -> Any:
        """Asynchronously retrieve example objects relevant to a given user message
        Optionally overrides for few-shot prompt building.

        Args:
            query (str): The user's natural language question.
            metadata (T): A generic object of type `T` to be used during retrieval.

        Returns:
            Any example objects. Default implementation returns an empty list.
        """
        return []

    @abstractmethod
    def build_system_prompt(self, context: T) -> str:
        """Compose the system prompt for SQL generation.

        Args:
            query (str): The user's natural language question.
            metadata (T): A generic object of type `T` to be used during retrieval.

        Returns:
            str: The system prompt to send to the LLM. By default, returns an empty string.

                 To include examples or custom templates, override this method:
                    1. Call self._get_examples(query, metadata) to retrieve examples.
                    2. Format and inject them into your prompt template.
        """
        ...

    @abstractmethod
    async def abuild_system_prompt(self, context: T) -> str:
        """Asynchronously compose the system prompt for SQL generation.

        Args:
            metadata (T): A generic object of type `T` to be used during retrieval.

        Returns:
            str: The system prompt to send to the LLM. By default, returns an empty string.

                 To include examples or custom templates, override this method:
                    1. Call self._get_examples(query, metadata) to retrieve examples.
                    2. Format and inject them into your prompt template.
        """
        ...
