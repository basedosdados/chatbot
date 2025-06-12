from abc import ABC
from typing import Any

from chatbot.agents.prompts import SQL_AGENT_BASE_SYSTEM_PROMPT


class BasePromptFormatter(ABC):
    """Base class for building system prompts for SQL generation, with optional examples.

    Subclasses should override one or more of the following hooks to customize prompt behavior:

    1. `_get_sql_examples(query: str, selected_datasets: list[str])`
       - Return a sequence of example objects to include in the prompt.
    2. `_aget_sql_examples(query: str, selected_datasets: list[str])`
       - Asynchronously return example objects for the prompt.
    3. `build_system_prompt(query: str, selected_datasets: list[str])`
       - Compose and return the final system prompt string, given the retrieved examples.
    4. `abuild_system_prompt(query: str, selected_datasets: list[str])`
       - Compose and return the final system prompt string, given the asynchronously retrieved examples.

    By default, no examples are returned, and the base system prompt is used. Override any
    of these methods in your subclass to implement custom few-shot logic or prompt templates.
    """

    def _get_sql_examples(self, query: str, selected_datasets: list[str]) -> Any:
        """Retrieve example objects relevant to a given user message.
        Optionally overrides for few-shot prompt building.

        Args:
            query (str): The user's natural language question.
            selected_datasets (list[str]): The selected dataset names. You could use them
                                           for filtering during example objects retrieval.

        Returns:
            Any: Any example objects. Default implementation returns an empty list.
        """
        return []

    async def _aget_sql_examples(self, query: str, selected_datasets: list[str]) -> Any:
        """Asynchronously retrieve example objects relevant to a given user message
        Optionally overrides for few-shot prompt building.

        Args:
            query (str): The user's natural language question.
            selected_datasets (list[str]): The selected dataset names. You could use them
                                           for filtering during example objects retrieval.

        Returns:
            Any example objects. Default implementation returns an empty list.
        """
        return []

    def build_system_prompt(self, query: str, selected_datasets: list[str]) -> str:
        """Compose the system prompt for SQL generation.

        Args:
            query (str): The user's natural language question.
            selected_datasets (list[str]): The selected dataset names. You could use them
                                           for filtering during example objects retrieval.

        Returns:
            str: The system prompt to send to the LLM. By default,
                 returns the base prompt with no few-shot examples.

                 To include examples or custom templates, override this method:
                 1. Call self._get_sql_examples(query) to retrieve examples.
                 2. Format and inject them into your prompt template.
        """
        return SQL_AGENT_BASE_SYSTEM_PROMPT

    async def abuild_system_prompt(self, query: str, selected_datasets: list[str]) -> str:
        """Asynchronously compose the system prompt for SQL generation.

        Args:
            query (str): The user's natural language question.
            selected_datasets (list[str]): The selected dataset names. You could use them
                                           for filtering during example objects retrieval.

        Returns:
            str: The system prompt to send to the LLM. By default,
                 returns the base prompt with no few-shot examples.

                 To include examples or custom templates, override this method:
                 1. Call self._get_sql_examples(query) to retrieve examples.
                 2. Format and inject them into your prompt template.
        """
        return SQL_AGENT_BASE_SYSTEM_PROMPT
