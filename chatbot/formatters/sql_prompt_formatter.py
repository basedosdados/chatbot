from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from chatbot.agents.prompts import (SQL_AGENT_BASE_SYSTEM_PROMPT,
                                    SQL_AGENT_SYSTEM_PROMPT)

from .prompt_formatter import BasePromptFormatter


class SQLExample(BaseModel):
    """A single example pairing a natural language question with its corresponding SQL query.

    Attributes:
        question (str): A natural language question.
        query (str): The SQL query that answers the question.
    """
    question: str = Field(description="A natural language question.")
    query: str = Field(description="The SQL query that answers the question.")

class SQLPromptContext(BaseModel):
    query: str = Field(description="The user's question")
    selected_datasets: list[str] = Field(description="A list of the selected dataset names")

class SQLPromptFormatter(BasePromptFormatter[SQLPromptContext]):
    """Default SQL prompt formatter that retrieves few-shot SQL examples
    from a vector store and builds system prompts for SQL query generation.

    Args:
        vector_store (VectorStore | None, optional):
            A langchain `VectorStore` instance to fetch example documents.
            If `None`, no few-shot examples will be retrieved. Defaults to `None`.
        top_k (int, optional):
            Number of similar examples to fetch for few-shot prompting. Defaults to `4`.
    """

    def __init__(self, vector_store: VectorStore | None = None, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def _get_examples(self, context: SQLPromptContext) -> list[SQLExample]:
        """Retrieve example SQL queries relevant to a given user question.

        Args:
            context (SQLPromptContext): A `SQLPromptContext` instance, containing
                                        the user's question and a list of the
                                        selected datasets names.

        Returns:
            list[SQLExample]: A list of `SQLExample` instances, each containing
                              a sample question and its corresponding SQL query.
        """
        if self.vector_store is None:
            return []

        if context.selected_datasets:
            query_filter = {"dataset_name": {"$in": context.selected_datasets}}
        else:
            query_filter = None

        examples = self.vector_store.similarity_search(
            context.query, self.top_k, filter=query_filter
        )

        return [
            SQLExample(
                question=ex.page_content,
                query=ex.metadata["query"],
            )
            for ex in examples
        ]

    async def _aget_examples(self, context: SQLPromptContext) -> list[SQLExample]:
        """Asynchronously retrieve example SQL queries relevant to a given user question.

        Args:
            context (SQLPromptContext): A `SQLPromptContext` instance, containing
                                        the user's question and a list of the
                                        selected datasets names.

        Returns:
            list[SQLExample]: A list of `SQLExample` instances, each containing
                              a sample question and its corresponding SQL query.
        """
        if self.vector_store is None:
            return []

        if context.selected_datasets:
            query_filter = {"dataset_name": {"$in": context.selected_datasets}}
        else:
            query_filter = None

        examples = await self.vector_store.asimilarity_search(
            context.query, self.top_k, filter=query_filter
        )

        return [
            SQLExample(
                question=ex.page_content,
                query=ex.metadata["query"],
            )
            for ex in examples
        ]

    def build_system_prompt(self, context: SQLPromptContext) -> str:
        """Build a system prompt for SQL query generation.

        Args:
            context (SQLPromptContext): A `SQLPromptContext` instance, containing
                                        the user's question and a list of the
                                        selected datasets names.

        Returns:
            str: A system prompt string. If no few‑shot examples are retrieved from the vector store,
                 returns the base system prompt. Otherwise, returns the few‑shot prompt template
                 populated with formatted examples.
        """
        examples = self._get_examples(context)

        if not examples:
            return SQL_AGENT_BASE_SYSTEM_PROMPT

        few_shot_examples = "\n\n".join(
            f"Question: {ex.question}\nSQL Query:\n```sql\n{ex.query}\n```"
            for ex in examples
        )

        return SQL_AGENT_SYSTEM_PROMPT.format(examples=few_shot_examples)

    async def abuild_system_prompt(self, context: SQLPromptContext) -> str:
        """Asynchronously build a system prompt for SQL query generation.

        Args:
            context (SQLPromptContext): A `SQLPromptContext` instance, containing
                                        the user's question and a list of the
                                        selected datasets names.

        Returns:
            str: A system prompt string. If no few‑shot examples are retrieved from the vector store,
                 returns the base system prompt. Otherwise, returns the few‑shot prompt template
                 populated with formatted examples.
        """
        examples = await self._aget_examples(context)

        if not examples:
            return SQL_AGENT_BASE_SYSTEM_PROMPT

        few_shot_examples = "\n\n".join(
            f"Question: {ex.question}\nSQL Query:\n```sql\n{ex.query}\n```"
            for ex in examples
        )

        return SQL_AGENT_SYSTEM_PROMPT.format(examples=few_shot_examples)
