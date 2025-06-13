from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from chatbot.agents.prompts import (CHART_PREPROCESS_BASE_SYSTEM_PROMPT,
                                    CHART_PREPROCESS_SYSTEM_PROMPT)

from .prompt_formatter import BasePromptFormatter

EXAMPLE_TEMPLATE = """<example>
User question: {user_question}

<query_results>
{query_results}
</query_results>

Output: {output}
</example>"""

class PreProcessingExample(BaseModel):
    """A single example containing a natural language question, its corresponding
    SQL query results and the preprocessed query results.

    Attributes:
        question (str): A natural language question.
        query_results (str): The query results that asnwers the question.
        output (str): The preprocessed query results ready for visualization.
    """
    question: str = Field(description="A natural language question.")
    query_results: str = Field(description="The query results that asnwers the question.")
    output: str = Field(description="The preprocessed query results ready for visualization.")

class VizPromptContext(BaseModel):
    query: str = Field(description="The user's question")

class VizPromptFormatter(BasePromptFormatter[VizPromptContext]):
    """Default prompt formatter that retrieves few-shot examples from a vector store
    and builds system prompts to guide data preprocessing for visualization.

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

    def _get_examples(self, context: VizPromptContext) -> list[PreProcessingExample]:
        """Retrieve example preprocessed data relevant to a given user question.

        Args:
            context (VizPromptContext): A `VizPromptContext` instance, containing
                                        the user's question, the query results and
                                        the preprocessed query results.

        Returns: list[PreProcessingExample]: A list of `PreProcessingExample` instances.
        """
        if self.vector_store is None:
            return []

        examples = self.vector_store.similarity_search(context.query, self.top_k)

        return [
            PreProcessingExample(
                question=ex.page_content,
                query=ex.metadata["query"],
            )
            for ex in examples
        ]

    async def _aget_examples(self, context: VizPromptContext) -> list[PreProcessingExample]:
        """Asynchronously retrieve example preprocessed data relevant to a given user question.

        Args:
            context (VizPromptContext): A `VizPromptContext` instance, containing
                                        the user's question, the query results and
                                        the preprocessed query results.

        Returns: list[PreProcessingExample]: A list of `PreProcessingExample` instances.
        """
        if self.vector_store is None:
            return []

        examples = await self.vector_store.asimilarity_search(context.query, self.top_k)

        return [
            PreProcessingExample(
                question=ex.page_content,
                query_results=ex.metadata["query_results"],
                output=ex.metadata["query_results_preprocessed"]
            )
            for ex in examples
        ]

    def build_system_prompt(self, context: VizPromptContext) -> str:
        """Build a system prompt for SQL query generation.

        Args:
            context (VizPromptContext): A `VizPromptContext` instance, containing
                                        the user's question, the query results and
                                        the preprocessed query results.

        Returns:
            str: A system prompt string. If no few‑shot examples are retrieved from the vector store,
                 returns the base system prompt. Otherwise, returns the few‑shot prompt template
                 populated with formatted examples.
        """
        examples = self._get_examples(context)

        examples = "\n\n".join(
            EXAMPLE_TEMPLATE.format(
                user_question=e.question,
                query_results=e.query_results,
                output=e.output
            )
            for e in examples
        )

        if examples:
            return CHART_PREPROCESS_SYSTEM_PROMPT.format(examples=examples)

        return CHART_PREPROCESS_BASE_SYSTEM_PROMPT

    async def abuild_system_prompt(self, context: VizPromptContext) -> str:
        """Asynchronously build a system prompt for SQL query generation.

        Args:
            context (VizPromptContext): A `VizPromptContext` instance, containing
                                        the user's question, the query results and
                                        the preprocessed query results.

        Returns:
            str: A system prompt string. If no few‑shot examples are retrieved from the vector store,
                 returns the base system prompt. Otherwise, returns the few‑shot prompt template
                 populated with formatted examples.
        """
        examples = await self._aget_examples(context)

        examples = "\n\n".join(
            EXAMPLE_TEMPLATE.format(
                user_question=e.question,
                query_results=e.query_results,
                output=e.output
            )
            for e in examples
        )

        if examples:
            return CHART_PREPROCESS_SYSTEM_PROMPT.format(examples=examples)

        return CHART_PREPROCESS_BASE_SYSTEM_PROMPT
