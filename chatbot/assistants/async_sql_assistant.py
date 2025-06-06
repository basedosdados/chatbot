from langchain.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from chatbot.agents import SQLAgent
from chatbot.databases import Database

from .formatting import format_sql_agent_response
from .messages import SQLAssistantMessage


class AsyncSQLAssistant:
    """Async LLM-powered assistant for querying databases.

    Args:
        database (Database):
            A `Database` object implementing the `Database` protocol.
        model (BaseChatModel):
            A LangChain `ChatModel` with support to structured outputs and tool calling.
        checkpointer (AsyncPostgresSaver | None, optional):
            A checkpointer that will be used for persisting state across assistant's runs.
            If set to `None`, the assistant will not retain memory of previous messages.
            Defaults to `None`.
        vector_store (VectorStore | None, optional):
            A vector store that contains examples for the `SQLAgent` LLM calls.
            If set to `None`, no examples will be used. Defaults to `None`.
        question_limit (int | None, optional):
            Maximum number of previous questions to keep in memory.
            If `None`, all questions are kept. Defaults to `5`.

    Raises:
        EnvironmentVariableUnset: If `model_uri` is not provided and `MODEL_URI` is not set.
    """

    def __init__(
        self,
        database: Database,
        model: BaseChatModel,
        checkpointer: AsyncPostgresSaver | None = None,
        vector_store: VectorStore | None = None,
        question_limit: int | None = 5,
    ):
        if checkpointer is not None and not isinstance(checkpointer, AsyncPostgresSaver):
            raise TypeError(
                "`checkpointer` must be an instance of langgraph `AsyncPostgresSaver` "
                f"or `None`, but got `{type(checkpointer)}`."
            )

        if vector_store is not None and not isinstance(vector_store, VectorStore):
            raise TypeError(
                "`vector_store` must be an instance of langchain `VectorStore` "
                f"or `None`, but got `{type(vector_store)}`."
            )

        self.sql_agent = SQLAgent(
            db=database,
            model=model,
            checkpointer=checkpointer,
            vector_store=vector_store,
            question_limit=question_limit
        )

    async def invoke(self, message: str, config: dict|None=None) -> SQLAssistantMessage:
        """Asynchronously sends a user message to the `SQLAgent` and returns its response

        Args:
            message (str): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLAssistantMessage: The generated response
        """
        response = await self.sql_agent.ainvoke(message, config)
        response = format_sql_agent_response(response)

        if config is not None and "run_id" in config:
            response["id"] = config["run_id"]

        return SQLAssistantMessage(**response)

    async def clear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        await self.sql_agent.aclear_thread(thread_id)
