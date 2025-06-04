import codecs
import uuid
from typing import Any

import sqlparse
from langchain.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger

from chatbot.agents import SQLAgent
from chatbot.databases import Database

from .datatypes import SQLAssistantMessage, UserMessage


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

    @staticmethod
    def _format_response(response: dict[str, Any]) -> dict[str, Any]:
        """Formats the response that will be presented to the user

        Args:
            response (dict[str, list]): The model's response

        Returns:
            tuple[str, str|None]: The final answer and the generated sql queries if any
        """
        answer = response["final_answer"]
        answer = codecs.escape_decode(answer)[0]
        answer = answer.decode("utf-8").strip()

        sql_queries = []

        for item in response.get("sql_queries", []):
            sql_query = sqlparse.format(
                item.content,
                reindent=True,
                keyword_case="upper"
            )
            sql_queries.append(sql_query)

        formatted_response = {
            "content": answer,
            "sql_queries": sql_queries or None,
        }

        return formatted_response

    async def invoke(self, message: str, config: dict|None) -> SQLAssistantMessage:
        """Asynchronously sends a user message to the `SQLAgent` and returns its response

        Args:
            message (UserMessage): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLAssistantMessage: The generated response
        """
        if "run_id" not in config:
            config["run_id"] = str(uuid.uuid4())

        response = await self.sql_agent.ainvoke(message, config)
        response = self._format_response(response)

        response["id"] = config["run_id"]

        return SQLAssistantMessage(**response)

    async def clear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        logger.info(f"Clearing memory for thread {thread_id}")
        await self.sql_agent.aclear_thread(thread_id)
