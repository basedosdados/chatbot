import codecs
import uuid
from typing import Any

import sqlparse
from langchain.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger

from chatbot.agents import RouterAgent, SQLAgent, VizAgent
from chatbot.databases import Database

from .datatypes import SQLVizAssistantMessage, UserMessage


class AsyncSQLVizAssistant:
    """Async LLM-powered assistant for querying and visualizing databases.

    Args:
        database (Database):
            A `Database` object implementing the `Database` protocol.
        model (BaseChatModel):
            A LangChain `ChatModel` with support to structured outputs and tool calling.
        checkpointer (AsyncPostgresSaver | None, optional):
            A checkpointer that will be used for persisting state across assistant's runs.
            If set to `None`, the assistant will not retain memory of previous messages.
            Defaults to `None`.
        sql_vector_store (VectorStore | None, optional):
            A vector store that contains examples for the `SQLAgent` LLM calls.
            If set to `None`, no examples will be used. Defaults to `None`.
        viz_vector_store (VectorStore | None, optional):
            A vector store that contains examples for the `VizAgent` LLM calls.
            If set to `None`, no examples will be used. Defaults to `None`.
        question_limit (int | None, optional):
            Maximum number of previous questions to keep in memory.
            If `None`, all questions are kept. Defaults to `5`.

    Raises:
        EnvironmentVariableError: If `model_uri` is not provided and `MODEL_URI` is not set.
    """

    def __init__(
        self,
        database: Database,
        model: BaseChatModel,
        checkpointer: AsyncPostgresSaver | None = None,
        sql_vector_store: VectorStore | None = None,
        viz_vector_store: VectorStore | None = None,
        question_limit: int | None = 5,
    ):
        if checkpointer is None:
            subgraph_checkpointer = None
        elif isinstance(checkpointer, AsyncPostgresSaver):
            subgraph_checkpointer = True
        else:
            raise TypeError(
                "`checkpointer` must be an instance of langgraph `PostgresSaver` "
                f"or `None`, but got `{type(checkpointer)}`."
            )

        if sql_vector_store is not None and not isinstance(sql_vector_store, VectorStore) \
        or viz_vector_store is not None and not isinstance(viz_vector_store, VectorStore):
            raise TypeError(
                "`sql_vector_store` and `viz_vector_store` must be instances of langchain `VectorStore` or `None`, "
                f"but got `sql_vector_store`: {type(sql_vector_store)}, `viz_vector_store`: {type(viz_vector_store)}."
            )

        sql_agent = SQLAgent(
            db=database,
            model=model,
            checkpointer=subgraph_checkpointer,
            vector_store=sql_vector_store,
            question_limit=question_limit
        )

        viz_agent = VizAgent(
            model=model,
            checkpointer=subgraph_checkpointer,
            vector_store=viz_vector_store,
            question_limit=question_limit
        )

        self.router_agent = RouterAgent(
            model=model,
            sql_agent=sql_agent,
            viz_agent=viz_agent,
            checkpointer=checkpointer,
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
            "chart": response.get("chart"),
        }

        return formatted_response

    async def invoke(self, message: UserMessage, thread_id: str|None=None) -> SQLVizAssistantMessage:
        """Asynchronously sends a user message to the `RouterAgent` and returns its response

        Args:
            message (UserMessage): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLVizAssistantMessage: The generated response
        """
        logger.info(f"Received message {message.id}: {message.content}")

        run_id = str(uuid.uuid4())

        config = {
            "run_id": run_id,
            "recursion_limit": 32
        }

        if thread_id is not None:
            config["configurable"] = {
                "thread_id": thread_id
            }

        try:
            response = await self.router_agent.ainvoke(message.content, config)
            response = self._format_response(response)
        except Exception:
            logger.exception(f"Error on responding message {message.id}:")
            response = {
                "content": f"Ops, algo deu errado! Ocorreu um erro inesperado. Por favor, tente novamente. "\
                    "Se o problema persistir, avise-nos. Obrigado pela paciência!",
            }

        response["id"] = run_id

        logger.info(f"Returning response for message {message.id}")

        return SQLVizAssistantMessage(**response)

    async def clear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        logger.info(f"Clearing memory for thread {thread_id}")
        await self.router_agent.aclear_thread(thread_id)
