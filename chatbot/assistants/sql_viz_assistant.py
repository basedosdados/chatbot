import codecs
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import sqlparse
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from chatbot.agents import RouterAgent, SQLAgent, VizAgent
from chatbot.databases import Database
from chatbot.exceptions import EnvironmentVariableUnset
from chatbot.loguru_logging import get_logger
from chatbot.models import ModelFactory
from chatbot.storage import get_chroma_client

from .datatypes import ModelURI, SQLVizAssistantMessage, UserMessage


@asynccontextmanager
async def create_sqlviz_assistant(
    database: Database,
    db_url: str | None = None,
    model_uri: str | ModelURI | None = None,
    question_limit: int | None = 5,
    vector_store_url: str | None = None,
    sql_agent_collection: str | None = None,
    viz_agent_collection: str | None = None,
):
    """Yields a `SQLVizAssistant` instance with an async PostgreSQL checkpointer

    Args:
        database (Database):
            A `Database` object, i.e., an object that implements the `Database` protocol
        db_url (str | None, optional):
            The checkpointer database URL. If set to `None`, it will be obtained
            from the `DB_URL` env variable. Defaults to `None`.
        model_uri (str | ModelURI | None, optional):
            An URI for the LLM to be used by the assistant. It must be in the format
            `<provider>/<model_name>`. If set to `None`, it will be obtained
            from the `MODEL_URI` env variable. Defaults to `None`
        question_limit (int | None, optional):
            Number of questions to keep in memory. If set to `None`,
            all questions will be kept. Defaults to `5`
        vector_store_url (str | None, optional):
            The URL to a vector database that contains examples for use in LLM calls.
            If set to `None`, no examples will be used. Defaults to `None`
        sql_agent_collection (str | None, optional):
            Name of the collection that contains examples for the `SQLAgent`.
            If set to `None`, no examples will be used. Defaults to `None`
        viz_agent_collection (str | None, optional):
            Name of the collection that contains examples for the `VizAgent`.
            If set to `None`, no examples will be used. Defaults to `None`

    Yields:
        SQLVizAssistant: The assistant
    """
    db_url = db_url or os.getenv("DB_URL")

    if db_url is None:
        raise EnvironmentVariableUnset(
            "The checkpointer database URL was not passed and could not be inferred "
            "from the environment. Please pass a valid database URL to the `db_url` "
            "argument or set the `DB_URL` environment variable"
        )

    # Connection kwargs defined according to:
    # https://github.com/langchain-ai/langgraph/issues/2887
    # https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres
    conn_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0
    }

    async with AsyncConnectionPool(
        conninfo=db_url,
        max_size=8,
        kwargs=conn_kwargs
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        await checkpointer.setup()

        assistant = SQLVizAssistant(
            database=database,
            model_uri=model_uri,
            checkpointer=checkpointer,
            question_limit=question_limit,
            vector_store_url=vector_store_url,
            sql_agent_collection=sql_agent_collection,
            viz_agent_collection=viz_agent_collection,
        )

        yield assistant

class SQLVizAssistant:
    """LLM-powered assistant for querying and visualizing BigQuery datasets

    Args:
        database (Database):
            A `Database` object, i.e., an object that implements the `Database` protocol
        model_uri (str | ModelURI | None, optional):
            An URI for the LLM to be used. It must be in the format `<provider>/<model_name>`.
            For example, it could be `openai/gpt-4o` or `google/gemini-1.5-flash-001`. If set
            to `None`, it will be obtained from the `MODEL_URI` env variable. Defaults to `None`
        checkpointer (AsyncPostgresSaver | None, optional):
            A checkpointer that will be used for persisting state across assistant's runs.
            If set to `None`, the agent will have no memory. Defaults to `None`
        question_limit (int | None, optional):
            Number of questions to keep in memory. If set to `None`,
            all questions will be kept. Defaults to `5`
        vector_store_url (str | None, optional):
            The URL to a vector database that contains examples for use in LLM calls.
            If set to `None`, no examples will be used. Defaults to `None`
        sql_agent_collection (str | None, optional):
            Name of the collection that contains examples for the `SQLAgent`.
            If set to `None`, no examples will be used. Defaults to `None`
        viz_agent_collection (str | None, optional):
            Name of the collection that contains examples for the `VizAgent`.
            If set to `None`, no examples will be used. Defaults to `None`

    Raises:
        TypeError: If the checkpointer is not `None` or an instance of `AsyncPostgresSaver`
    """

    def __init__(
        self,
        database: Database,
        model_uri: str | ModelURI | None = None,
        checkpointer: PostgresSaver | AsyncPostgresSaver | None = None,
        question_limit: int | None = 5,
        vector_store_url: str | None = None,
        sql_agent_collection: str | None = None,
        viz_agent_collection: str | None = None,
    ):
        if isinstance(checkpointer, (PostgresSaver, AsyncPostgresSaver)):
            subgraph_checkpointer = True
        elif checkpointer is None:
            subgraph_checkpointer = None
        else:
            raise TypeError(
                "The checkpointer must be an instance of `PostgresSaver`, "
                f"`AsyncPostgresSaver` or `None`, but got `{type(checkpointer)}`"
            )

        model_uri = model_uri or os.getenv("MODEL_URI")

        if isinstance(model_uri, str):
            model_uri = ModelURI(model_uri)
        elif model_uri is None:
            raise EnvironmentVariableUnset(
                "The model URI was not passed and could not be inferred from the environment. "
                "Please pass a valid model URI to the `model_uri` argument or "
                "set the `MODEL_URI` environment variable"
            )
        elif not isinstance(model_uri, ModelURI):
            raise TypeError(
                f"`model_uri` must be of type `str` or `ModelURI`, got `{type(model_uri)}`"
            )

        vector_store_url = vector_store_url or os.getenv("VECTOR_DB_URL")
        sql_agent_collection = sql_agent_collection or os.getenv("SQL_AGENT_COLLECTION")
        viz_agent_collection = viz_agent_collection or os.getenv("VIZ_AGENT_COLLECTION")

        if vector_store_url is not None:
            sql_vector_store = get_chroma_client(
                url=vector_store_url,
                collection=sql_agent_collection
            )
            viz_vector_store = get_chroma_client(
                url=vector_store_url,
                collection=viz_agent_collection
            )
        else:
            sql_vector_store = None
            viz_vector_store = None

        self.model_uri = model_uri

        model = ModelFactory.from_model_uri(self.model_uri)

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

        self.logger = get_logger(self.__class__.__name__)

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

        for item in response["sql_queries"]:
            sql_query = sqlparse.format(
                item.content,
                reindent=True,
                keyword_case="upper"
            )
            sql_queries.append(sql_query)

        formatted_response = {
            "content": answer,
            "sql_queries": sql_queries or None,
            "chart": response["chart"],
        }

        return formatted_response

    def invoke(self, message: UserMessage, thread_id: str|None=None) -> SQLVizAssistantMessage:
        """Sends a user message to the `RouterAgent` and returns its response

        Args:
            message (UserMessage): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLVizAssistantMessage: The generated response
        """
        self.logger.info(f"Received message {message.id}: {message.content}")

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
            response = self.router_agent.invoke(message.content, config)
            response = self._format_response(response)
        except Exception:
            self.logger.exception(f"Error on responding message {message.id}:")
            response = {
                "content": f"Ops, algo deu errado! Ocorreu um erro inesperado. Por favor, tente novamente. "\
                    "Se o problema persistir, avise-nos. Obrigado pela paciência!",
            }

        response.update({
            "id": run_id,
            "model_uri": self.model_uri
        })

        self.logger.info(f"Returning response for message {message.id}")

        return SQLVizAssistantMessage(**response)

    async def ainvoke(self, message: UserMessage, thread_id: str|None=None) -> SQLVizAssistantMessage:
        """Asynchronously sends a user message to the `RouterAgent` and returns its response

        Args:
            message (UserMessage): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLVizAssistantMessage: The generated response
        """
        self.logger.info(f"Received message {message.id}: {message.content}")

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
            self.logger.exception(f"Error on responding message {message.id}:")
            response = {
                "content": f"Ops, algo deu errado! Ocorreu um erro inesperado. Por favor, tente novamente. "\
                    "Se o problema persistir, avise-nos. Obrigado pela paciência!",
            }

        response.update({
            "id": run_id,
            "model_uri": self.model_uri
        })

        self.logger.info(f"Returning response for message {message.id}")

        return SQLVizAssistantMessage(**response)

    def clear_thread(self, thread_id: str):
        """Clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        self.logger.info(f"Clearing memory for thread {thread_id}")
        self.router_agent.clear_thread(thread_id)

    async def aclear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        self.logger.info(f"Clearing memory for thread {thread_id}")
        await self.router_agent.aclear_thread(thread_id)
