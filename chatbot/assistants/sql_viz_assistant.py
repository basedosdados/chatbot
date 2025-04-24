import codecs
import os
import uuid
from typing import Any, Self

import sqlparse
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from chatbot.agents import RouterAgent, SQLAgent, VizAgent
from chatbot.databases import Database
from chatbot.exceptions import EnvironmentVariableUnset, NotInitializedError
from chatbot.loguru_logging import get_logger
from chatbot.models import ModelFactory
from chatbot.storage import get_chroma_or_none

from .datatypes import ModelURI, SQLVizAssistantMessage, UserMessage


class SQLVizAssistant:
    """LLM-powered assistant for querying and visualizing databases.

    Args:
        database (Database):
            A `Database` object implementing the `Database` protocol.
        model_uri (str | ModelURI | None, optional):
            URI of the LLM to be used, in the format `<provider>/<model_name>`, e.g.,
            `openai/gpt-4o` or `google/gemini-1.5-flash-001`. If `None`, falls back
            to the `MODEL_URI` environment variable. Defaults to `None`.
        checkpointer (bool, optional):
            If `True`, uses an `AsyncPostgresSaver` checkpointer to persist state across assistant runs.
            If `False`, the assistant will not retain memory of previous messages. Defaults to `True`.
        checkpointer_db_url (str | None, optional):
            PostgreSQL database URL used to persist checkpoints when `checkpointer` is `True`.
            If `None` and `checkpointer` is `True`, falls back to the `DB_URL` environment variable.
            Defaults to `None`.
        chroma_host (str | None, optional):
            Hostname for the ChromaDB client. If `None`, falls back to the `CHROMA_HOST`
            environment variable. Defaults to `None`.
        chroma_port (str | int | None, optional):
            Port for the ChromaDB client. If `None`, falls back to the `CHROMA_PORT`
            environment variable. Defaults to `None`.
        sql_chroma_collection (str | None, optional):
            Name of the ChromaDB collection that contains examples for the `SQLAgent` LLM calls.
            If set to `None`, will fallback to the `SQL_CHROMA_COLLECTION` env variable. Defaults to None.
        viz_chroma_collection (str | None, optional):
            Name of the ChromaDB collection that contains examples for the `VizAgent` LLM calls.
            If set to `None`, will fallback to the `VIZ_CHROMA_COLLECTION` env variable. Defaults to None.
        question_limit (int | None, optional):
            Maximum number of previous questions to keep in memory. If `None`, all questions are kept.
            Defaults to `5`.

    Raises:
        EnvironmentVariableError: If `model_uri` is not provided and `MODEL_URI` is not set.
        EnvironmentVariableError: If `checkpointer_db_url` is not provided and `DB_URL` is not set.
        TypeError: If `model_uri` is neither a string nor a `ModelURI`.
    """

    def __init__(
        self,
        database: Database,
        model_uri: str | ModelURI | None = None,
        checkpointer: bool = True,
        checkpointer_db_url: str | None = None,
        chroma_host: str | None = None,
        chroma_port: str | int | None = None,
        sql_chroma_collection: str | None = None,
        viz_chroma_collection: str | None = None,
        question_limit: int | None = 5,
    ):
        model_uri = model_uri or os.getenv("MODEL_URI")

        if model_uri is None:
            raise EnvironmentVariableUnset(
                "The model URI was not passed and could not be inferred from the environment. "
                "Please pass a valid model URI to the `model_uri` argument or "
                "set the `MODEL_URI` environment variable"
            )
        elif isinstance(model_uri, str):
            model_uri = ModelURI(model_uri)
        elif not isinstance(model_uri, ModelURI):
            raise TypeError(
                f"`model_uri` must be of type `str` or `ModelURI`, got `{type(model_uri)}`"
            )

        self.model_uri = model_uri
        model = ModelFactory.from_model_uri(self.model_uri)

        if checkpointer:
            checkpointer_db_url = checkpointer_db_url or os.getenv("DB_URL")

            if checkpointer_db_url is None:
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

            self._pool = ConnectionPool(
                conninfo=checkpointer_db_url,
                kwargs=conn_kwargs,
                max_size=8,
                open=False,
            )
            self._checkpointer = PostgresSaver(self._pool)
            subgraph_checkpointer = True
        else:
            self._pool = None
            self._checkpointer = None
            subgraph_checkpointer = None

        chroma_host = chroma_host or os.getenv("CHROMA_HOST")
        chroma_port = chroma_port or os.getenv("CHROMA_PORT")
        sql_chroma_collection = sql_chroma_collection or os.getenv("SQL_CHROMA_COLLECTION")
        viz_chroma_collection = viz_chroma_collection or os.getenv("VIZ_CHROMA_COLLECTION")

        sql_vector_store = get_chroma_or_none(
            host=chroma_host,
            port=chroma_port,
            collection=sql_chroma_collection
        )

        viz_vector_store = get_chroma_or_none(
            host=chroma_host,
            port=chroma_port,
            collection=viz_chroma_collection
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
            checkpointer=self._checkpointer,
            question_limit=question_limit
        )

        self.logger = get_logger(self.__class__.__name__)

        self._is_setup = False

    def setup(self):
        """Opens the connection pool and setup the checkpoints tables"""
        if self._is_setup:
            return

        if self._checkpointer is not None:
            self._pool.open()
            self._checkpointer.setup()

        self._is_setup = True

    def shutdown(self):
        """Closes the connection pool"""
        if self._pool is not None:
            self._pool.close()

    def _ensure_setup(self):
        """Ensures the `setup()` method was called"""
        if not self._is_setup:
            raise NotInitializedError(
                "The `setup()` method must be called before using this method."
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
        self._ensure_setup()

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

    def clear_thread(self, thread_id: str):
        """Clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        self._ensure_setup()
        self.logger.info(f"Clearing memory for thread {thread_id}")
        self.router_agent.clear_thread(thread_id)

    def __enter__(self) -> Self:
        self.setup()
        return self

    def __exit__(self, exc_t, exc_v, exc_tb):
        self.shutdown()
