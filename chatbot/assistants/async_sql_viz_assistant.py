from langchain.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from chatbot.agents import RouterAgent, SQLAgent, VizAgent
from chatbot.databases import Database

from .formatting import format_router_agent_response
from .messages import SQLVizAssistantMessage


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

    async def invoke(self, message: str, config: dict|None=None) -> SQLVizAssistantMessage:
        """Asynchronously sends a user message to the `RouterAgent` and returns its response

        Args:
            message (str): The user message
            thread_id (str | None, optional): The thread unique identifier. Defaults to None.

        Returns:
            SQLVizAssistantMessage: The generated response
        """
        response = await self.router_agent.ainvoke(message, config)
        response = format_router_agent_response(response)

        if "run_id" in config:
            response["id"] = config["run_id"]

        return SQLVizAssistantMessage(**response)

    async def clear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        await self.router_agent.aclear_thread(thread_id)
