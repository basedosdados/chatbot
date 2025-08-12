import operator
from typing import Annotated, AsyncIterator, Iterator, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field

from chatbot.agents.sql_agent import SQLAgent
from chatbot.agents.visualization_agent import VizAgent

from .prompts import (INITIAL_ROUTING_SYSTEM_PROMPT,
                      POST_SQL_ROUTING_SYSTEM_PROMPT)
from .reducers import Item
from .structured_outputs import InitialRouting, PostSQLRouting, Visualization
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages


class ChatTurn(BaseModel):
    # turn_id: int
    user_question: str
    ai_response: str
    data: list[Item] | None = Field(default=None)

class RouterAgentState(TypedDict):
    # node before routing
    _previous: str|None

    # next node to route to
    _next: str

    # input question
    question: str

    # flag indicating if the input question should be rewritten
    # for the current run when calling the SQLAgent
    rewrite_query: bool

    turn_counter: int

    # intermediate answers and final answer
    sql_answer: str
    final_answer: str

    # sql queries that were executed without errors and its results
    sql_queries: list[Item]
    sql_queries_results: list[Item]

    question_for_viz_agent: str | None
    data_turn_ids: list[int] | None

    # visualization object for plotting charts, if any
    visualization: Visualization | None

    # router agent's message list
    messages: Annotated[list[BaseMessage], add_messages]

    history: Annotated[list[ChatTurn], operator.add]

class RouterAgent:
    """LLM-powered Agent that orchestrates SQL querying and data visualization via a multi-agent workflow.

    Args:
        model (BaseChatModel):
            A LangChain chat model with structured output support. Used to:
                1. Route incoming user queries to `SQLAgent` or `VizAgent`.
                2. Assess whether the `SQLAgent` response would benefit from a graphical visuzalization.
        sql_agent (SQLAgent):
            An instance of `SQLAgent`, for interacting with the target SQL database.
        viz_agent (VizAgent):
            An instance of `VizAgent`, for generating visualization recommendation.
        checkpointer (PostgresSaver | AsyncPostgresSaver | bool | None, optional):
            PostgresSaver/AsyncPostgresSaver instance to persist per-thread state across
            runs. If the agent is used a subgraph, pass `True` instead. If set to `None`,
            no state is persisted. Defaults to `None`.
        question_limit (int | None, optional):
            Maximum number of Q&A turns to retain in the conversation history
            sent to the LLM. If `None`, the context is unlimited. Defaults to `5`.
    """

    def __init__(
        self,
        model: BaseChatModel,
        sql_agent: SQLAgent,
        viz_agent: VizAgent,
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        question_limit: int | None = 5,
    ):
        self.model = model
        self.sql_agent = sql_agent
        self.viz_agent = viz_agent
        self.checkpointer = checkpointer
        self.question_limit = question_limit
        self.graph = self._compile()

    def _call_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, Literal["sql_agent", "viz_agent"] | None]
    ):
        """Defines whether the `SQLAgent` or `VizAgent` should be called.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Literal["sql_agent", "viz_agent"] | None: The graph state update.
        """
        final_content = format_chat_history(
            current_question=state["question"],
            chat_history=state["history"],
        )

        messages = [
            SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=final_content)
        ]

        router = self.model.with_structured_output(InitialRouting)

        response: InitialRouting = router.invoke(messages, config)

        return {
            "_previous": None,
            "_next": response.agent,
            "question_for_viz_agent": response.question_for_viz_agent,
            "data_turn_ids": response.data_turn_ids,
        }

    async def _acall_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, Literal["sql_agent", "viz_agent"] | None]
    ):
        """Asynchronously defines whether the `SQLAgent` or `VizAgent` should be called.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Literal["sql_agent", "viz_agent"] | None: The graph state update.
        """
        final_content = format_chat_history(
            current_question=state["question"],
            chat_history=state["history"],
        )

        messages = [
            SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=final_content)
        ]

        router = self.model.with_structured_output(InitialRouting)

        response: InitialRouting = await router.ainvoke(messages, config)

        return {
            "_previous": None,
            "_next": response.agent,
            "question_for_viz_agent": response.question_for_viz_agent,
            "data_turn_ids": response.data_turn_ids,
        }

    def _call_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, Literal["process_answers", "viz_agent"]]
    ):
        """Defines if the `VizAgent` should be called before returning a response.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Literal["process_answers", "viz_agent"]]: The graph state update.
        """
        router = self.model.with_structured_output(PostSQLRouting)

        query_results = [qr.content for qr in state['sql_queries_results']]
        if len(query_results) == 1:
            query_results = query_results[0]

        messages = [
            SystemMessage(POST_SQL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(
                f"User question: {state['question']}\n\n"\
                f"Query results: {query_results}\n\n"\
                f"Answer: {state['sql_answer']}"
            )
        ]

        response: PostSQLRouting = router.invoke(messages, config)

        return {
            "_previous": "sql_agent",
            "_next": response.next
        }

    async def _acall_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, Literal["process_answers", "viz_agent"]]
    ):
        """Asynchronously defines if the `VizAgent` should be called before returning a response.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Literal["process_answers", "viz_agent"]]: The graph state update.
        """
        router = self.model.with_structured_output(PostSQLRouting)

        query_results = [qr.content for qr in state['sql_queries_results']]
        if len(query_results) == 1:
            query_results = query_results[0]

        messages = [
            SystemMessage(POST_SQL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(
                f"User question: {state['question']}\n\n"\
                f"Query results: {query_results}\n\n"\
                f"Answer: {state['sql_answer']}"
            )
        ]

        response: PostSQLRouting = await router.ainvoke(messages, config)

        return {
            "_previous": "sql_agent",
            "_next": response.next
        }

    def _call_sql_agent(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, str|list[Item]|list[AIMessage]]
    ):
        """Calls the `SQLAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|list[Item]|list[AIMessage]]: The graph state update.
        """
        response = self.sql_agent.invoke(
            question=state["question"],
            rewrite_query=state["rewrite_query"],
            config=config,
        )

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
        }

    async def _acall_sql_agent(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, str|list[Item]|list[AIMessage]]
    ):
        """Asynchronously calls the `SQLAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|list[Item]|list[AIMessage]]: The graph state update.
        """
        response = await self.sql_agent.ainvoke(
            question=state["question"],
            rewrite_query=state["rewrite_query"],
            config=config,
        )

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
        }

    def _call_viz_agent(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, Visualization]:
        """Calls the `VizAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Visualization]: The graph state update.
        """
        if state["_previous"] == "sql_agent":
            question_for_viz = state["question"]
            sql_queries_results = state["sql_queries_results"]
        else:
            question_for_viz = state["question_for_viz_agent"]
            # Turn IDs are 1-based, list indices are 0-based
            chat_turns = [
                state["history"][turn_id - 1]
                for turn_id in state["data_turn_ids"]
            ]
            sql_queries_results = [item for turn in chat_turns for item in turn.data]

        try:
            response = self.viz_agent.invoke(
                question=question_for_viz,
                sql_queries_results=sql_queries_results,
                config=config
            )
            visualization = response["visualization"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            visualization = None

        return {"visualization": visualization}

    async def _acall_viz_agent(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, Visualization]:
        """Asynchronously calls the `VizAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, Visualization]: The graph state update.
        """
        if state["_previous"] == "sql_agent":
            question_for_viz = state["question"]
            sql_queries_results = state["sql_queries_results"]
        else:
            question_for_viz = state["question_for_viz_agent"]
            # Turn IDs are 1-based, list indices are 0-based
            chat_turns = [
                state["history"][turn_id - 1]
                for turn_id in state["data_turn_ids"]
            ]
            sql_queries_results = [item for turn in chat_turns for item in turn.data]

        try:
            response = await self.viz_agent.ainvoke(
                question=question_for_viz,
                sql_queries_results=sql_queries_results,
                config=config
            )
            visualization = response["visualization"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            visualization = None

        return {"visualization": visualization}

    def _process_answers(self, state: RouterAgentState) -> dict[str, str]:
        """Builds the final answer that will be presented to the user.

        Args:
            state (RouterAgentState): The graph state.

        Returns:
            dict[str, str]: The final answer state update.
        """
        state_update = {}

        previous = state["_previous"]
        next = state["_next"]

        if next == "viz_agent":
            chart = state["visualization"]
            # sql_agent → viz_agent
            if previous == "sql_agent":
                sql_answer = state["sql_answer"]
                final_answer = f"{sql_answer}\n\n{chart.insights}"
                data = state["sql_queries_results"]
            # viz_agent called directly
            else:
                final_answer = chart.insights
                chat_turns = [
                    state["history"][turn_id - 1]
                    for turn_id in state["data_turn_ids"]
                ]
                data = [item for turn in chat_turns for item in turn.data]
        # sql_agent → process_answers
        else:
            state_update["visualization"] = None
            final_answer = state["sql_answer"]
            data = state["sql_queries_results"]

        chat_turn = ChatTurn(
            user_question=state["question"],
            ai_response=final_answer,
            data=data,
        )

        state_update["final_answer"] = final_answer
        state_update["history"] = [chat_turn]

        return state_update

    def _prune_messages(self, state: RouterAgentState) -> dict[str, list[RemoveMessage]]:
        """Prunes the message list to ensure that only a limited number of questions and their
        corresponding AI messages and Tool messages are sent to the LLM.

        Args:
            state (RouterAgentState): The graph state containing the message list.

        Returns:
            dict[str, list[RemoveMessage]]: The pruned message list.
        """
        history = state["history"]

        if self.question_limit and len(history) == self.question_limit:
            return {"history": history[-self.question_limit-1:]}

        return {"history": []}

    def _compile(self) -> CompiledGraph:
        """Compiles the state graph into a LangChain Runnable.

        Returns:
            CompiledGraph: The compiled state graph.
        """
        graph = StateGraph(RouterAgentState)

        # initial router node, calls the sql_agent or the viz_agent
        graph.add_node("initial_router", RunnableLambda(self._call_initial_router, self._acall_initial_router))

        # nodes for calling the SQL and Visualization agents. These are subgraphs.
        # For more information on subgraphs, refer to https://langchain-ai.github.io/langgraph/how-tos/subgraph
        graph.add_node("sql_agent", RunnableLambda(self._call_sql_agent, self._acall_sql_agent))
        graph.add_node("viz_agent", RunnableLambda(self._call_viz_agent, self._acall_viz_agent))

        # post sql router node, calls the viz_agent or process_answers
        graph.add_node("post_sql_router", RunnableLambda(self._call_post_sql_router, self._acall_post_sql_router))

        # node for building the final answer
        graph.add_node("process_answers", self._process_answers)

        # node for deleting old messages
        graph.add_node("prune_messages", self._prune_messages)

        graph.add_edge("sql_agent", "post_sql_router")
        graph.add_edge("viz_agent", "process_answers")
        graph.add_edge("process_answers", "prune_messages")
        graph.add_conditional_edges("initial_router", _route)
        graph.add_conditional_edges("post_sql_router", _post_sql_route)

        graph.set_entry_point("initial_router")
        graph.set_finish_point("prune_messages")

        # The checkpointer is ignored by default when the graph is used as a subgraph
        # For more information, visit https://langchain-ai.github.io/langgraph/how-tos/subgraph-persistence
        # If you want to persist the subgraph state between runs, you must use checkpointer=True
        # For more information, visit https://github.com/langchain-ai/langgraph/issues/3020
        return graph.compile(self.checkpointer)

    def invoke(self, question: str, config: RunnableConfig | None = None, rewrite_query: bool = False) -> RouterAgentState:
        """Runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The input question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            rewrite_query (bool | None, optional): Whether to rewrite the question for semantic search
                                                   when calling the `SQLAgent`. Defaults to `False`.

        Returns:
            RouterAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = self.graph.invoke(
            input={
                "question": question,
                "rewrite_query": rewrite_query,
            },
            config=config,
        )

        return response

    async def ainvoke(self, question: str, config: RunnableConfig | None = None, rewrite_query: bool = False) -> RouterAgentState:
        """Asynchronously runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The input question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            rewrite_query (bool | None, optional): Whether to rewrite the question for semantic search
                                                   when calling the `SQLAgent`. Defaults to `False`.

        Returns:
            RouterAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "rewrite_query": rewrite_query,
            },
            config=config,
        )

        return response

    def stream(
        self,
        question: str,
        config: RunnableConfig | None = None,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        rewrite_query: bool = False
    ) -> Iterator[dict|tuple]:
        """Stream graph steps.

        Args:
            question (str): The input question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.
            subgraphs (bool, optional): Whether to stream events from inside subgraphs. Defaults to `False`.
            rewrite_query (bool, optional): Whether to rewrite the question for semantic search. Defaults to `False`.

        Yields:
            dict|tuple: The output for each step in the graph.
                Its Its type, shape and content depends on the `stream_mode` and `subgraphs` args.
        """
        question = question.strip()

        for chunk in self.graph.stream(
            input={
                "question": question,
                "rewrite_query": rewrite_query,
            },
            config=config,
            stream_mode=stream_mode,
            subgraphs=subgraphs,
        ):
            yield chunk

    async def astream(
        self,
        question: str,
        config: RunnableConfig | None = None,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        rewrite_query: bool = False
    ) -> AsyncIterator[dict|tuple]:
        """Asynchronously stream graph steps.

        Args:
            question (str): The input question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.
            subgraphs (bool, optional): Whether to stream events from inside subgraphs. Defaults to `False`.
            rewrite_query (bool, optional): Whether to rewrite the question for semantic search. Defaults to `False`.

        Yields:
            dict|tuple: The output for each step in the graph.
                Its Its type, shape and content depends on the `stream_mode` and `subgraphs` args.
        """
        question = question.strip()

        async for chunk in self.graph.astream(
            input={
                "question": question,
                "rewrite_query": rewrite_query,
            },
            config=config,
            stream_mode=stream_mode,
            subgraphs=subgraphs,
        ):
            yield chunk

    # Unfortunately, there is no clean way to delete an agent's memory
    # except by deleting its checkpoints, as noted in this github discussion:
    # https://github.com/langchain-ai/langgraph/discussions/912
    def clear_thread(self, thread_id: str):
        """Deletes all checkpoints for a given thread.

        Args:
            thread_id (str): The thread unique identifier.
        """
        if self.checkpointer is None:
            logger.info("Checkpointer is None, ignoring...")
        else:
            delete_checkpoints(self.checkpointer, thread_id)
            logger.info(f"Deleted checkpoints for thread {thread_id}")

    async def aclear_thread(self, thread_id: str):
        """Asynchronously deletes all checkpoints for a given thread.

        Args:
            thread_id (str): The thread unique identifier.
        """
        if self.checkpointer is None:
            logger.info("Checkpointer is None, ignoring...")
        else:
            await async_delete_checkpoints(self.checkpointer, thread_id)
            logger.info(f"Deleted checkpoints for thread {thread_id}")

def _route(state: RouterAgentState) -> Literal["sql_agent", "viz_agent"]:
    "Routes to the next node."
    return state["_next"]

def _post_sql_route(state: RouterAgentState) -> Literal["viz_agent", "process_answers"]:
    "Routes to the next node."
    return state["_next"]

def format_chat_history(current_question: str, chat_history: list[ChatTurn]):
    history_content = []

    for i, chat_turn in enumerate(chat_history):
        history_content.append(
            f"### Turn {i+1}:\n"
            f"**User:** {chat_turn.user_question}\n"
            f"**AI:** {chat_turn.ai_response}"
        )

    if history_content:
        final_content = (
            "# Conversation History\n\n"
            + "\n---\n".join(history_content)
            + "\n\n# Current Turn:\n"
            + f"**User:** {current_question}"
        )
    else:
        final_content = (
            "# Conversation History\n"
            + "(empty)"
            + "\n\n# Current Turn:\n"
            + f"**User:** {current_question}"
        )

    return final_content
