from typing import Annotated, Any, Literal, TypeAlias, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages

from chatbot.agents.sql_agent import SQLAgent
from chatbot.agents.visualization_agent import VizAgent
from chatbot.loguru_logging import get_logger

from .prompts import ROUTER_SYSTEM_PROMPT
from .reducers import Item
from .structured_outputs import Chart, Route
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages

RouterAgentOutput: TypeAlias = dict[str, Literal["sql_agent", "viz_agent"]]
SQLAgentOutput: TypeAlias = dict[str, str|list[Item]|list[AIMessage]]
VizAgentOutput: TypeAlias = dict[str, str|Chart]


class State(TypedDict):
    # next node to route to
    next: str

    # input question
    question: str

    # intermediate answers and final answer
    sql_answer: str
    chart_answer: str
    final_answer: str

    # sql queries that were executed without errors and its results
    sql_queries: list[Item]
    sql_queries_results: list[Item]

    # chart object for plotting
    chart: Chart

    # router agent's message list
    messages: Annotated[list[BaseMessage], add_messages]

class RouterAgent:
    def __init__(
        self,
        model: BaseChatModel,
        sql_agent: SQLAgent,
        viz_agent: VizAgent,
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        question_limit: int | None = 5,
    ):
        router_system_message = SystemMessage(ROUTER_SYSTEM_PROMPT)

        self.router = (
            lambda messages: [router_system_message] + messages
        ) | model.with_structured_output(Route)

        self.sql_agent = sql_agent

        self.viz_agent = viz_agent

        self.checkpointer = checkpointer

        self.question_limit = question_limit

        self.logger = get_logger(self.__class__.__name__)

        self.graph = self._compile()

    def _call_router(self, state: State, config: RunnableConfig) -> RouterAgentOutput:
        """Calls the router agent

        Returns:
            RouterAgentOutput: The next node to route to
        """
        response: Route = self.router.invoke(state["messages"], config)
        return {"next": response.next}

    async def _acall_router(self, state: State, config: RunnableConfig) -> RouterAgentOutput:
        """Asynchronously calls the router agent

        Returns:
            RouterAgentOutput: The next node to route to
        """
        response: Route = await self.router.ainvoke(state["messages"], config)
        return {"next": response.next}

    def _call_sql_agent(self, state: State, config: RunnableConfig) -> SQLAgentOutput:
        """Calls the `SQLAgent`

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the agent

        Returns:
            SQLAgentOutput: The graph state update
        """
        response = self.sql_agent.invoke(state["question"], config)

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
            "messages": [AIMessage(response["final_answer"])]
        }

    async def _acall_sql_agent(self, state: State, config: RunnableConfig) -> SQLAgentOutput:
        """Asynchronously calls the `SQLAgent`

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the agent

        Returns:
            SQLAgentOutput: The graph state update
        """
        response = await self.sql_agent.ainvoke(state["question"], config)

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
            "messages": [AIMessage(response["final_answer"])]
        }

    def _call_viz_agent(self, state: State, config: RunnableConfig) -> VizAgentOutput:
        """Calls the `VizAgent`

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the agent

        Returns:
            VizAgentOutput: The graph state update
        """
        response = self.viz_agent.invoke(
            question=state["question"],
            sql_answer=state["sql_answer"],
            sql_queries=state["sql_queries"],
            sql_queries_results=state["sql_queries_results"],
            config=config
        )

        return {
            "chart": response["chart"],
            "chart_answer": response["chart_answer"],
        }

    async def _acall_viz_agent(self, state: State, config: RunnableConfig) -> VizAgentOutput:
        """Asynchronously calls the `VizAgent`

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the agent

        Returns:
            VizAgentOutput: The graph state update
        """
        response = await self.viz_agent.ainvoke(
            question=state["question"],
            sql_answer=state["sql_answer"],
            sql_queries=state["sql_queries"],
            sql_queries_results=state["sql_queries_results"],
            config=config
        )

        return {
            "chart": response["chart"],
            "chart_answer": response["chart_answer"],
        }

    def _process_answers(self, state: State) -> dict[str, str]:
        """Builds the final answer that will be presented to the user

        Args:
            state (State): The graph state

        Returns:
            dict[str, str]: The final answer
        """
        next = state["next"]
        sql_answer = state["sql_answer"]
        chart_answer = state["chart_answer"]
        chart = state["chart"]

        if next == "sql_agent" and chart.is_valid:
            answer = f"{sql_answer}\n\n{chart_answer}"
        elif next == "sql_agent":
            answer = sql_answer
        elif next == "viz_agent":
            answer = chart_answer

        return {"final_answer": answer}

    def _prune_messages(self, state: State) -> dict[str, list[RemoveMessage]]:
        """Prunes the message list to ensure that only a limited number of questions and their
        corresponding AI messages and Tool messages are sent to the LLM.

        Args:
            state (State): The graph state containing the message list

        Returns:
            dict[str, list[RemoveMessage]]: The pruned message list
        """
        if self.question_limit is None:
            return {"messages": []}

        return {
            "messages": prune_messages(
                messages=state["messages"],
                question_limit=self.question_limit
            )
        }

    def _compile(self) -> CompiledGraph:
        """Compiles the state graph into a LangChain Runnable

        Returns:
            CompiledGraph: The compiled state graph
        """
        graph = StateGraph(State)

        # router node, calls the sql_agent or the viz_agent
        graph.add_node("router", RunnableLambda(self._call_router, self._acall_router))

        # nodes for calling the SQL and Visualization agents. These are subgraphs.
        # For more information on subgraphs, refer to https://langchain-ai.github.io/langgraph/how-tos/subgraph
        graph.add_node("sql_agent", RunnableLambda(self._call_sql_agent, self._acall_sql_agent))
        graph.add_node("viz_agent", RunnableLambda(self._call_viz_agent, self._acall_viz_agent))

        # node for building the final answer
        graph.add_node("process_answers", self._process_answers)

        # node for deleting old messages
        graph.add_node("prune_messages", self._prune_messages)

        graph.add_edge("sql_agent", "viz_agent")
        graph.add_edge("viz_agent", "process_answers")
        graph.add_edge("process_answers", "prune_messages")
        graph.add_conditional_edges("router", _route)

        graph.set_entry_point("router")
        graph.set_finish_point("prune_messages")

        # The checkpointer is ignored by default when the graph is used as a subgraph
        # For more information, visit https://langchain-ai.github.io/langgraph/how-tos/subgraph-persistence
        # If you want to persist the subgraph state between runs, you must use checkpointer=True
        # For more information, visit https://github.com/langchain-ai/langgraph/issues/3020
        return graph.compile(self.checkpointer)

    def invoke(self, question: str, config: RunnableConfig | None = None) -> dict[str, Any] | Any:
        """Runs the compiled graph with a question and an optional configuration

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): The configuration. Defaults to None.

        Returns:
            dict[str, Any] | Any: The last output of the graph run
        """
        question = question.strip()

        message = HumanMessage(content=question)

        response = self.graph.invoke(
            input={
                "question": question,
                "messages": [message],
            },
            config=config,
        )

        return response

    async def ainvoke(self, question: str, config: RunnableConfig | None = None) -> dict[str, Any] | Any:
        """Asynchronously runs the compiled graph with a question and an optional configuration

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): The configuration. Defaults to None.

        Returns:
            dict[str, Any] | Any: The last output of the graph run
        """
        question = question.strip()

        message = HumanMessage(content=question)

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "messages": [message],
            },
            config=config,
        )

        return response

    # Unfortunately, there is no clean way to delete an agent's memory
    # except by deleting its checkpoints, as noted in this github discussion:
    # https://github.com/langchain-ai/langgraph/discussions/912
    def clear_thread(self, thread_id: str):
        """Clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        try:
            if self.checkpointer is None:
                self.logger.info("Checkpointer is None, ignoring...")
            else:
                delete_checkpoints(self.checkpointer, thread_id)
                self.logger.info(f"Deleted checkpoints for thread {thread_id}")
        except Exception:
            self.logger.exception(f"Error on clearing thread {thread_id}:")

    async def aclear_thread(self, thread_id: str):
        """Asynchronously clears a thread

        Args:
            thread_id (str): The thread unique identifier
        """
        try:
            if self.checkpointer is None:
                self.logger.info("Checkpointer is None, ignoring...")
            else:
                await async_delete_checkpoints(self.checkpointer, thread_id)
                self.logger.info(f"Deleted checkpoints for thread {thread_id}")
        except Exception:
            self.logger.exception(f"Error on clearing thread {thread_id}:")

def _route(state: State) -> Literal["sql_agent", "viz_agent"]:
    "Routes to the next node"
    return state["next"]
