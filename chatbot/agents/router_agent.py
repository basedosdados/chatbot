from typing import Annotated, Literal, TypeAlias, TypedDict

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

from chatbot.agents.sql_agent import SQLAgent
from chatbot.agents.visualization_agent import VizAgent

from .prompts import (INITIAL_ROUTING_SYSTEM_PROMPT,
                      POST_SQL_ROUTING_SYSTEM_PROMPT)
from .reducers import Item
from .structured_outputs import (Chart, ChartData, ChartMetadata,
                                 InitialRouting, PostSQLRouting)
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages

RouterAgentOutput: TypeAlias = dict[str, str|None]
SQLAgentOutput: TypeAlias = dict[str, str|list[Item]|list[AIMessage]]
VizAgentOutput: TypeAlias = dict[str, str|Chart]


class RouterAgentState(TypedDict):
    # node before routing
    _previous: str|None

    # next node to route to
    _next: str

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
        self.model = model
        self.sql_agent = sql_agent
        self.viz_agent = viz_agent
        self.checkpointer = checkpointer
        self.question_limit = question_limit
        self.graph = self._compile()

    def _call_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> RouterAgentOutput:
        """Calls the router agent.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            RouterAgentOutput: The graph state update.
        """
        router = (
            lambda messages: [SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT)] + messages
        ) | self.model.with_structured_output(InitialRouting)

        response: InitialRouting = router.invoke(state["messages"], config)

        return {
            "_previous": None,
            "_next": response.next
        }

    async def _acall_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> RouterAgentOutput:
        """Asynchronously calls the router agent.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            RouterAgentOutput: The graph state update.
        """
        router = (
            lambda messages: [SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT)] + messages
        ) | self.model.with_structured_output(InitialRouting)

        response: InitialRouting = await router.ainvoke(state["messages"], config)

        return {
            "_previous": None,
            "_next": response.next
        }

    def _call_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> RouterAgentOutput:
        """Calls the router agent after the `sql_agent` node.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            RouterAgentOutput: The graph state update.
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

    async def _acall_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> RouterAgentOutput:
        """Asynchronously calls the router agent after the `sql_agent` node.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            RouterAgentOutput: The graph state update.
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

    def _call_sql_agent(self, state: RouterAgentState, config: RunnableConfig) -> SQLAgentOutput:
        """Calls the `SQLAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            SQLAgentOutput: The graph state update.
        """
        response = self.sql_agent.invoke(state["question"], config)

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
            "messages": [AIMessage(response["final_answer"])]
        }

    async def _acall_sql_agent(self, state: RouterAgentState, config: RunnableConfig) -> SQLAgentOutput:
        """Asynchronously calls the `SQLAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            SQLAgentOutput: The graph state update.
        """
        response = await self.sql_agent.ainvoke(state["question"], config)

        return {
            "sql_answer": response["final_answer"],
            "sql_queries": response["sql_queries"],
            "sql_queries_results": response["sql_queries_results"],
            "messages": [AIMessage(response["final_answer"])]
        }

    def _call_viz_agent(self, state: RouterAgentState, config: RunnableConfig) -> VizAgentOutput:
        """Calls the `VizAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            VizAgentOutput: The graph state update.
        """
        try:
            response = self.viz_agent.invoke(
                question=state["question"],
                sql_answer=state["sql_answer"],
                sql_queries=state["sql_queries"],
                sql_queries_results=state["sql_queries_results"],
                config=config
            )

            chart = response["chart"]
            chart_answer = response["chart_answer"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            chart = Chart(
                data=ChartData(),
                metadata=ChartMetadata(),
                is_valid=False
            )
            chart_answer = f"Ops, algo deu errado na construção do gráfico! Por favor, tente novamente. "\
                    "Se o problema persistir, avise-nos. Obrigado pela paciência!",

        return {
            "chart": chart,
            "chart_answer": chart_answer
        }

    async def _acall_viz_agent(self, state: RouterAgentState, config: RunnableConfig) -> VizAgentOutput:
        """Asynchronously calls the `VizAgent`.

        Args:
            state (RouterAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            VizAgentOutput: The graph state update.
        """
        try:
            response = await self.viz_agent.ainvoke(
                question=state["question"],
                sql_answer=state["sql_answer"],
                sql_queries=state["sql_queries"],
                sql_queries_results=state["sql_queries_results"],
                config=config
            )

            chart = response["chart"]
            chart_answer = response["chart_answer"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            chart = Chart(
                data=ChartData(),
                metadata=ChartMetadata(),
                is_valid=False
            )
            chart_answer = f"Ops, algo deu errado na construção do gráfico! Por favor, tente novamente. "\
                    "Se o problema persistir, avise-nos. Obrigado pela paciência!",

        return {
            "chart": chart,
            "chart_answer": chart_answer,
        }

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
            chart = state["chart"]
            chart_answer = state["chart_answer"]

            # sql_agent → viz_agent and the chart is valid
            if previous == "sql_agent" and chart.is_valid:
                sql_answer = state["sql_answer"]
                final_answer = f"{sql_answer}\n\n{chart_answer}"
            # sql_agent → viz_agent and the chart is not valid
            elif previous == "sql_agent":
                final_answer = state["sql_answer"]
            # viz_agent called directly
            else:
                final_answer = chart_answer

        # sql_agent → process_answers
        else:
            chart = Chart(
                data=ChartData(),
                metadata=ChartMetadata(),
                is_valid=False
            )
            state_update["chart"] = chart
            final_answer = state["sql_answer"]

        state_update["final_answer"] = final_answer

        return state_update

    def _prune_messages(self, state: RouterAgentState) -> dict[str, list[RemoveMessage]]:
        """Prunes the message list to ensure that only a limited number of questions and their
        corresponding AI messages and Tool messages are sent to the LLM.

        Args:
            state (RouterAgentState): The graph state containing the message list.

        Returns:
            dict[str, list[RemoveMessage]]: The pruned message list.
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

    def invoke(self, question: str, config: RunnableConfig | None = None) -> RouterAgentState:
        """Runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            RouterAgentState: The output of the agent execution.
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

    async def ainvoke(self, question: str, config: RunnableConfig | None = None) -> RouterAgentState:
        """Asynchronously runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            RouterAgentState: The output of the agent execution.
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

def _route(state: RouterAgentState) -> Literal["sql_agent", "viz_agent", "process_answers"]:
    "Routes to the next node."
    return state["_next"]

def _post_sql_route(state: RouterAgentState) -> Literal["viz_agent", "process_answers"]:
    "Routes to the next node."
    return state["_next"]
