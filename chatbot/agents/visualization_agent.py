import json
from typing import Annotated, Any, AsyncIterator, Iterator, Literal, TypedDict

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

from .prompts import REPHRASER_VIZ_SYSTEM_PROMPT, VIZ_SYSTEM_PROMPT
from .reducers import Item
from .structured_outputs import Rephrase, VizScript, Visualization
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages


class VizAgentState(TypedDict):
    # input question
    question: str

    # rephrased question
    question_rephrased: str

    # sql queries results
    sql_queries_results: list[Item]

    # normalized sql queries results
    normalized_data: list[dict[str, Any]]

    # visualization agent's message list
    messages: Annotated[list[BaseMessage], add_messages]

    # visualization output
    visualization: Visualization | None

class VizAgent:
    """LLM-powered Visualization Agent for visualization recommendations.

    Args:
        model (BaseChatModel):
            A LangChain chat model with structured output support.
        prompt_formatter (VizPromptFormatter):
            A formatter responsible for constructing the LLM system prompt during data preprocessing
            step, based on the user's question and optional few-shot examples. Must implement how
            examples are retrieved and how the prompt template is composed.
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
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        question_limit: int | None = 5,
    ):
        self.checkpointer = checkpointer
        self.question_limit = question_limit

        rephraser_system_message = SystemMessage(REPHRASER_VIZ_SYSTEM_PROMPT)
        viz_system_message = SystemMessage(VIZ_SYSTEM_PROMPT)

        self.rephraser_runnable = (
            lambda question: [rephraser_system_message] + [question]
        ) | model.with_structured_output(Rephrase)

        self.viz_runnable = (
            lambda messages: [viz_system_message] + messages
        ) | model.with_structured_output(VizScript)

        self.graph = self._compile()

    def _call_rephrase_question(self, state: VizAgentState, config: RunnableConfig) -> dict[str, str|list[HumanMessage]]:
        """Calls the model for rephrasing the user's question.

        Args:
            state (VizAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|HumanMessage]: The state update containing the rephrased question.
        """
        question = state["question"]
        response: Rephrase = self.rephraser_runnable.invoke(question, config)
        return {"question_rephrased": response.rephrased}

    async def _acall_rephrase_question(self, state: VizAgentState, config: RunnableConfig) -> dict[str, str|list[HumanMessage]]:
        """Asynchronously calls the model for rephrasing the user's question.

        Args:
            state (VizAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|HumanMessage]: The state update containing the rephrased question.
        """
        question = state["question"]
        response: Rephrase = await self.rephraser_runnable.ainvoke(question, config)
        return {"question_rephrased": response.rephrased}

    def _get_queries_and_results(self, state: VizAgentState) -> dict[str, list[HumanMessage]]:
        """Builds the message list from the SQL queries and SQL queries restuls lists.

        Args:
            state (VizAgentState): The graph state.

        Returns:
            dict[str, list[HumanMessage]]: A dictionary containing the message list.
        """
        question = state["question_rephrased"]

        queries_results = []

        for qr in state["sql_queries_results"]:
            if qr.content:
                try:
                    for row in json.loads(qr.content):
                        queries_results.append(row)
                except json.JSONDecodeError:
                    logger.exception(f"JSON decode error, skipping {qr.content = }:")

        message = HumanMessage(
            f"Question: {question}\nData: {queries_results}"
        )

        return {"messages": [message], "normalized_data": queries_results}

    def _generate_visualization(self, state: VizAgentState, config: RunnableConfig) -> dict[str, AIMessage|VizScript]:
        messages = state["messages"]

        viz_script: VizScript = self.viz_runnable.invoke(messages, config)

        if viz_script.script:
            script_data = viz_script.model_dump()
            script_data["data"] = state["normalized_data"]
            visualization = Visualization(**script_data)
        else:
            visualization = None

        return {
            "messages": [AIMessage(viz_script.model_dump_json(indent=2))],
            "visualization": visualization
        }

    async def _agenerate_visualization(self, state: VizAgentState, config: RunnableConfig) -> dict[str, AIMessage|VizScript]:
        messages = state["messages"]

        viz_script: VizScript = await self.viz_runnable.ainvoke(messages, config)

        if viz_script.script:
            script_data = viz_script.model_dump()
            script_data["data"] = state["normalized_data"]
            visualization = Visualization(**script_data)
        else:
            visualization = None

        return {
            "messages": [AIMessage(viz_script.model_dump_json(indent=2))],
            "visualization": visualization
        }

    def _prune_messages(self, state: VizAgentState) -> dict[str, list[RemoveMessage]]:
        """Prunes the message list to ensure that only a limited number of questions and their
        corresponding AI messages and Tool messages are sent to the LLM.

        Args:
            state (VizAgentState): The graph state containing the message list.

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
        graph = StateGraph(VizAgentState)

        # user's question rephrasing node
        graph.add_node("rephrase", RunnableLambda(self._call_rephrase_question, self._acall_rephrase_question))

        # node for getting sql queries and its results
        graph.add_node("get_data", self._get_queries_and_results)

        # node for generating visualization script
        graph.add_node("create_visualization", RunnableLambda(self._generate_visualization, self._agenerate_visualization))

        # message pruning node
        graph.add_node("prune_messages", self._prune_messages)

        graph.add_edge("rephrase", "get_data")
        graph.add_edge("get_data", "create_visualization")
        graph.add_edge("create_visualization", "prune_messages")

        graph.set_entry_point("rephrase")
        graph.set_finish_point("prune_messages")

        # The checkpointer is ignored by default when the graph is used as a subgraph
        # For more information, visit https://langchain-ai.github.io/langgraph/how-tos/subgraph-persistence
        # If you want to persist the subgraph state between runs, you must use checkpointer=True
        # For more information, visit https://github.com/langchain-ai/langgraph/issues/3020
        return graph.compile(self.checkpointer)

    def invoke(
        self,
        question: str,
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None
    ) -> VizAgentState:
        """Runs the compiled graph.

        Args:
            question (str): The input question.
            sql_answer (str): The answer generated by the `SQLAgent`.
            sql_queries (list[Item]): The SQL queries generated by the `SQLAgent`.
            sql_queries_results (list[Item]): The SQL queries results.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            VizAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = self.graph.invoke(
            input={
                "question": question,
                "sql_queries_results": sql_queries_results
            },
            config=config,
        )

        return response

    async def ainvoke(
        self,
        question: str,
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None
    ) -> VizAgentState:
        """Asynchronously runs the compiled graph.

        Args:
            question (str): The input question.
            sql_answer (str): The answer generated by the `SQLAgent`.
            sql_queries (list[Item]): The SQL queries generated by the `SQLAgent`.
            sql_queries_results (list[Item]): The SQL queries results.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            VizAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "sql_queries_results": sql_queries_results
            },
            config=config,
        )

        return response

    def stream(
        self,
        question: str,
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None,
        stream_mode: list[str] | None = None,
    ) -> Iterator[dict|tuple]:
        """Stream graph steps.

        Args:
            question (str): The input question.
            sql_answer (str): The answer generated by the `SQLAgent`.
            sql_queries (list[Item]): The SQL queries generated by the `SQLAgent`.
            sql_queries_results (list[Item]): The SQL queries results.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.

        Yields:
            dict|tuple: The output for each step in the graph. Its type, shape and content depends on the `stream_mode` arg.
        """
        question = question.strip()

        for chunk in self.graph.stream(
            input={
                "question": question,
                "sql_queries_results": sql_queries_results
            },
            config=config,
            stream_mode=stream_mode,
        ):
            yield chunk

    async def astream(
        self,
        question: str,
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None,
        stream_mode: list[str] | None = None,
    ) -> AsyncIterator[dict|tuple]:
        """Asynchronously stream graph steps.

        Args:
            question (str): The input question.
            sql_answer (str): The answer generated by the `SQLAgent`.
            sql_queries (list[Item]): The SQL queries generated by the `SQLAgent`.
            sql_queries_results (list[Item]): The SQL queries results.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution. Defaults to `None`.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.

        Yields:
            dict|tuple: The output for each step in the graph. Its type, shape and content depends on the `stream_mode` arg.
        """
        question = question.strip()

        async for chunk in self.graph.astream(
            input={
                "question": question,
                "sql_queries_results": sql_queries_results
            },
            config=config,
            stream_mode=stream_mode,
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
