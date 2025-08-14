import json
from typing import Annotated, Any, AsyncIterator, Iterator, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from loguru import logger

from chatbot.agents.sql_agent import SQLAgent
from chatbot.agents.visualization_agent import VizAgent

from .prompts import (INITIAL_ROUTING_SYSTEM_PROMPT,
                      POST_SQL_ROUTING_SYSTEM_PROMPT)
from .reducers import ChatTurn, ChatTurnRemove, Item, add_chat_turn
from .structured_outputs import InitialRouting, PostSQLRouting, Visualization
from .utils import async_delete_checkpoints, delete_checkpoints


class RouterAgentState(TypedDict):
    # node before routing
    _previous: str|None

    # node after routing
    _next: str

    # input question
    question: str

    # flag indicating if the input question should be rewritten
    # for the current run when calling the SQLAgent
    rewrite_query: bool

    # SQLAgent output
    sql_answer: str

    # RouterAgent output
    final_answer: str

    # sql queries that were executed without errors and its results
    sql_queries: list[Item]
    sql_queries_results: list[Item]

    # list of turn IDs whose data should be passed to the VizAgent
    data_turn_ids: list[int] | None

    # rephrased question for VizAgent
    question_for_viz_agent: str | None

    # visualization object for plotting charts
    visualization: Visualization | None

    # RouterAgent conversation history
    chat_history: Annotated[dict[int, ChatTurn], add_chat_turn]

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

    def _call_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, str|None]:
        """Defines whether the `SQLAgent` or `VizAgent` should be called.

        Args:
            state (RouterAgentState): Current graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|None]: The graph state update.
        """
        input_message = _format_input_message(
            current_question=state["question"],
            chat_history=state["chat_history"],
        )

        messages = [
            SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=input_message)
        ]

        router = self.model.with_structured_output(InitialRouting)

        response: InitialRouting = router.invoke(messages, config)

        return {
            "_previous": None,
            "_next": response.agent,
            "question_for_viz_agent": response.question_for_viz_agent,
            "data_turn_ids": response.data_turn_ids,
        }

    async def _acall_initial_router(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, str|None]:
        """Asynchronously defines whether the `SQLAgent` or `VizAgent` should be called.

        Args:
            state (RouterAgentState): Current graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str|None]: The graph state update.
        """
        input_message = _format_input_message(
            current_question=state["question"],
            chat_history=state["chat_history"],
        )

        messages = [
            SystemMessage(INITIAL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=input_message)
        ]

        router = self.model.with_structured_output(InitialRouting)

        response: InitialRouting = await router.ainvoke(messages, config)

        return {
            "_previous": None,
            "_next": response.agent,
            "question_for_viz_agent": response.question_for_viz_agent,
            "data_turn_ids": response.data_turn_ids,
        }

    def _call_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, str]:
        """Defines if the `VizAgent` should be called before returning a response.

        Args:
            state (RouterAgentState): Current graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str]: The graph state update.
        """
        router = self.model.with_structured_output(PostSQLRouting)

        data = _normalize_data(state["sql_queries_results"])

        input_message = {
            "user_question": state["question"],
            "data": data,
            "text_answer": state["sql_answer"]
        }

        messages = [
            SystemMessage(POST_SQL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(json.dumps(input_message, ensure_ascii=False, indent=2))
        ]

        response: PostSQLRouting = router.invoke(messages, config)

        if response.action == "trigger_visualization":
            next_node = "viz_agent"
        else:
            next_node = "process_outputs"

        return {
            "_previous": "sql_agent",
            "_next": next_node,
            "question_for_viz_agent": response.question_for_viz_agent
        }

    async def _acall_post_sql_router(self, state: RouterAgentState, config: RunnableConfig) -> dict[str, str]:
        """Asynchronously defines if the `VizAgent` should be called before returning a response.

        Args:
            state (RouterAgentState): Current graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, str]: The graph state update.
        """
        router = self.model.with_structured_output(PostSQLRouting)

        data = _normalize_data(state["sql_queries_results"])

        input_message = {
            "user_question": state["question"],
            "data": data,
            "text_answer": state["sql_answer"]
        }

        messages = [
            SystemMessage(POST_SQL_ROUTING_SYSTEM_PROMPT),
            HumanMessage(json.dumps(input_message, ensure_ascii=False, indent=2))
        ]

        response: PostSQLRouting = await router.ainvoke(messages, config)

        if response.action == "trigger_visualization":
            next_node = "viz_agent"
        else:
            next_node = "process_outputs"

        return {
            "_previous": "sql_agent",
            "_next": next_node,
            "question_for_viz_agent": response.question_for_viz_agent
        }

    def _call_sql_agent(self, state: RouterAgentState, config: RunnableConfig) -> (
        dict[str, str|list[Item]|list[AIMessage]]
    ):
        """Calls the `SQLAgent`.

        Args:
            state (RouterAgentState): Current graph state.
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
            state (RouterAgentState): Current graph state.
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

    def _call_viz_agent(self, state: RouterAgentState) -> dict[str, Visualization]:
        """Calls the `VizAgent`.

        Args:
            state (RouterAgentState): Current graph state.

        Returns:
            dict[str, Visualization]: The graph state update.
        """
        question_for_viz = state["question_for_viz_agent"] or state["question"]

        # if SQLAgent was called in this turn, get the data fetched by it
        if state["_previous"] == "sql_agent":
            normalized_data = _normalize_data(state["sql_queries_results"])
        # else, get the data fetched in previous turns
        else:
            data = _get_data_from_chat_turns(
                turn_ids=state["data_turn_ids"],
                chat_history=state["chat_history"]
            )
            normalized_data = _normalize_data(data)

        try:
            response = self.viz_agent.invoke(
                question=question_for_viz,
                data=normalized_data,
            )
            visualization = response["visualization"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            visualization = None

        return {"visualization": visualization}

    async def _acall_viz_agent(self, state: RouterAgentState) -> dict[str, Visualization]:
        """Asynchronously calls the `VizAgent`.

        Args:
            state (RouterAgentState): Current graph state.

        Returns:
            dict[str, Visualization]: The graph state update.
        """
        question_for_viz = state["question_for_viz_agent"] or state["question"]

        # if SQLAgent was called in this turn, get the data fetched by it
        if state["_previous"] == "sql_agent":
            normalized_data = _normalize_data(state["sql_queries_results"])
        # else, get the data fetched in previous turns
        else:
            data = _get_data_from_chat_turns(
                turn_ids=state["data_turn_ids"],
                chat_history=state["chat_history"]
            )
            normalized_data = _normalize_data(data)

        try:
            response = await self.viz_agent.ainvoke(
                question=question_for_viz,
                data=normalized_data,
            )
            visualization = response["visualization"]
        except Exception:
            logger.exception(f"Error on calling the visualization agent:")
            visualization = None

        return {"visualization": visualization}

    def _process_outputs(self, state: RouterAgentState) -> dict[str, Any]:
        """Builds the final answer for the user and updates the conversation history.

        Args:
            state (RouterAgentState): Current graph state.

        Returns:
            dict[str, Any]: A dictionary containing the final answer and the updated history.
                If the visualization step was skipped, it also contains a `None` visualization.
        """
        state_update = {}

        previous_node = state["_previous"]
        next_node = state["_next"]

        if next_node == "viz_agent":
            viz = state["visualization"]
            # sql_agent → viz_agent
            if previous_node == "sql_agent":
                sql_answer = state["sql_answer"]
                final_answer = f"{sql_answer}\n\n{viz.insights}" if viz else sql_answer
                data = state["sql_queries_results"]
            # viz_agent called directly
            else:
                final_answer = viz.insights if viz else ""
                data = _get_data_from_chat_turns(
                    turn_ids=state["data_turn_ids"],
                    chat_history=state["chat_history"]
                )
        # sql_agent → process_outputs
        else:
            state_update["visualization"] = None
            final_answer = state["sql_answer"]
            data = state["sql_queries_results"]

        # update final answer
        state_update["final_answer"] = final_answer

        # create the current chat turn and add it to chat history
        if keys := state["chat_history"].keys():
            current_turn_id = max(keys) + 1
        else:
            current_turn_id = 1

        chat_turn = ChatTurn(
            id=current_turn_id,
            user_question=state["question"],
            ai_response=final_answer,
            data=data,
        )

        state_update["chat_history"] = {chat_turn.id: chat_turn}

        return state_update

    def _prune_history(self, state: RouterAgentState) -> dict[str, dict[int, ChatTurnRemove]]:
        """Removes the oldest chat turn from the conversation history when the maximum
        allowed number of turns is reached, implementing a sliding window strategy.

        This ensures that only the most recent chat turns up to the defined
        `question_limit` are retained when sending context to the LLM.

        Args:
            state (RouterAgentState): The current graph state containing the
                conversation history under the `chat_history` key.

        Returns:
            dict[str, dict[int, ChatTurnRemove]]: The state update.
        """
        chat_history = state["chat_history"]

        if self.question_limit and len(chat_history) >= self.question_limit:
            oldest_turn_id = min(chat_history.keys())
            chat_turn_remove = {oldest_turn_id: ChatTurnRemove(id=oldest_turn_id)}
            return {"chat_history": chat_turn_remove}

        return {"chat_history": {}}

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

        # post sql router node, calls the viz_agent or process_outputs
        graph.add_node("post_sql_router", RunnableLambda(self._call_post_sql_router, self._acall_post_sql_router))

        # node for building the final answer and updating chat history
        graph.add_node("process_outputs", self._process_outputs)

        # node for deleting old chat turns
        graph.add_node("prune_history", self._prune_history)

        graph.add_edge("sql_agent", "post_sql_router")
        graph.add_edge("viz_agent", "process_outputs")
        graph.add_edge("process_outputs", "prune_history")
        graph.add_conditional_edges("initial_router", _initial_route)
        graph.add_conditional_edges("post_sql_router", _post_sql_route)

        graph.set_entry_point("initial_router")
        graph.set_finish_point("prune_history")

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

def _initial_route(state: RouterAgentState) -> Literal["sql_agent", "viz_agent"]:
    """Route to the next node in the beginning of the router agent workflow.

    Returns:
        str: The identifier of the next node to execute. Must be one of:
            - "sql_agent": Route to the SQL agent.
            - "viz_agent": Route to the visualization agent.
    """
    return state["_next"]

def _post_sql_route(state: RouterAgentState) -> Literal["viz_agent", "process_outputs"]:
    """Route to the next node in the router agent workflow after calling the SQL agent.

    Returns:
        str: The identifier of the next node to execute. Must be one of:
            - "viz_agent": Route to the visualization agent.
            - "process_outputs": Route to output processing and response formatting.
    """
    return state["_next"]

def _format_input_message(current_question: str, chat_history: dict[int, ChatTurn]) -> str:
    """Format conversation data into a JSON message for the `RouterAgent`.

    Transforms the current question and chat history into a structured JSON format
    containing the conversation history and the current question. Each chat turn is
    converted to a dictionary with `turn_id`, `user_question`, and `ai_response` fields.

    Args:
        current_question (str): The user's question for the current chat turn.
        chat_history (dict[int, ChatTurn]): Dictionary mapping turn IDs to `ChatTurn` objects.

    Returns:
        str: A JSON-formatted string containing the conversation history
            and the current question, ready for `RouterAgent` processing.

    Example:
        >>> format_input_message("What is AI?", {1: ChatTurn(...)})
        {
          "conversation_history": [...],
          "current_question": "What is AI?"
        }
    """
    chat_history = [
        {
            "turn_id": turn_id,
            "user_question": chat_turn.user_question,
            "ai_response": chat_turn.ai_response
        }
        for turn_id, chat_turn in chat_history.items()
    ]

    input_message = {
        "conversation_history": chat_history,
        "current_question": current_question
    }

    return json.dumps(input_message, ensure_ascii=False, indent=2)

def _normalize_data(data: list[Item]) -> list[Any]:
    """Extract and flatten JSON content from `Item` objects for `VizAgent` processing.

    Processes a list of `Item` objects, extracts JSON content from each item's `content`
    field, parses the JSON, and flattens all parsed objects into a single list.

    Args:
        data (list[Item]): List of `Item` objects where each item must contain
            JSON-serialized content.

    Returns:
        list[Any]: Flattened list containing all successfully parsed JSON objects
            from the input items. Empty list if no valid JSON content is found.

    Note:
        - Items with empty/None content are skipped.
        - JSON parsing errors are logged and the problematic item is skipped.
        - Each item's content should contain a JSON array for proper flattening.

    Example:
        >>> items = [Item(content='[{"a": 1}, {"b": 2}]'), Item(content='[{"c": 3}]')]
        >>> normalize_data(items)
        [{"a": 1}, {"b": 2}, {"c": 3}]
    """
    normalized_data = []
    for item in data:
        if item.content:
            try:
                for obj in json.loads(item.content):
                    normalized_data.append(obj)
            except json.JSONDecodeError:
                logger.exception(f"JSON decode error, skipping {item.content = }:")
    return normalized_data

def _get_data_from_chat_turns(turn_ids: list[int], chat_history: dict[int, ChatTurn]) -> list[Item]:
    """Retrieve and aggregate data from specified chat turns.

    Extracts data from one or more chat turns identified by their IDs and returns
    a flattened list of all `Item` objects found in those turns.

    Args:
        turn_ids (list[int]): List of chat turn identifiers to retrieve data from.
        chat_history (dict[str, ChatTurn]): Dictionary mapping turn IDs to `ChatTurn` objects.

    Returns:
        list[Item]: Flattened list containing all `Item` objects from the `data`
            attribute of each found chat turn. Empty list if no valid turns found.

    Note:
        - Missing turn IDs are logged as warnings but don't halt processing.

    Example:
        >>> chat_turns = {
        ...     1: ChatTurn(..., data=[Item(id="a", ...)]),
        ...     2: ChatTurn(..., data=[Item(id="b", ...)]),
        ... }
        >>> get_data_from_chat_turns([1, 2], chat_turns)
        [Item(id="a", ...), Item(id="b", ...)]
    """
    if turn_ids is None:
        return []
    chat_turns: list[ChatTurn] = []
    for turn_id in turn_ids:
        if chat_turn := chat_history.get(turn_id):
            chat_turns.append(chat_turn)
        else:
            logger.warning(f"Chat turn {turn_id} not found")
    return [item for turn in chat_turns for item in turn.data]
