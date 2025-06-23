import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
from langchain_core.runnables import (RunnableConfig, RunnableLambda,
                                      RunnableSequence)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from loguru import logger

from chatbot.contexts import BaseContextProvider
from chatbot.formatters import SQLPromptContext, SQLPromptFormatter
from chatbot.tools import (DatasetsTablesInfoTool, ListDatasetsTool,
                           QueryCheckTool, QueryTableTool)

from .prompts import REWRITE_QUERY_SYSTEM_PROMPT, SELECT_DATASETS_SYSTEM_PROMPT
from .reducers import BaseItem, ItemRemove, add_item
from .structured_outputs import RewrittenQuery
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages


class SQLAgentState(TypedDict):
    # input question and rewritten question
    question: str
    question_rewritten: str | None

    # final answer
    final_answer: str

    # sql queries that were executed without errors and its results
    sql_queries: Annotated[list[BaseItem], add_item]
    sql_queries_results: Annotated[list[BaseItem], add_item]

    # messages list
    messages: Annotated[list[BaseMessage], add_messages]

    # flag indicating if the query should be rewritten for this run
    rewrite_query: bool

    # flag indicating if the recursion limit has been reached
    is_last_step: IsLastStep

class SQLAgent:
    """LLM-powered SQL Agent for interacting with a SQL database.

    Args:
        model (BaseChatModel):
            A LangChain chat model with tool-calling support. Used to:
                1. Select datasets and tables via the provided context provider.
                2. Generate SQL queries and produce the final answer messages.
        context_provider (BaseContextProvider):
            A context provider that supplies all metadata needed by the agent. Implement
            this abstract base to plug in any data source (BigQuery, Postgres, etc.)
            without changing the agent's orchestration logic.
        prompt_formatter (SQLPromptFormatter):
            A formatter responsible for constructing the LLM system prompt during SQL generation step,
            based on the user's question and optional few-shot examples. Must implement how examples
            are retrieved and how the prompt template is composed.
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
        context_provider: BaseContextProvider,
        prompt_formatter: SQLPromptFormatter,
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        question_limit: int | None = 5,
    ):
        self.context_provider = context_provider
        self.prompt_formatter = prompt_formatter
        self.checkpointer = checkpointer
        self.question_limit = question_limit

        # query rewriting model
        rewriter_system_message = SystemMessage(
            content=REWRITE_QUERY_SYSTEM_PROMPT
        )

        self.rewriting_runnable = (
            lambda messages: [rewriter_system_message] + messages
        ) | model.with_structured_output(RewrittenQuery)

        # datasets and tables selection tools
        self.list_datasets_tool = ListDatasetsTool(
            context_provider=self.context_provider
        )
        self.tables_info_tool = DatasetsTablesInfoTool(
            context_provider=self.context_provider
        )

        # query checking and execution tools
        self.query_check_tool = QueryCheckTool(llm=model)
        self.query_table_tool = QueryTableTool(
            context_provider=self.context_provider
        )

        select_datasets_system_message = SystemMessage(
            content=SELECT_DATASETS_SYSTEM_PROMPT
        )

        select_datasets_model = model.bind_tools(
            [self.tables_info_tool],
            tool_choice=self.tables_info_tool.name
        )

        self.select_datasets_runnable = (
            lambda messages: [select_datasets_system_message] + messages
        ) | select_datasets_model

        self.query_model = model.bind_tools([
            self.query_check_tool,
            self.query_table_tool
        ])

        self.graph = self._compile()

    def _call_model(
        self,
        messages: list[BaseMessage],
        is_last_step: bool,
        config: RunnableConfig,
        model_runnable: RunnableSequence
    ) -> dict[str, list[BaseMessage]]:
        """Calls the LLM on a message list.

        Args:
            messages: The message list.
            is_last_step: Flag that indicates if the maximum number of steps has been reached.
            config (RunnableConfig): Configuration for the agent execution.
            model_runnable: A model runnable.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        response = model_runnable.invoke(messages, config)

        if is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Desculpe, não consegui encontrar uma resposta para a sua pergunta. Sinta-se à vontade para reformulá-la ou tentar perguntar algo diferente."
                    )
                ]
            }

        return {"messages": [response]}

    async def _acall_model(
        self,
        messages: list[BaseMessage],
        is_last_step: bool,
        config: RunnableConfig,
        model_runnable: RunnableSequence
    ) -> dict[str, list[BaseMessage]]:
        """Asynchronously calls the LLM on a message list.

        Args:
            messages: The message list.
            is_last_step: Flag that indicates if the maximum number of steps has been reached.
            config (RunnableConfig): Configuration for the agent execution.
            model_runnable: A model runnable.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        response = await model_runnable.ainvoke(messages, config)

        if is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Desculpe, não consegui encontrar uma resposta para a sua pergunta. Sinta-se à vontade para reformulá-la ou tentar perguntar algo diferente."
                    )
                ]
            }

        return {"messages": [response]}

    def _call_rewrite_query(self, state: SQLAgentState) -> dict[str, str]:
        """Rewrites the user query for semantic search.

        Args:
            state (SQLAgentState): The graph state.

        Returns:
            dict[str, str]: The state update containing the rewritten query.
        """
        if not state["rewrite_query"]:
            return {"question_rewritten": None}

        user_queries = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]

        chat_history = user_queries[:-1]
        chat_history = "\n".join([f"{i+1}. {msg.content}" for i, msg in enumerate(user_queries)])
        chat_history = "\n" + chat_history if chat_history else chat_history

        latest_query = user_queries[-1].content

        message = (
            f"Conversation History:{chat_history}\n\n"
            f"Latest User Query:\n{latest_query}\n\n"
            "Rewritten Query:\n"
        )

        response: RewrittenQuery = self.rewriting_runnable.invoke([message])

        return {"question_rewritten": response.rewritten}

    async def _acall_rewrite_query(self, state: SQLAgentState) -> dict[str, str]:
        """Asynchronously rewrites the user query for semantic search.

        Args:
            state (SQLAgentState): The graph state.

        Returns:
            dict[str, str]: The state update containing the rewritten query.
        """
        if not state["rewrite_query"]:
            return {"question_rewritten": None}

        user_queries = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]

        chat_history = user_queries[:-1]
        chat_history = "\n".join([f"{i+1}. {msg.content}" for i, msg in enumerate(user_queries)])
        chat_history = "\n" + chat_history if chat_history else chat_history

        latest_query = user_queries[-1].content

        message = (
            f"Conversation History:{chat_history}\n\n"
            f"Latest User Query:\n{latest_query}\n\n"
            "Rewritten Query:\n"
        )

        response: RewrittenQuery = await self.rewriting_runnable.ainvoke([message])

        return {"question_rewritten": response.rewritten}

    def _call_list_datasets(self, state: SQLAgentState) -> dict[str, list[AIMessage]]:
        """Forces the dataset listing tool call.

        Args:
            state (SQLAgentState): The graph state.

        Returns:
            dict[str, list[AIMessage]]: The dataset listing tool call message.
        """
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": str(uuid.uuid4()),
                    "name": self.list_datasets_tool.name,
                    "args": {"query": self._get_query(state)}
                }
            ]
        )

        return {"messages": [message]}

    async def _acall_list_datasets(self, state: SQLAgentState) -> dict[str, list[AIMessage]]:
        """Asynchronously forces the dataset listing tool call.

        Args:
            state (SQLAgentState): The graph state.

        Returns:
            dict[str, list[AIMessage]]: The dataset listing tool call message.
        """
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": str(uuid.uuid4()),
                    "name": self.list_datasets_tool.name,
                    "args": {"query": self._get_query(state)}
                }
            ]
        )

        return {"messages": [message]}

    def _call_select_datasets(self, state: SQLAgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Calls the model responsible for choosing the table and for calling the tables info tool.

        Args:
            state (SQLAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        # Only the human message and the messages related to the ListDatasets tool
        # are sent to the model responsible for calling the DatasetsTablesInfo tool
        messages = state["messages"]
        filtered_messages = []

        for msg in reversed(messages):
            filtered_messages.insert(0, msg)
            if isinstance(msg, HumanMessage):
                break

        is_last_step = state["is_last_step"]

        return self._call_model(filtered_messages, is_last_step, config, self.select_datasets_runnable)

    async def _acall_select_datasets(self, state: SQLAgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Asynchronously calls the model responsible for choosing the table and for calling the tables info tool.

        Args:
            state (SQLAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        # Only the human message and the messages related to the ListDatasets tool
        # are sent to the model responsible for calling the DatasetsTablesInfo tool
        messages = state["messages"]
        filtered_messages = []

        for msg in reversed(messages):
            filtered_messages.insert(0, msg)
            if isinstance(msg, HumanMessage):
                break

        is_last_step = state["is_last_step"]

        return await self._acall_model(filtered_messages, is_last_step, config, self.select_datasets_runnable)

    def _get_selected_datasets(self, messages: list[BaseMessage]) -> list[str]:
        """Gets a filter to be applied in the similarity search.

        Args:
            messages (list[BaseMessage]): The message list.

        Returns:
            list[str]: A list of the selected datasets names.
        """
        selected_datasets = []
        stop = False

        for message in reversed(messages):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if tool_call["name"] == self.tables_info_tool.name:
                    dataset_names: str = tool_call["args"]["dataset_names"]
                    dataset_names = [name.strip() for name in dataset_names.split(",")]
                    selected_datasets.extend(dataset_names)
                    stop = True
            if stop:
                break

        return selected_datasets

    def _call_query_agent(self, state: SQLAgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Calls the model responsible for the query generation.

        Args:
            state (SQLAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        messages = state["messages"]
        is_last_step = state["is_last_step"]

        selected_datasets = self._get_selected_datasets(messages)

        context = SQLPromptContext(
            query=self._get_query(state),
            selected_datasets=selected_datasets
        )

        system_prompt = self.prompt_formatter.build_system_prompt(context)
        system_message = SystemMessage(content=system_prompt)

        query_model_runnable = (lambda messages: [system_message] + messages) | self.query_model

        return self._call_model(messages, is_last_step, config, query_model_runnable)

    async def _acall_query_agent(self, state: SQLAgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Asynchronously calls the model responsible for the query generation.

        Args:
            state (SQLAgentState): The graph state.
            config (RunnableConfig): Configuration for the agent execution.

        Returns:
            dict[str, list[BaseMessage]]: The updated message list.
        """
        messages = state["messages"]
        is_last_step = state["is_last_step"]

        selected_datasets = self._get_selected_datasets(messages)

        context = SQLPromptContext(
            query=self._get_query(state),
            selected_datasets=selected_datasets
        )

        system_prompt = await self.prompt_formatter.abuild_system_prompt(context)
        system_message = SystemMessage(content=system_prompt)

        query_model_runnable = (lambda messages: [system_message] + messages) | self.query_model

        return await self._acall_model(messages, is_last_step, config, query_model_runnable)

    def _get_query(state: SQLAgentState) -> str:
        if state["rewrite_query"]:
            return state["question_rewritten"]
        return state["question"]

    def _get_answer(self, state: SQLAgentState) -> dict[str, str]:
        last_message = state["messages"][-1]
        return {"final_answer": last_message.content}

    def _clear_sql(self, state: SQLAgentState) -> dict[str, str|list[ItemRemove]]:
        """Clears the SQL agent's answer and the SQL queries and SQL queries results lists.

        Args:
            state (SQLAgentState): The graph state.

        Returns:
            dict[str, list[ItemRemove]]: An empty SQL agent's answer
            and the updated SQL queries and SQL queries results lists.
        """
        sql_queries = [
            ItemRemove(id=item.id) for item in state["sql_queries"]
        ]

        sql_queries_results = [
            ItemRemove(id=item.id) for item in state["sql_queries_results"]
        ]

        return {
            "final_answer": "", # clearing the final_answer is not necessary, but it makes sense
            "sql_queries": sql_queries,
            "sql_queries_results": sql_queries_results
        }

    def _prune_messages(self, state: SQLAgentState) -> dict[str, list[RemoveMessage]]:
        """Prunes the message list to ensure that only a limited number of questions and their
        corresponding AI messages and Tool messages are sent to the LLM.

        Args:
            state (SQLAgentState): The graph state containing the message list.

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
        graph = StateGraph(SQLAgentState)

        # node for clearing previous sql answer, queries, and results
        graph.add_node("clear_sql", self._clear_sql)

        # node for query rewriting
        graph.add_node("maybe_rewrite_query", RunnableLambda(self._call_rewrite_query, self._acall_rewrite_query))

        # list datasets nodes
        graph.add_node("call_list_datasets", RunnableLambda(self._call_list_datasets, self._acall_list_datasets))
        graph.add_node("list_datasets", ToolNode([self.list_datasets_tool]))

        # datasets selection nodes
        graph.add_node("call_select_datasets", RunnableLambda(self._call_select_datasets, self._acall_select_datasets))
        graph.add_node("tables_info", ToolNode([self.tables_info_tool]))

        # ReAct nodes
        graph.add_node("query_agent", RunnableLambda(self._call_query_agent, self._acall_query_agent))
        graph.add_node("tools", ToolNode([self.query_check_tool, self.query_table_tool]))

        # answer retrieval node
        graph.add_node("get_answer", self._get_answer)

        # message pruning node
        graph.add_node("prune_messages", self._prune_messages)

        graph.add_edge("clear_sql", "maybe_rewrite_query")
        graph.add_edge("maybe_rewrite_query", "call_list_datasets")
        graph.add_edge("call_list_datasets", "list_datasets")
        graph.add_edge("list_datasets", "call_select_datasets")
        graph.add_edge("call_select_datasets", "tables_info")
        graph.add_conditional_edges("tables_info", _check_tables_info)
        graph.add_conditional_edges("query_agent", _should_continue)
        graph.add_edge("tools", "query_agent")
        graph.add_edge("get_answer", "prune_messages")

        graph.set_entry_point("clear_sql")
        graph.set_finish_point("prune_messages")

        # The checkpointer is ignored by default when the graph is used as a subgraph
        # For more information, visit https://langchain-ai.github.io/langgraph/how-tos/subgraph-persistence
        # If you want to persist the subgraph state between runs, you must use checkpointer=True
        # For more information, visit https://github.com/langchain-ai/langgraph/issues/3020
        return graph.compile(self.checkpointer)

    def invoke(self, question: str, config: RunnableConfig | None = None, rewrite_query: bool = False) -> SQLAgentState:
        """Runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            SQLAgentState: The output of the agent execution.
        """
        question = question.strip()

        message = HumanMessage(content=question)

        response = self.graph.invoke(
            input={
                "question": question,
                "messages": [message],
                "rewrite_query": rewrite_query,
            },
            config=config,
        )

        return response

    async def ainvoke(self, question: str, config: RunnableConfig | None = None, rewrite_query: bool = False) -> SQLAgentState:
        """Asynchronously runs the compiled graph with a question and an optional configuration.

        Args:
            question (str): The question.
            config (RunnableConfig | None, optional): Optional configuration for the agent execution.

        Returns:
            SQLAgentState: The output of the agent execution.
        """
        question = question.strip()

        message = HumanMessage(content=question)

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "messages": [message],
                "rewrite_query": rewrite_query,
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

def _check_tables_info(state: SQLAgentState) -> Literal["call_select_datasets", "query_agent"]:
    """Checks if the datasets_tables_info tool call returned an error and routes back to the
    call_select_datasets node if it did. Otherwise, proceeds.

    Returns:
        str: The next node to route to.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and "Error: " in last_message.content:
        return "call_select_datasets"
    return "query_agent"

def _should_continue(state: SQLAgentState) -> Literal["tools", "get_answer"]:
    """Routes to the tools node if the last message has any tool calls.
    Otherwise, routes to the end node.

    Args:
        state (SQLAgentState): The graph state.

    Returns:
        str: The next node to route to.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "get_answer"
