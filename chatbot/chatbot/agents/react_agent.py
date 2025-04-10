from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from chatbot.loguru_logging import get_logger

from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep

def should_continue(state: State) -> Literal["prune", "tools"]:
    """Routes to the tools node if the last message has any tool calls.
    Otherwise, routes to the message pruning node

    Args:
        state (State): The graph state

    Returns:
        str: The next node to route to
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "prune"

class ReActAgent:
    """A LangGraph ReAct Agent"""

    agent_node = "agent"
    tools_node = "tools"
    prune_node = "prune"

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | BaseToolkit,
        system_message: SystemMessage | str | None = None,
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        question_limit: int | None = 5,
    ):
        if isinstance(tools, BaseToolkit):
            self.tools = tools.get_tools()
        else:
            self.tools = tools

        if isinstance(system_message, str):
            self.system_message = SystemMessage(system_message)
        else:
            self.system_message = system_message

        self.model = model.bind_tools(self.tools)

        if self.system_message:
            self.model_runnable = (lambda messages: [self.system_message] + messages) | self.model
        else:
            self.model_runnable = self.model

        self.checkpointer = checkpointer

        self.question_limit = question_limit

        self.logger = get_logger(self.__class__.__name__)

        self.graph = self._compile()

    def _call_model(self, state: State, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Calls the LLM on a message list

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, list[BaseMessage]]: The updated message list
        """
        messages = state["messages"]
        is_last_step = state["is_last_step"]

        response = self.model_runnable.invoke(messages, config)

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

    async def _acall_model(self, state: State, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        """Asynchronously calls the LLM on a message list

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, list[BaseMessage]]: The updated message list
        """
        messages = state["messages"]
        is_last_step = state["is_last_step"]

        response = await self.model_runnable.ainvoke(messages, config)

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

        graph.add_node(self.agent_node, RunnableLambda(self._call_model, self._acall_model))
        graph.add_node(self.tools_node, ToolNode(self.tools))
        graph.add_node(self.prune_node, self._prune_messages)

        graph.set_entry_point(self.agent_node)
        graph.add_conditional_edges(self.agent_node, should_continue)
        graph.add_edge(self.tools_node, self.agent_node)
        graph.set_finish_point(self.prune_node)

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
        message = HumanMessage(content=question.strip())

        response = self.graph.invoke(
            input={"messages": [message]},
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
        message = HumanMessage(content=question.strip())

        response = await self.graph.ainvoke(
            input={"messages": [message]},
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
