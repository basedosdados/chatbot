import json
from typing import Any, AsyncIterator, Iterator, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

from .prompts import VIZ_SYSTEM_PROMPT
from .structured_outputs import VizScript, Visualization


class VizAgentState(TypedDict):
    # input question
    question: str

    # normalized sql queries results
    data: list[dict[str, Any]]

    # visualization output
    visualization: Visualization | None

class VizAgent:
    """LLM-powered Agent for creating visualizations.

    Args:
        model (BaseChatModel):
            A LangChain chat model with structured output support.
    """

    def __init__(self, model: BaseChatModel):
        self.viz_runnable = (
            lambda messages: [SystemMessage(VIZ_SYSTEM_PROMPT)] + messages
        ) | model.with_structured_output(VizScript)

        self.graph = self._compile()

    def _create_visualization(self, state: VizAgentState) -> dict[str, Visualization|None]:
        input_message = {
            "user_question": state["question"],
            "data": state["daa"]
        }

        messages = [HumanMessage(json.dumps(input_message, indent=2))]

        viz_script: VizScript = self.viz_runnable.invoke(messages)

        if viz_script.script:
            script_data = viz_script.model_dump()
            script_data["data"] = state["data"]
            visualization = Visualization(**script_data)
        else:
            visualization = None

        return {"visualization": visualization}

    async def _acreate_visualization(self, state: VizAgentState) -> dict[str, Visualization|None]:
        input_message = {
            "user_question": state["question"],
            "data": state["data"]
        }

        messages = [HumanMessage(json.dumps(input_message, indent=2))]

        viz_script: VizScript = await self.viz_runnable.ainvoke(messages)

        if viz_script.script:
            script_data = viz_script.model_dump()
            script_data["data"] = state["data"]
            visualization = Visualization(**script_data)
        else:
            visualization = None

        return {"visualization": visualization}

    def _compile(self) -> CompiledGraph:
        """Compiles the state graph into a LangChain Runnable.

        Returns:
            CompiledGraph: The compiled state graph.
        """
        graph = StateGraph(VizAgentState)

        graph.add_node(
            "create_visualization",
            RunnableLambda(self._create_visualization, self._acreate_visualization)
        )

        graph.set_entry_point("create_visualization")
        graph.set_finish_point("create_visualization")

        return graph.compile()

    def invoke(
        self,
        question: str,
        data: list[dict[str, Any]],
    ) -> VizAgentState:
        """Runs the compiled graph.

        Args:
            question (str): The input question.
            data (list[Item]): The data fetched for the input question.

        Returns:
            VizAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = self.graph.invoke(
            input={
                "question": question,
                "data": data
            }
        )

        return response

    async def ainvoke(
        self,
        question: str,
        data: list[dict[str, Any]],
    ) -> VizAgentState:
        """Asynchronously runs the compiled graph.

        Args:
            question (str): The input question.
            data (list[Item]): The data fetched for the input question.

        Returns:
            VizAgentState: The output of the agent execution.
        """
        question = question.strip()

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "data": data
            }
        )

        return response

    def stream(
        self,
        question: str,
        data: list[dict[str, Any]],
        stream_mode: list[str] | None = None,
    ) -> Iterator[dict|tuple]:
        """Stream graph steps.

        Args:
            question (str): The input question.
            data (list[Item]): The data fetched for the input question.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.

        Yields:
            dict|tuple: The output for each step in the graph. Its type, shape and content depends on the `stream_mode` arg.
        """
        question = question.strip()

        for chunk in self.graph.stream(
            input={
                "question": question,
                "data": data
            },
            stream_mode=stream_mode,
        ):
            yield chunk

    async def astream(
        self,
        question: str,
        data: list[dict[str, Any]],
        stream_mode: list[str] | None = None,
    ) -> AsyncIterator[dict|tuple]:
        """Asynchronously stream graph steps.

        Args:
            question (str): The input question.
            data (list[Item]): The data fetched for the input question.
            stream_mode (list[str] | None, optional): The mode to stream output. See the LangGraph streaming guide in
                https://langchain-ai.github.io/langgraph/how-tos/streaming for more details. Defaults to `None`.

        Yields:
            dict|tuple: The output for each step in the graph. Its type, shape and content depends on the `stream_mode` arg.
        """
        question = question.strip()

        async for chunk in self.graph.astream(
            input={
                "question": question,
                "data": data
            },
            stream_mode=stream_mode,
        ):
            yield chunk
