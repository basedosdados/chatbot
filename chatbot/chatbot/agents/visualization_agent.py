from typing import Annotated, Any, TypedDict

import pydantic
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage)
from langchain_core.runnables import (RunnableConfig, RunnableLambda,
                                      RunnableSequence)
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages

from chatbot.loguru_logging import get_logger

from .prompts import (CHART_METADATA_SYSTEM_PROMPT,
                      CHART_PREPROCESS_BASE_SYSTEM_PROMPT,
                      CHART_PREPROCESS_SYSTEM_PROMPT,
                      REPHRASER_VIZ_SYSTEM_PROMPT,
                      VALIDATION_VIZ_SYSTEM_PROMPT)
from .reducers import Item
from .structured_outputs import Chart, ChartData, ChartMetadata, Rephrase
from .utils import async_delete_checkpoints, delete_checkpoints, prune_messages

EXAMPLE_TEMPLATE = """<example>
User question: {user_question}

<query_results>
{query_results}
</query_results>

Output: {output}
</example>"""


class State(TypedDict):
    # input question
    question: str

    # rephrased question
    question_rephrased: str

    # the sql answer, which will serve as a starting point for the chart answer
    sql_answer: str

    # the chart answer
    chart_answer: str

    # sql queries that were executed without errors and its results
    sql_queries: list[Item]
    sql_queries_results: list[Item]

    # visualization agent's message list
    messages: Annotated[list[BaseMessage], add_messages]

    # data and metadata for chart plotting and the final chart object
    chart: Chart
    chart_data: ChartData
    chart_metadata: ChartMetadata

class VizAgent:
    def __init__(
        self,
        model: BaseChatModel,
        checkpointer: PostgresSaver | AsyncPostgresSaver | bool | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = 4,
        question_limit: int | None = 5,
    ):
        rephraser_sytem_message = SystemMessage(REPHRASER_VIZ_SYSTEM_PROMPT)

        self.rephraser_runnable = (
            lambda question: [rephraser_sytem_message] + [question]
        ) | model.with_structured_output(Rephrase)

        chart_metadata_sytem_message = SystemMessage(CHART_METADATA_SYSTEM_PROMPT)

        self.chart_metadata_runnable = (
            lambda messages: [chart_metadata_sytem_message] + messages
        ) | model.with_structured_output(ChartMetadata)

        validation_system_message = SystemMessage(VALIDATION_VIZ_SYSTEM_PROMPT)

        self.validation_runnable = (
            lambda messages: [validation_system_message] + messages
        ) | model

        self.preprocess_model = model.with_structured_output(ChartData)

        self.checkpointer = checkpointer

        self.vector_store = vector_store
        self.top_k = top_k

        self.question_limit = question_limit

        self.logger = get_logger(self.__class__.__name__)

        self.graph = self._compile()

    def _call_rephrase_question(self, state: State, config: RunnableConfig) -> dict[str, str|list[HumanMessage]]:
        """Calls the model for rephrasing the user's question

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|HumanMessage]: The state update containing the rephrased question
        """
        question = state["question"]
        response: Rephrase = self.rephraser_runnable.invoke(question, config)
        return {"question_rephrased": response.rephrased}

    async def _acall_rephrase_question(self, state: State, config: RunnableConfig) -> dict[str, str|list[HumanMessage]]:
        """Asynchronously calls the model for rephrasing the user's question

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|HumanMessage]: The state update containing the rephrased question
        """
        question = state["question"]
        response: Rephrase = await self.rephraser_runnable.ainvoke(question, config)
        return {"question_rephrased": response.rephrased}

    def _get_queries_and_results(self, state: State) -> dict[str, list[HumanMessage]]:
        """Builds the message list from the SQL queries and SQL queries restuls lists

        Args:
            state (State): The graph state

        Returns:
            dict[str, list[HumanMessage]]: A dictionary containing the message list
        """
        question = state["question_rephrased"]

        queries = "\n\n".join([
            f"<query>\n{q.content}\n</query>"
            for q in state["sql_queries"]
        ])

        if queries:
            queries = "\n" + queries

        queries_results = "\n\n".join([
            f"<query_results>\n{qr.content}\n</query_results>"
            for qr in state["sql_queries_results"]
        ])

        if queries_results:
            queries_results = "\n" + queries_results

        message = HumanMessage(
            f"User question: {question}\n\nQueries:{queries}\n\nQueries results:{queries_results}"
        )

        return {"messages": [message]}

    def _similarity_search(self, question: str) -> list[Document]:
        """Searches for the top-k most similar examples to the input question

        Args:
            question (str): The input question

        Returns:
            list[Document]: A list of the top-k most similar examples
        """
        if self.vector_store is not None:
            return self.vector_store.similarity_search(
                query=question,
                k=self.top_k
            )
        return []

    async def _asimilarity_search(self, question: str) -> list[Document]:
        """Asynchronously searches for the top-k most similar examples to the input question

        Args:
            question (str): The input question

        Returns:
            list[Document]: A list of the top-k most similar examples
        """
        if self.vector_store is not None:
            return await self.vector_store.asimilarity_search(
                query=question,
                k=self.top_k
            )
        return []

    def _get_preprocess_runnable(self, documents: list[Document]) -> RunnableSequence:
        """Dynamically builds a model runnable with a few-shot system prompt and the preprocessing model

        Args:
            documents (list[Document]): A list of documents

        Returns:
            RunnableSequence: The model runnable
        """
        examples = "\n\n".join(
            EXAMPLE_TEMPLATE.format(
                user_question=doc.page_content,
                query_results=doc.metadata['query_results'],
                output=doc.metadata['query_results_preprocessed']
            )
            for doc in documents
        )

        if examples:
            system_message = SystemMessage(
                content=CHART_PREPROCESS_SYSTEM_PROMPT.format(examples=examples)
            )
        else:
            system_message = SystemMessage(content=CHART_PREPROCESS_BASE_SYSTEM_PROMPT)

        return (lambda messages: [system_message] + messages) | self.preprocess_model

    def _call_preprocess_data(self, state: State, config: RunnableConfig) -> dict[str, ChartData]:
        """Calls the model for preprocessing the queries results

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, list]: A dictionary containing the processed queries results
        """
        question = state["question_rephrased"]
        messages = state["messages"]

        documents = self._similarity_search(question)

        preprocess_runnable = self._get_preprocess_runnable(documents)

        try:
            chart_data: ChartData = preprocess_runnable.invoke(messages, config)
        except pydantic.ValidationError:
            self.logger.exception("Error on data preprocessing:")
            chart_data = ChartData()

        return {
            "chart_data": chart_data,
            "messages": [AIMessage(chart_data.model_dump_json(indent=4))]
        }

    async def _acall_preprocess_data(self, state: State, config: RunnableConfig) -> dict[str, ChartData]:
        """Asynchronously calls the model for preprocessing the queries results

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, list]: A dictionary containing the processed queries results
        """
        question = state["question_rephrased"]
        messages = state["messages"]

        documents = await self._asimilarity_search(question)

        preprocess_runnable = self._get_preprocess_runnable(documents)

        try:
            chart_data: ChartData = await preprocess_runnable.ainvoke(messages, config)
        except pydantic.ValidationError:
            self.logger.exception("Error on data preprocessing:")
            chart_data = ChartData()

        return {
            "chart_data": chart_data,
            "messages": [AIMessage(chart_data.model_dump_json(indent=4))]
        }

    def _call_chart_metadata(self, state: State, config: RunnableConfig) -> dict[str, ChartMetadata]:
        """Calls the model for generating chart metadata

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|ChartMetadata]: A dictionary containing the final answer and the chart metadata
        """
        question = state["question_rephrased"]
        chart_data = state["chart_data"]

        messages = [HumanMessage(
            f"User question: {question}\n\nQuery results: {chart_data.data}"
        )]

        try:
            chart_metadata: ChartMetadata = self.chart_metadata_runnable.invoke(messages, config)
        except pydantic.ValidationError:
            self.logger.exception("Error on metadata extraction:")
            chart_metadata = ChartMetadata()

        return {"chart_metadata": chart_metadata}

    async def _acall_chart_metadata(self, state: State, config: RunnableConfig) -> dict[str, ChartMetadata]:
        """Asynchronously calls the model for generating chart metadata

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|ChartMetadata]: A dictionary containing the final answer and the chart metadata
        """
        question = state["question_rephrased"]
        chart_data = state["chart_data"]

        messages = [HumanMessage(
            f"User question: {question}\n\nQuery results: {chart_data.data}"
        )]

        try:
            chart_metadata: ChartMetadata = await self.chart_metadata_runnable.ainvoke(messages, config)
        except pydantic.ValidationError:
            self.logger.exception("Error on metadata extraction:")
            chart_metadata = ChartMetadata()

        return {"chart_metadata": chart_metadata}

    def _validate_chart(self, chart_data: ChartData, chart_metadata: ChartMetadata) -> bool:
        """Checks if the recommended chart can be plotted using Plotly

        Args:
            chart_data (ChartData): The chart data
            chart_metadata (ChartMetadata): The chart metadata

        Returns:
            bool: Whether the chart can be plotted or not
        """
        if chart_data.data is None or \
           chart_metadata.chart_type is None:
            return False

        vars = {
            chart_metadata.x_axis,
            chart_metadata.y_axis
        }

        if chart_metadata.label is not None:
            vars.add(chart_metadata.label)

        if vars == chart_data.data.keys():
           return True
        else:
            self.logger.error(f"Chart validation error: one or more of {vars} are not in {chart_data.data.keys()}")
            return False

    def _call_get_answer(self, state: State, config: RunnableConfig) -> dict[str, str|Chart]:
        """Builds the Chart object and generates an answer to introduce it

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|Chart]: A dictionary containing the Chart object and the generated answer
        """
        question = state["question"]
        sql_answer = state["sql_answer"]
        chart_data = state["chart_data"]
        chart_metadata = state["chart_metadata"]

        chart = Chart(
            data=chart_data,
            metadata=chart_metadata,
            is_valid=self._validate_chart(chart_data, chart_metadata)
        )

        response = self.validation_runnable.invoke(
            input=[
                HumanMessage(f"User question: {question}"),
                HumanMessage(f"Question answer: \n\n{sql_answer}"),
                HumanMessage(f"Chart: {chart.model_dump_json(indent=4)}"),
            ],
            config=config
        )

        # ensuring we won't duplicate the sql answer
        response.content = response.content.replace(sql_answer, "").strip()

        return {
            "chart": chart,
            "chart_answer": response.content
        }

    async def _acall_get_answer(self, state: State, config: RunnableConfig) -> dict[str, str|Chart]:
        """Builds the Chart object and generates an answer to introduce it

        Args:
            state (State): The graph state
            config (RunnableConfig): A config to use when calling the LLM

        Returns:
            dict[str, str|Chart]: A dictionary containing the Chart object and the generated answer
        """
        question = state["question"]
        sql_answer = state["sql_answer"]
        chart_data = state["chart_data"]
        chart_metadata = state["chart_metadata"]

        chart = Chart(
            data=chart_data,
            metadata=chart_metadata,
            is_valid=self._validate_chart(chart_data, chart_metadata)
        )

        response = await self.validation_runnable.ainvoke(
            input=[
                HumanMessage(f"User question: {question}"),
                HumanMessage(f"Question answer: \n\n{sql_answer}"),
                HumanMessage(f"Chart: {chart.model_dump_json(indent=4)}"),
            ],
            config=config
        )

        # ensuring we won't duplicate the sql answer
        response.content = response.content.replace(sql_answer, "").strip()

        return {
            "chart": chart,
            "chart_answer": response.content
        }

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

        # user's question rephrasing node
        graph.add_node("rephrase", RunnableLambda(self._call_rephrase_question, self._acall_rephrase_question))

        # node for getting sql queries and its results
        graph.add_node("get_data", self._get_queries_and_results)

        # preprocessing node
        graph.add_node("preprocess_data", RunnableLambda(self._call_preprocess_data, self._acall_preprocess_data))

        # plot recommendation node
        graph.add_node("get_metadata", RunnableLambda(self._call_chart_metadata, self._acall_chart_metadata))

        # plot validation and answer generation node
        graph.add_node("get_answer", RunnableLambda(self._call_get_answer, self._acall_get_answer))

        # message pruning node
        graph.add_node("prune_messages", self._prune_messages)

        graph.add_edge("rephrase", "get_data")
        graph.add_edge("get_data", "preprocess_data")
        graph.add_edge("preprocess_data", "get_metadata")
        graph.add_edge("get_metadata", "get_answer")
        graph.add_edge("get_answer", "prune_messages")

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
        sql_answer: str,
        sql_queries: list[Item],
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None
    ) -> dict[str, Any] | Any:
        """Runs the compiled graph with a question and an optional configuration

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): The configuration. Defaults to None.

        Returns:
            dict[str, Any] | Any: The last output of the graph run
        """
        question = question.strip()

        response = self.graph.invoke(
            input={
                "question": question,
                "sql_answer": sql_answer,
                "sql_queries": sql_queries,
                "sql_queries_results": sql_queries_results
            },
            config=config,
        )

        return response

    async def ainvoke(
        self,
        question: str,
        sql_answer: str,
        sql_queries: list[Item],
        sql_queries_results: list[Item],
        config: RunnableConfig | None = None
    ) -> dict[str, Any] | Any:
        """Asynchronously runs the compiled graph with a question and an optional configuration

        Args:
            question (str): The question
            config (RunnableConfig | None, optional): The configuration. Defaults to None.

        Returns:
            dict[str, Any] | Any: The last output of the graph run
        """
        question = question.strip()

        response = await self.graph.ainvoke(
            input={
                "question": question,
                "sql_answer": sql_answer,
                "sql_queries": sql_queries,
                "sql_queries_results": sql_queries_results
            },
            config=config,
        )

        return response

    # Unfortunately, there is no clean way to delete an agent's memory
    # except by deleting its checkpoints, as noted in this github discussion:
    # https://github.com/langchain-ai/langgraph/discussions/912
    def clear_memory(self, thread_id: str):
        """Clears the assistant memory

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
            self.logger.exception(f"Error on clearing memory for thread {thread_id}:")

    async def aclear_memory(self, thread_id: str):
        """Asynchronously Clears the assistant memory

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
            self.logger.exception(f"Error on clearing memory for thread {thread_id}:")
