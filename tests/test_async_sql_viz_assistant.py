import uuid

import pytest
import pytest_asyncio

from chatbot.agents.router_agent import RouterAgent, RouterAgentState
from chatbot.agents.structured_outputs import Chart, ChartData, ChartMetadata
from chatbot.assistants import AsyncSQLVizAssistant, SQLVizAssistantMessage


@pytest_asyncio.fixture
async def assistant(monkeypatch):
    """Mock AsyncSQLVizAssistant"""
    def mock_agent_init(self, checkpointer):
        self.checkpointer = checkpointer

    async def mock_ainvoke(self, question, config, rewrite_query):
        chart_data = ChartData()
        chart_metadata = ChartMetadata()
        chart = Chart(
            data=chart_data,
            metadata=chart_metadata,
            is_valid=False
        )
        return RouterAgentState(
            _previous="mock previous",
            _next="mock next",
            question=question,
            sql_answer="mock sql answer",
            chart_answer="mock chart answer",
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            chart=chart,
            messages=[],
        )

    def mock_assistant_init(self, router_agent):
        self.router_agent = router_agent

    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(RouterAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(AsyncSQLVizAssistant, "__init__", mock_assistant_init)

    mock_agent = RouterAgent(checkpointer=None)

    mock_assistant = AsyncSQLVizAssistant(router_agent=mock_agent)

    return mock_assistant

@pytest.mark.asyncio
async def test_invoke(assistant: AsyncSQLVizAssistant):
    response = await assistant.invoke("mock question")

    expected_response = SQLVizAssistantMessage(
        content="mock final answer",
        sql_queries=None,
        chart=Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries
    assert response.chart == expected_response.chart

@pytest.mark.asyncio
async def test_invoke_with_config(assistant: AsyncSQLVizAssistant):
    run_id = str(uuid.uuid4())

    response = await assistant.invoke("mock question", {"run_id": run_id})

    expected_response = SQLVizAssistantMessage(
        id=run_id,
        content="mock final answer",
        sql_queries=None,
        chart=Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    )

    assert response.id == expected_response.id
    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries
    assert response.chart == expected_response.chart

@pytest.mark.asyncio
async def test_clear_thread(assistant: AsyncSQLVizAssistant):
    await assistant.clear_thread(thread_id=str(uuid.uuid4()))
