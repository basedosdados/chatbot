import uuid

import pytest
import pytest_asyncio

from chatbot.agents.router_agent import RouterAgent, RouterAgentState
from chatbot.assistants import AsyncSQLVizAssistant, SQLVizAssistantMessage


@pytest_asyncio.fixture
async def assistant(monkeypatch):
    """Mock AsyncSQLVizAssistant"""
    def mock_agent_init(self, checkpointer):
        self.checkpointer = checkpointer

    async def mock_ainvoke(self, question, config, rewrite_query):
        return RouterAgentState(
            _previous="mock previous",
            _next="mock next",
            question=question,
            rewrite_query=False,
            sql_answer="mock sql answer",
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            data_turn_ids=None,
            question_for_viz_agent=None,
            visualization=None,
            chat_history={},
        )

    async def mock_astream(self, question, config, stream_mode, subgraphs, rewrite_query):
        yield RouterAgentState(
            _previous="mock previous",
            _next="mock next",
            question=question,
            rewrite_query=False,
            sql_answer="mock sql answer",
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            data_turn_ids=None,
            question_for_viz_agent=None,
            visualization=None,
            chat_history={},
        )

    def mock_assistant_init(self, router_agent):
        self.router_agent = router_agent

    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(RouterAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(RouterAgent, "astream", mock_astream)
    monkeypatch.setattr(AsyncSQLVizAssistant, "__init__", mock_assistant_init)

    mock_agent = RouterAgent(checkpointer=None)

    mock_assistant = AsyncSQLVizAssistant(router_agent=mock_agent)

    return mock_assistant

@pytest.mark.asyncio
async def test_invoke(assistant: AsyncSQLVizAssistant):
    response = await assistant.ainvoke("mock question")

    expected_response = SQLVizAssistantMessage(
        content="mock final answer",
        sql_queries=None,
        visualization=None,
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries
    assert response.visualization == expected_response.visualization

@pytest.mark.asyncio
async def test_invoke_with_config(assistant: AsyncSQLVizAssistant):
    run_id = str(uuid.uuid4())

    response = await assistant.ainvoke("mock question", {"run_id": run_id})

    expected_response = SQLVizAssistantMessage(
        id=run_id,
        content="mock final answer",
        sql_queries=None,
        visualization=None,
    )

    assert response.id == expected_response.id
    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries
    assert response.visualization == expected_response.visualization

@pytest.mark.asyncio
async def test_stream(assistant: AsyncSQLVizAssistant):
    async for chunk in assistant.astream("mock question"):
        assert isinstance(chunk, dict)

    final_state = chunk

    for k in RouterAgentState.__required_keys__:
        assert k in final_state.keys()

@pytest.mark.asyncio
async def test_clear_thread(assistant: AsyncSQLVizAssistant):
    await assistant.aclear_thread(thread_id=str(uuid.uuid4()))
