import uuid

import pytest
import pytest_asyncio

from chatbot.agents.sql_agent import SQLAgent, SQLAgentState
from chatbot.assistants import AsyncSQLAssistant, SQLAssistantMessage


@pytest_asyncio.fixture
async def assistant(monkeypatch):
    """Mock AsyncSQLAssistant"""
    def mock_agent_init(self, checkpointer):
        self.checkpointer = checkpointer

    async def mock_ainvoke(self, question, config, rewrite_query):
        return SQLAgentState(
            question=question,
            question_rewritten=None,
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            messages=[],
            rewrite_query=False,
            is_last_step=False
        )

    async def mock_astream(self, question, stream_mode, config, rewrite_query):
        mock_final_state = SQLAgentState(
            question=question,
            question_rewritten=None,
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            messages=[],
            rewrite_query=False,
            is_last_step=False
        )

        yield mock_final_state

    def mock_assistant_init(self, sql_agent):
        self.sql_agent = sql_agent

    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(SQLAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(SQLAgent, "astream", mock_astream)
    monkeypatch.setattr(AsyncSQLAssistant, "__init__", mock_assistant_init)

    mock_agent = SQLAgent(checkpointer=None)

    mock_assistant = AsyncSQLAssistant(sql_agent=mock_agent)

    return mock_assistant

@pytest.mark.asyncio
async def test_invoke(assistant: AsyncSQLAssistant):
    response = await assistant.ainvoke("mock question")

    expected_response = SQLAssistantMessage(
        content="mock final answer",
        sql_queries=None,
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

@pytest.mark.asyncio
async def test_invoke_with_config(assistant: AsyncSQLAssistant):
    run_id = str(uuid.uuid4())

    response = await assistant.ainvoke("mock question", {"run_id": run_id})

    expected_response = SQLAssistantMessage(
        id=run_id,
        content="mock final answer",
        sql_queries=None,
    )

    assert response.id == expected_response.id
    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

@pytest.mark.asyncio
async def test_stream(assistant: AsyncSQLAssistant):
    async for chunk in assistant.astream("mock question"):
        assert isinstance(chunk, dict)

    final_state = chunk

    for k in SQLAgentState.__required_keys__:
        assert k in final_state.keys()

@pytest.mark.asyncio
async def test_clear_thread(assistant: AsyncSQLAssistant):
    await assistant.aclear_thread(thread_id=str(uuid.uuid4()))
