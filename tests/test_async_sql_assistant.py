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

    async def mock_ainvoke(self, question, config):
        return SQLAgentState(
            question=question,
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            similar_examples=[],
            messages=[],
            is_last_step=False
        )

    def mock_assistant_init(self, sql_agent):
        self.sql_agent = sql_agent

    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(SQLAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(AsyncSQLAssistant, "__init__", mock_assistant_init)

    mock_agent = SQLAgent(checkpointer=None)

    mock_assistant = AsyncSQLAssistant(sql_agent=mock_agent)

    return mock_assistant

@pytest.mark.asyncio
async def test_invoke(assistant: AsyncSQLAssistant):
    response = await assistant.invoke("mock question")

    expected_response = SQLAssistantMessage(
        content="mock final answer",
        sql_queries=None,
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

@pytest.mark.asyncio
async def test_invoke_with_config(assistant: AsyncSQLAssistant):
    run_id = str(uuid.uuid4())

    response = await assistant.invoke("mock question", {"run_id": run_id})

    expected_response = SQLAssistantMessage(
        id=run_id,
        content="mock final answer",
        sql_queries=None,
    )

    assert response.id == expected_response.id
    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

@pytest.mark.asyncio
async def test_clear_thread(assistant: AsyncSQLAssistant):
    await assistant.clear_thread(thread_id=str(uuid.uuid4()))
