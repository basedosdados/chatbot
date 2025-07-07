import uuid

import pytest

from chatbot.agents.sql_agent import SQLAgent, SQLAgentState
from chatbot.assistants import SQLAssistant, SQLAssistantMessage


@pytest.fixture
def assistant(monkeypatch):
    """Mock SQLAssistant"""
    def mock_agent_init(self, checkpointer):
        self.checkpointer = checkpointer

    def mock_invoke(self, question, config, rewrite_query):
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

    def mock_stream(self, question, config, stream_mode, rewrite_query):
        yield SQLAgentState(
            question=question,
            question_rewritten=None,
            final_answer="mock final answer",
            sql_queries=[],
            sql_queries_results=[],
            messages=[],
            rewrite_query=False,
            is_last_step=False
        )

    def mock_assistant_init(self, sql_agent):
        self.sql_agent = sql_agent

    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(SQLAgent, "invoke", mock_invoke)
    monkeypatch.setattr(SQLAgent, "stream", mock_stream)
    monkeypatch.setattr(SQLAssistant, "__init__", mock_assistant_init)

    mock_agent = SQLAgent(checkpointer=None)

    mock_assistant = SQLAssistant(sql_agent=mock_agent)

    return mock_assistant

def test_invoke(assistant: SQLAssistant):
    response = assistant.invoke("mock question")

    expected_response = SQLAssistantMessage(
        content="mock final answer",
        sql_queries=None,
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

def test_invoke_with_config(assistant: SQLAssistant):
    run_id = str(uuid.uuid4())

    response = assistant.invoke("mock question", {"run_id": run_id})

    expected_response = SQLAssistantMessage(
        id=run_id,
        content="mock final answer",
        sql_queries=None,
    )

    assert response.id == expected_response.id
    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries

def test_stream(assistant: SQLAssistant):
    for chunk in assistant.stream("mock question"):
        assert isinstance(chunk, dict)

    final_state = chunk

    for k in SQLAgentState.__required_keys__:
        assert k in final_state.keys()

def test_clear_thread(assistant: SQLAssistant):
    assistant.clear_thread(thread_id=str(uuid.uuid4()))
