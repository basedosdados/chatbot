import uuid

import pytest

from chatbot.agents.router_agent import RouterAgent, RouterAgentState
from chatbot.assistants import SQLVizAssistant, SQLVizAssistantMessage


@pytest.fixture
def assistant(monkeypatch):
    """Mock SQLVizAssistant"""

    def mock_agent_init(self, checkpointer):
        self.checkpointer = checkpointer

    def mock_invoke(self, question, config, rewrite_query):
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

    def mock_stream(self, question, config, stream_mode, subgraphs, rewrite_query):
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
    monkeypatch.setattr(RouterAgent, "invoke", mock_invoke)
    monkeypatch.setattr(RouterAgent, "stream", mock_stream)
    monkeypatch.setattr(SQLVizAssistant, "__init__", mock_assistant_init)

    mock_agent = RouterAgent(checkpointer=None)

    mock_assistant = SQLVizAssistant(router_agent=mock_agent)

    return mock_assistant

def test_invoke(assistant: SQLVizAssistant):
    response = assistant.invoke("mock question")

    expected_response = SQLVizAssistantMessage(
        content="mock final answer",
        sql_queries=None,
        visualization=None,
    )

    assert response.content == expected_response.content
    assert response.sql_queries == expected_response.sql_queries
    assert response.visualization == expected_response.visualization

def test_invoke_with_config(assistant: SQLVizAssistant):
    run_id = str(uuid.uuid4())

    response = assistant.invoke("mock question", {"run_id": run_id})

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

def test_stream(assistant: SQLVizAssistant):
    for chunk in assistant.stream("mock question"):
        assert isinstance(chunk, dict)

    final_state = chunk

    for k in RouterAgentState.__required_keys__:
        assert k in final_state.keys()

def test_clear_thread(assistant: SQLVizAssistant):
    assistant.clear_thread(thread_id=str(uuid.uuid4()))
