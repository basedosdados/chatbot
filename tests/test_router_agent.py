import pytest
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage)

from chatbot.agents.router_agent import RouterAgent
from chatbot.agents.structured_outputs import Chart, ChartData, ChartMetadata


@pytest.fixture
def agent(monkeypatch) -> RouterAgent:
    def mock_agent_init(self, question_limit: int = 5):
        self.question_limit = question_limit

    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)

    return RouterAgent()

@pytest.fixture()
def messages() -> list[BaseMessage]:
    return [
        HumanMessage(""),
        AIMessage(""),
    ]

@pytest.fixture()
def valid_chart() -> Chart:
    return Chart(
        data=ChartData(),
        metadata=ChartMetadata(),
        is_valid=True
    )

@pytest.fixture()
def invalid_chart() -> Chart:
    return Chart(
        data=ChartData(),
        metadata=ChartMetadata(),
        is_valid=False
    )

def test_prune_messages_not_delete(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected

def test_prune_messages_delete(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    # the message list is being repeated "question_limit" times
    # so we just need to delete the first repetition
    idx_messages = len(messages)

    messages = messages*agent.question_limit

    response = agent._prune_messages({"messages": messages})

    expected = {
        "messages": [
            RemoveMessage(msg.id) for msg in messages[:idx_messages]
        ]
    }

    assert response == expected

def test_prune_messages_delete_last_is_human(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    # the question limit is set to 2 to ensure that
    # the last message is a human message when the limit is reached.
    agent.question_limit = 2

    messages += [HumanMessage("")]

    response = agent._prune_messages({"messages": messages})

    expected = {
        "messages": [RemoveMessage(msg.id) for msg in messages]
    }

    assert response == expected

def test_prune_messages_delete_consecutive_human(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    # the message list is being repeated "question_limit" times
    # so we just need to delete the first repetition + the next human message
    idx_messages = len(messages) + 1

    messages = messages + [HumanMessage("")] + messages*(agent.question_limit-2)

    response = agent._prune_messages({"messages": messages})

    expected = {
        "messages": [
            RemoveMessage(msg.id) for msg in messages[:idx_messages]
        ]
    }

    assert response == expected

def test_prune_messages_question_limit_1(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    # setting question limit to 1
    # means all messages must be deleted
    agent.question_limit = 1

    response = agent._prune_messages({"messages": messages})

    expected = {
        "messages": [RemoveMessage(msg.id) for msg in messages]
    }

    assert response == expected

def test_prune_messages_question_limit_none(
    agent: RouterAgent,
    messages: list[BaseMessage]
):
    # setting question limit to 1
    # means all messages must be deleted
    agent.question_limit = None
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected

def test_build_final_answer_previous_sql_valid_chart(agent: RouterAgent, valid_chart: Chart):
    state = {
        "_previous": "sql_agent",
        "_next": "viz_agent",
        "sql_answer": "mock sql answer",
        "chart_answer": "mock_chart_answer",
        "chart": valid_chart
    }

    response = agent._process_answers(state)

    expected = {
        "final_answer": f"{state['sql_answer']}\n\n{state['chart_answer']}"
    }

    assert response == expected

def test_build_final_answer_previous_sql_invalid_chart(agent: RouterAgent, invalid_chart: Chart):
    state = {
        "_previous": "sql_agent",
        "_next": "viz_agent",
        "sql_answer": "mock sql answer",
        "chart_answer": "mock_chart_answer",
        "chart": invalid_chart
    }

    response = agent._process_answers(state)

    expected = {"final_answer": state["sql_answer"]}

    assert response == expected

def test_build_final_answer_previous_sql_next_process_answers(agent: RouterAgent, invalid_chart: Chart):
    state = {
        "_previous": "sql_agent",
        "_next": "process_answers",
        "sql_answer": "mock sql answer",
        "chart_answer": "mock_chart_answer",
        "chart": valid_chart
    }

    response = agent._process_answers(state)

    expected = {
        "chart": invalid_chart,
        "final_answer": state['sql_answer']
    }

    assert response == expected

def test_build_final_answer_previous_initial_router_valid_chart(agent: RouterAgent, valid_chart: Chart):
    state = {
        "_previous": "initial_router",
        "_next": "viz_agent",
        "sql_answer": "mock sql answer",
        "chart_answer": "mock_chart_answer",
        "chart": valid_chart
    }

    response = agent._process_answers(state)

    expected = {"final_answer": state["chart_answer"]}

    assert response == expected

def test_build_final_answer_previous_initial_router_invalid_chart(agent: RouterAgent, invalid_chart: Chart):
    state = {
        "_previous": "initial_router",
        "_next": "viz_agent",
        "sql_answer": "mock sql answer",
        "chart_answer": "mock_chart_answer",
        "chart": invalid_chart
    }

    response = agent._process_answers(state)

    expected = {"final_answer": state["chart_answer"]}

    assert response == expected
