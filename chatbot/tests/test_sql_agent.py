import uuid

import pytest
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, ToolMessage)

from chatbot.agents import SQLAgent
from chatbot.agents.reducers import Item, ItemRemove


@pytest.fixture
def agent(monkeypatch) -> SQLAgent:
    def mock_agent_init(self, question_limit: int = 5):
        self.question_limit = question_limit

    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)

    return SQLAgent()

@pytest.fixture()
def messages() -> list[BaseMessage]:
    return [
        HumanMessage(""),
        AIMessage(""),
        ToolMessage("", tool_call_id=str(uuid.uuid4())),
        AIMessage("")
    ]

def test_prune_messages_not_delete(
    agent: SQLAgent,
    messages: list[BaseMessage]
):
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected

def test_prune_messages_delete(
    agent: SQLAgent,
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
    agent: SQLAgent,
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
    agent: SQLAgent,
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
    agent: SQLAgent,
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
    agent: SQLAgent,
    messages: list[BaseMessage]
):
    # setting question limit to 1
    # means all messages must be deleted
    agent.question_limit = None
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected

def test_clear_sql_empty(agent: SQLAgent):
    response = agent._clear_sql({
        "final_answer": "",
        "sql_queries": [],
        "sql_queries_results": []
    })

    expected = {
        "final_answer": "",
        "sql_queries": [],
        "sql_queries_results": []
    }

    assert response == expected

def test_clear_sql(agent: SQLAgent):
    state = {
        "final_answer": "mock sql answer",
        "sql_queries": [Item(content=i) for i in range(5)],
        "sql_queries_results": [Item(content=i) for i in range(5)]
    }

    response = agent._clear_sql(state)

    expected = {
        "final_answer": "",
        "sql_queries": [
            ItemRemove(id=item.id) for item in state["sql_queries"]
        ],
        "sql_queries_results": [
            ItemRemove(id=item.id) for item in state["sql_queries_results"]
        ]
    }

    assert response == expected
