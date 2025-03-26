import pytest
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage)

from chatbot.agents import ReActAgent


@pytest.fixture
def agent(monkeypatch) -> ReActAgent:
    def mock_agent_init(self, question_limit: int = 5):
        self.question_limit = question_limit

    monkeypatch.setattr(ReActAgent, "__init__", mock_agent_init)

    return ReActAgent()

@pytest.fixture()
def messages() -> list[BaseMessage]:
    return [
        HumanMessage(""),
        AIMessage(""),
    ]

def test_prune_messages_not_delete(
    agent: ReActAgent,
    messages: list[BaseMessage]
):
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected

def test_prune_messages_delete(
    agent: ReActAgent,
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
    agent: ReActAgent,
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
    agent: ReActAgent,
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
    agent: ReActAgent,
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
    agent: ReActAgent,
    messages: list[BaseMessage]
):
    # setting question limit to 1
    # means all messages must be deleted
    agent.question_limit = None
    response = agent._prune_messages({"messages": messages})
    expected = {"messages": []}
    assert response == expected
