import uuid

import pytest

from chatbot.agents import SQLAgent
from chatbot.agents.reducers import Item
from chatbot.assistants import SQLAssistant, SQLAssistantMessage, UserMessage

MODEL_URI = "openai/gpt-4o-mini"

@pytest.fixture
def assistant(monkeypatch):
    """Mock SQLAssistant"""
    def mock_agent_init(self):
        ...

    def mock_invoke(self, question, config):
        return {
            "question": "mock question",
            "final_answer": "mock final answer",
            "sql_queries": [],
            "sql_queries_results": [],
            "similar_examples": [],
            "messages": [],
            "is_last_step": False
        }

    def mock_assistant_init(self, model_uri, sql_agent):
        self.model_uri = model_uri
        self.sql_agent = sql_agent

    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(SQLAgent, "invoke", mock_invoke)
    monkeypatch.setattr(SQLAssistant, "__init__", mock_assistant_init)

    mock_agent = SQLAgent()

    mock_assistant = SQLAssistant(
        model_uri=MODEL_URI,
        sql_agent=mock_agent,
    )

    return mock_assistant

@pytest.fixture
def user_message() -> UserMessage:
    return UserMessage(content="mock question")

def test_format_response(assistant: SQLAssistant):
    response = {
        "final_answer": "hello world!",
        "sql_queries": [
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ]
    }

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_special_characters(assistant: SQLAssistant):
    response = {
        "final_answer": "\n\\xc4\\xa7&\\xc5\\x82\\xc5\\x82\\xc3\\xb8 w\\xc3\\xb8\\xc2\\xae\\xc5\\x82\\xc3\\xb0!\n",
        "sql_queries": [
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ],
    }

    expected_formatted_response = {
        "content": "ħ&łłø wø®łð!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_no_sql_queries(assistant: SQLAssistant):
    response = {
        "final_answer": "hello world!",
        "sql_queries": [],
    }

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": None,
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_invoke(assistant: SQLAssistant, user_message: UserMessage):
    response = assistant.invoke(user_message)

    expected_response = SQLAssistantMessage(
        content="mock final answer",
        model_uri=MODEL_URI,
        sql_queries=None,
    )

    assert response.content == expected_response.content
    assert response.model_uri == expected_response.model_uri
    assert response.sql_queries == expected_response.sql_queries

def test_clear_thread(assistant: SQLAssistant):
    assistant.clear_thread(thread_id=str(uuid.uuid4()))
