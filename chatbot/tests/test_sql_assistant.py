import uuid

import pytest
from loguru import logger

from chatbot.agents import SQLAgent
from chatbot.agents.reducers import Item
from chatbot.assistants import SQLAssistant, SQLAssistantAnswer,UserQuestion
from chatbot.models import ModelURI

MODEL_URI = ModelURI.gpt_4o_mini

@pytest.fixture
def assistant(monkeypatch):
    """Mocks SQLAssistant, as it makes calls to external APIs and logs activities"""
    def mock_logger_info(self):
        ...

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

    async def mock_ainvoke(self, question, config):
        return {
            "question": "mock question",
            "final_answer": "mock final answer",
            "sql_queries": [],
            "sql_queries_results": [],
            "similar_examples": [],
            "messages": [],
            "is_last_step": False
        }

    def mock_assistant_init(self, model_uri, sql_agent, logger):
        self.model_uri = model_uri
        self.sql_agent = sql_agent
        self.logger = logger

    monkeypatch.setattr(logger, "info", mock_logger_info)
    monkeypatch.setattr(SQLAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(SQLAgent, "invoke", mock_invoke)
    monkeypatch.setattr(SQLAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(SQLAssistant, "__init__", mock_assistant_init)

    mock_agent = SQLAgent()

    mock_assistant = SQLAssistant(
        model_uri=MODEL_URI,
        sql_agent=mock_agent,
        logger=logger
    )

    return mock_assistant

@pytest.fixture
def user_question() -> UserQuestion:
    return UserQuestion(
        id=str(uuid.uuid4()),
        question="mock question"
    )

def test_format_response(assistant: SQLAssistant):
    response = {
        "final_answer": "hello world!",
        "sql_queries": [
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ]
    }

    expected_formatted_response = {
        "answer": "hello world!",
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
        "answer": "ħ&łłø wø®łð!",
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
        "answer": "hello world!",
        "sql_queries": None,
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_ask(assistant: SQLAssistant, user_question: UserQuestion):
    response = assistant.ask(user_question, str(uuid.uuid4()))

    expected_response = SQLAssistantAnswer(
        id=response.id,
        question_id=user_question.id,
        question=user_question.question,
        model_uri=MODEL_URI,
        answer="mock final answer",
        sql_queries=None,
    )

    assert response == expected_response

@pytest.mark.asyncio
async def test_aask(assistant: SQLAssistant, user_question: UserQuestion):
    response = await assistant.aask(user_question, str(uuid.uuid4()))

    expected_response = SQLAssistantAnswer(
        id=response.id,
        question_id=user_question.id,
        question=user_question.question,
        model_uri=MODEL_URI,
        answer="mock final answer",
        sql_queries=None,
    )

    assert response == expected_response
