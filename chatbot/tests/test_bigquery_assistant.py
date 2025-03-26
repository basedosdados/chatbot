import uuid

import pytest
from loguru import logger

from chatbot.agents import RouterAgent
from chatbot.agents.reducers import Item
from chatbot.agents.structured_outputs import Chart, ChartData, ChartMetadata
from chatbot.assistants import (BigQueryAssistant, BigQueryAssistantAnswer,
                                UserQuestion)
from chatbot.models import ModelURI

MODEL_URI = ModelURI.gpt_4o_mini

@pytest.fixture
def assistant(monkeypatch):
    """Mocks BigQueryAssistant, as it makes calls to external APIs and logs activities"""
    def mock_logger_info(self):
        ...

    def mock_agent_init(self):
        ...

    def mock_invoke(self, question, config):
        chart_data = ChartData()
        chart_metadata = ChartMetadata()
        chart = Chart(
            data=chart_data,
            metadata=chart_metadata,
            is_valid=False
        )
        return {
            "next": "mock next",
            "question": "mock question",
            "sql_answer": "mock sql answer",
            "chart_answer": "mock chart answer",
            "final_answer": "mock final answer",
            "sql_queries": [],
            "sql_queries_results": [],
            "chart": chart,
            "messages": [],
        }

    async def mock_ainvoke(self, question, config):
        chart_data = ChartData()
        chart_metadata = ChartMetadata()
        chart = Chart(
            data=chart_data,
            metadata=chart_metadata,
            is_valid=False
        )
        return {
            "next": "mock next",
            "question": "mock question",
            "sql_answer": "mock sql answer",
            "chart_answer": "mock chart answer",
            "final_answer": "mock final answer",
            "sql_queries": [],
            "sql_queries_results": [],
            "chart": chart,
            "messages": [],
        }

    def mock_assistant_init(self, model_uri, router_agent, logger):
        self.model_uri = model_uri
        self.router_agent = router_agent
        self.logger = logger

    monkeypatch.setattr(logger, "info", mock_logger_info)
    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(RouterAgent, "invoke", mock_invoke)
    monkeypatch.setattr(RouterAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(BigQueryAssistant, "__init__", mock_assistant_init)

    mock_agent = RouterAgent()

    mock_assistant = BigQueryAssistant(
        model_uri=MODEL_URI,
        router_agent=mock_agent,
        logger=logger
    )

    return mock_assistant

@pytest.fixture
def user_question() -> UserQuestion:
    return UserQuestion(
        id=str(uuid.uuid4()),
        question="mock question"
    )

def test_format_response(assistant: BigQueryAssistant):
    response = {
        "final_answer": "hello world!",
        "sql_queries": [
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    expected_formatted_response = {
        "answer": "hello world!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_special_characters(assistant: BigQueryAssistant):
    response = {
        "final_answer": "\n\\xc4\\xa7&\\xc5\\x82\\xc5\\x82\\xc3\\xb8 w\\xc3\\xb8\\xc2\\xae\\xc5\\x82\\xc3\\xb0!\n",
        "sql_queries": [
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    expected_formatted_response = {
        "answer": "ħ&łłø wø®łð!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_no_sql_queries(assistant: BigQueryAssistant):
    response = {
        "final_answer": "hello world!",
        "sql_queries": [],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    expected_formatted_response = {
        "answer": "hello world!",
        "sql_queries": None,
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_ask(assistant: BigQueryAssistant, user_question: UserQuestion):
    response = assistant.ask(user_question, str(uuid.uuid4()))

    expected_response = user_question.model_dump()
    expected_response.update({
        "model_uri": MODEL_URI,
        "answer": "mock final answer",
        "sql_queries": None,
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    })
    expected_response = BigQueryAssistantAnswer(**expected_response)

    assert response == expected_response

@pytest.mark.asyncio
async def test_aask(assistant: BigQueryAssistant, user_question: UserQuestion):
    response = await assistant.aask(user_question, str(uuid.uuid4()))

    expected_response = user_question.model_dump()
    expected_response.update({
        "model_uri": MODEL_URI,
        "answer": "mock final answer",
        "sql_queries": None,
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    })
    expected_response = BigQueryAssistantAnswer(**expected_response)

    assert response == expected_response
