import uuid

import pytest
import pytest_asyncio

from chatbot.agents import RouterAgent
from chatbot.agents.reducers import Item
from chatbot.agents.structured_outputs import Chart, ChartData, ChartMetadata
from chatbot.assistants import (AsyncSQLVizAssistant, SQLVizAssistantMessage,
                                UserMessage)

MODEL_URI = "openai/gpt-4o-mini"

@pytest_asyncio.fixture
async def assistant(monkeypatch):
    """Mock AsyncSQLVizAssistant"""
    def mock_agent_init(self):
        ...

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

    def mock_assistant_init(self, model_uri, router_agent):
        self.model_uri = model_uri
        self.router_agent = router_agent

    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)
    monkeypatch.setattr(RouterAgent, "ainvoke", mock_ainvoke)
    monkeypatch.setattr(AsyncSQLVizAssistant, "__init__", mock_assistant_init)

    mock_agent = RouterAgent()

    mock_assistant = AsyncSQLVizAssistant(
        model_uri=MODEL_URI,
        router_agent=mock_agent,
    )

    return mock_assistant

@pytest.fixture
def user_message() -> UserMessage:
    return UserMessage(content="mock question")

def test_format_response(assistant: AsyncSQLVizAssistant):
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
        "content": "hello world!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_special_characters(assistant: AsyncSQLVizAssistant):
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
        "content": "ħ&łłø wø®łð!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

def test_format_response_with_no_sql_queries(assistant: AsyncSQLVizAssistant):
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
        "content": "hello world!",
        "sql_queries": None,
        "chart": Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    }

    formatted_response = assistant._format_response(response)

    assert formatted_response == expected_formatted_response

@pytest.mark.asyncio
async def test_invoke(assistant: AsyncSQLVizAssistant, user_message: UserMessage):
    response = await assistant.invoke(user_message)

    expected_response = SQLVizAssistantMessage(
        content="mock final answer",
        model_uri=MODEL_URI,
        sql_queries=None,
        chart=Chart(
            data=ChartData(),
            metadata=ChartMetadata(),
            is_valid=False
        ),
    )

    assert response.content == expected_response.content
    assert response.model_uri == expected_response.model_uri
    assert response.sql_queries == expected_response.sql_queries
    assert response.chart == expected_response.chart

@pytest.mark.asyncio
async def test_clear_thread(assistant: AsyncSQLVizAssistant):
    await assistant.clear_thread(thread_id=str(uuid.uuid4()))
