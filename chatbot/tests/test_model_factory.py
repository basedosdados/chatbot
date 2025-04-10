import pytest
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from chatbot.models import ModelFactory


@pytest.fixture
def mock_google_cloud_project(monkeypatch):
    monkeypatch.setenv(
        "GOOGLE_CLOUD_PROJECT",
        "mockup_google_cloud_project"
    )

@pytest.fixture
def mock_openai_api_key(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "mockup_openai_api_key"
    )

def test_from_model_uri_google(mock_google_cloud_project):
    model = ModelFactory.from_model_uri(
        model_uri="google/gemini-1.5-flash-001"
    )
    assert isinstance(model, ChatVertexAI)

def test_from_model_uri_openai(mock_openai_api_key):
    model = ModelFactory.from_model_uri(
        model_uri="openai/gpt-4o-mini"
    )
    assert isinstance(model, ChatOpenAI)

def test_from_model_uri_invalid():
    with pytest.raises(ValueError):
        _ = ModelFactory.from_model_uri(
            model_uri="some/model",
        )
