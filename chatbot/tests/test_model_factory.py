import pytest

from chatbot.models import ModelFactory
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI


def test_from_model_uri_google():
    model = ModelFactory.from_model_uri(
        model_uri="google/gemini-1.5-flash-001"
    )
    assert isinstance(model, ChatVertexAI)

def test_from_model_uri_openai():
    model = ModelFactory.from_model_uri(
        model_uri="openai/gpt-4o-mini"
    )
    assert isinstance(model, ChatOpenAI)

def test_from_model_uri_invalid():
    with pytest.raises(ValueError):
        _ = ModelFactory.from_model_uri(
            model_uri="some/model",
        )
