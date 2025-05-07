import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI


class ModelFactory:

    @staticmethod
    def from_model_uri(model_uri: str) -> BaseChatModel:
        """
        Create a LangChain chat model instance from a model URI.

        The URI must follow the format `<provider>/<model_name>`, where:
          - `<provider>` is one of: `openai`, `google`.
          - `<model_name>` is the name of the model to use.

        Examples are `openai/gpt-4o` or `google/gemini-2.0-flash`.

        Args:
            model_uri (str): A string identifying the model provider and name.

        Raises:
            ValueError: If the URI format is invalid or the provider is unsupported.

        Returns:
            BaseChatModel: A LangChain chat model.
        """
        if re.fullmatch(r"^[^/]+/[^/]+$", model_uri) is None:
            raise ValueError(
                f"Invalid model URI format: '{model_uri}'. Expected format is '<provider>/<model_name>', "
                "e.g., 'openai/gpt-4o' or 'google/gemini-2.0-flash'."
            )

        model_provider, model_name = model_uri.split("/")

        match model_provider:
            case "google":
                return ChatVertexAI(
                    model=model_name,
                    temperature=0
                )
            case "openai":
                return ChatOpenAI(
                    model=model_name,
                    temperature=0
                )
            case _:
                raise ValueError(
                    f"Unsupported model provider: '{model_provider}'. Supported providers are 'google' and 'openai'."
                )
