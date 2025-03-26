from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from .datatypes import ModelURI


class ModelFactory:

    @staticmethod
    def from_model_uri(model_uri: ModelURI) -> BaseChatModel:
        """Get a LangChain Chat Model from a model uri

        Args:
            model_uri (ModelURI): The model uri

        Raises:
            ValueError: If the model uri is not supported

        Returns:
            BaseChatModel: LangChain Chat Model
        """
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
                values = ", ".join([repr(v) for v in ModelURI.values()])
                raise ValueError(f"Model URI should be one of the following: {values}")
