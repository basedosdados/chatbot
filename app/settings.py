from functools import cached_property
from typing import Literal

from google.oauth2.service_account import Credentials
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_PREFIX: Literal["/api/v1"] = "/api/v1"

    # ============================================================
    # ==                   Database settings                    ==
    # ============================================================
    DB_HOST: str = Field(description="PostgreSQL database host.")
    DB_PORT: int = Field(description="PostgreSQL database port.")
    DB_NAME: str = Field(description="PostgreSQL database name.")
    DB_USER: str = Field(description="PostgreSQL database user.")
    DB_PASSWORD: str = Field(description="PostgreSQL database password.")
    DB_SCHEMA_CHATBOT: str = Field(description="PostgreSQL chatbot database schema.")

    @computed_field
    @property
    def DB_URL(self) -> str:  # pragma: no cover
        """PostgreSQL database URL."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def SQLALCHEMY_DB_URL(self) -> str:  # pragma: no cover
        """PostgreSQL database URL for SQLAlchemy."""
        return f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # ============================================================
    # ==                Website Backend settings                ==
    # ============================================================
    BASEDOSDADOS_BASE_URL: str = Field(
        default="https://backend.basedosdados.org",
        description="Base URL for the basedados backend.",
    )
    JWT_ALGORITHM: str = Field(
        description="Algorithm used for signing and and verifying JWT tokens."
    )
    JWT_SECRET_KEY: str = Field(
        description="Secret key used for signing and verifying JWT tokens. Keep it secret. Keep it safe."
    )

    # ============================================================
    # ==                 Google Cloud settings                  ==
    # ============================================================
    GOOGLE_BIGQUERY_PROJECT: str = Field(description="Google BigQuery project ID.")
    GOOGLE_SERVICE_ACCOUNT: str = Field(
        description="Path to a google service account with required permissions."
    )

    @computed_field
    @cached_property
    def GOOGLE_CREDENTIALS(self) -> Credentials:  # pragma: no cover
        """Google Cloud credentials."""
        return Credentials.from_service_account_file(
            filename=self.GOOGLE_SERVICE_ACCOUNT,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    # ============================================================
    # ==                      LLM settings                      ==
    # ============================================================
    MODEL_URI: str = Field(
        description=(
            "Defines the LLM to be used. Refer to the LangChain docs for valid values: "
            "https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model."
        )
    )
    MODEL_TEMPERATURE: float = Field(
        description=(
            "Controls the randomness of the model’s output. "
            "A higher number makes responses more creative; "
            "lower ones make them more deterministic."
        )
    )
    MAX_TOKENS: int = Field(
        description=(
            "Maximum number of tokens for a single input. "
            "Must be defined according to the provided model uri "
            "and be less than the model's maximum input tokens."
        ),
    )

    # ============================================================
    # ==                   LangSmith settings                   ==
    # ============================================================
    LANGSMITH_TRACING: bool = Field(
        default=True, description="Whether to enable tracing to LangSmith."
    )
    LANGSMITH_API_KEY: str = Field(description="LangSmith API key.")
    LANGSMITH_PROJECT: str = Field(description="LangSmith project name.")

    # ============================================================
    # ==                    Logging settings                    ==
    # ============================================================
    LOG_LEVEL: str = Field(
        default="INFO", description="The minimum severity level for logging messages."
    )
    LOG_BACKTRACE: bool = Field(
        default=True,
        description="Whether the full stacktrace should be displayed when an exception occur.",
    )
    LOG_DIAGNOSE: bool = Field(
        default=False,
        description=(
            "Whether the exception trace should display the variables values to eases the debugging. "
            "This should be set to False in production to avoid leaking sensitive data."
        ),
    )
    LOG_ENQUEUE: bool = Field(
        default=False,
        description=(
            "Whether the messages to be logged should first pass through a multiprocessing-safe queue before reaching the sink. "
            "This is useful while logging to a file through multiple processes and also has the advantage of making logging calls non-blocking."
        ),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)


settings = Settings()
