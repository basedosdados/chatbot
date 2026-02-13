from functools import cached_property
from typing import Annotated, Literal
from urllib.parse import quote

from google.oauth2.service_account import Credentials
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

NonEmptyStr = Annotated[str, Field(min_length=1)]


class Settings(BaseSettings):
    API_PREFIX: Literal["/api/v1"] = "/api/v1"

    # ============================================================
    # ==                  Environment settings                  ==
    # ============================================================
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="production",
        description="The environment the application is running in.",
    )

    # ============================================================
    # ==                     Auth Dev Mode                      ==
    # ============================================================
    AUTH_DEV_MODE: bool = Field(
        default=False,
        description=(
            "When enabled, bypasses JWT validation and returns AUTH_DEV_USER_ID for all requests. "
            "Only works when ENVIRONMENT is set to 'development'. "
            "WARNING: Must NEVER be enabled in production."
        ),
    )
    AUTH_DEV_USER_ID: int = Field(
        default=1, description="The user ID to return when AUTH_DEV_MODE is enabled."
    )

    # ============================================================
    # ==                   Database settings                    ==
    # ============================================================
    DB_HOST: NonEmptyStr = Field(description="PostgreSQL database host.")
    DB_PORT: int = Field(description="PostgreSQL database port.")
    DB_NAME: NonEmptyStr = Field(description="PostgreSQL database name.")
    DB_USER: NonEmptyStr = Field(description="PostgreSQL database user.")
    DB_PASSWORD: NonEmptyStr = Field(description="PostgreSQL database password.")

    @computed_field
    @property
    def DB_URL(self) -> str:  # pragma: no cover
        """PostgreSQL database URL."""
        user = quote(self.DB_USER, safe="")
        password = quote(self.DB_PASSWORD, safe="")
        return f"postgresql://{user}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def SQLALCHEMY_DB_URL(self) -> str:  # pragma: no cover
        """PostgreSQL database URL for SQLAlchemy."""
        user = quote(self.DB_USER, safe="")
        password = quote(self.DB_PASSWORD, safe="")
        return f"postgresql+psycopg://{user}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # ============================================================
    # ==                Website Backend settings                ==
    # ============================================================
    BASEDOSDADOS_BASE_URL: str = Field(
        default="https://backend.basedosdados.org",
        description="Base URL for the basedados backend.",
    )
    JWT_ALGORITHM: NonEmptyStr = Field(
        description="Algorithm used for signing and and verifying JWT tokens."
    )
    JWT_SECRET_KEY: NonEmptyStr = Field(
        description="Secret key used for signing and verifying JWT tokens. Keep it secret. Keep it safe."
    )

    # ============================================================
    # ==                 Google Cloud settings                  ==
    # ============================================================
    GOOGLE_BIGQUERY_PROJECT: NonEmptyStr = Field(
        description="Google BigQuery project ID."
    )
    GOOGLE_SERVICE_ACCOUNT: NonEmptyStr = Field(
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
    MODEL_URI: NonEmptyStr = Field(
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
    LANGSMITH_API_KEY: NonEmptyStr = Field(description="LangSmith API key.")
    LANGSMITH_PROJECT: NonEmptyStr = Field(description="LangSmith project name.")

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
