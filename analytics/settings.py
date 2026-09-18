"""Env-driven config for the two databases and the GCP project.

Separate from `app.settings` (which requires the full app config). Per DB, set
CHATBOT_DB_* / BACKEND_DB_* (HOST, PORT, NAME, USER, PASSWORD).

A `.env.analytics` file in the working directory also works.
"""

from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# The BigQuery dataset lives in one project, which also bills the load jobs. A
# single, constrained value keeps the workload's IAM grants on that one project.
GCPProject = Literal["basedosdados-dev"]

# A dedicated env file (resolved against the CWD, i.e. the repo root where you run
# `python -m analytics.cli`) so this tool's config never conflates with the app's
# own .env. Matches the `.env.*` gitignore rule, so it is not committed.
ENV_FILE = ".env.analytics"


class DatabaseSettings(BaseSettings):
    """Connection settings for one PostgreSQL database.

    Subclassed with an env_prefix per database.
    """

    host: str
    port: int
    name: str
    user: str
    # min_length rejects an empty <PREFIX>_PASSWORD= line (easy to leave behind when
    # commenting passwords in and out of .env.analytics) up front, with the variable
    # named, instead of failing later as an opaque authentication error.
    password: SecretStr = Field(min_length=1)

    @computed_field
    @property
    def url(self) -> URL:
        """SQLAlchemy URL for this database."""
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.name,
        )


class ChatbotDatabaseSettings(DatabaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_DB_", env_file=ENV_FILE, extra="ignore"
    )


class BackendDatabaseSettings(DatabaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_DB_", env_file=ENV_FILE, extra="ignore"
    )


class Settings(BaseSettings):
    """Top-level export config: the two databases plus the GCP project.

    `gcp_project` (GCP_PROJECT) holds the target BigQuery dataset and bills the load
    jobs. Credentials are Application Default Credentials, not a setting (see
    GOOGLE_APPLICATION_CREDENTIALS for a key file, or workload identity in GKE).
    """

    chatbot_db: ChatbotDatabaseSettings = Field(default_factory=ChatbotDatabaseSettings)
    backend_db: BackendDatabaseSettings = Field(default_factory=BackendDatabaseSettings)
    gcp_project: GCPProject = "basedosdados-dev"

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")
