import uuid

from pydantic import BaseModel, Field

from chatbot.agents import Chart
from chatbot.models import ModelURI


class UserMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    thread_id: str

class SQLAssistantMessage(UserMessage):
    model_uri: str
    sql_queries: list[str] | None = Field(default=None)

class SQLVizAssistantMessage(SQLAssistantMessage):
    chart: Chart | None = Field(default=None)
