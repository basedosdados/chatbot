import uuid

from pydantic import BaseModel, Field

from chatbot.agents import Chart
from chatbot.models import ModelURI


class UserQuestion(BaseModel):
    id: str
    question: str

class SQLAssistantAnswer(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    question: str
    model_uri: ModelURI
    answer: str
    sql_queries: list[str] | None = Field(default=None)

class SQLVizAssistantAnswer(SQLAssistantAnswer):
    chart: Chart | None = Field(default=None)
