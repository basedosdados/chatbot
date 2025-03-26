from pydantic import BaseModel, Field

from chatbot.agents import Chart
from chatbot.models import ModelURI


class UserQuestion(BaseModel):
    id: str
    question: str

class SQLAssistantAnswer(UserQuestion):
    model_uri: ModelURI
    answer: str
    sql_queries: list[str] | None = Field(default=None)

class BigQueryAssistantAnswer(SQLAssistantAnswer):
    chart: Chart | None = Field(default=None)
