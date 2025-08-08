from typing import Literal

from pydantic import BaseModel, Field


class Rephrase(BaseModel):
    original: str = Field(description="The original user question")
    rephrased: str = Field(description="The rephrased user question")

class VizScript(BaseModel):
    script: str | None = Field("A Python script for creating Plotly figures")
    reasoning: str = Field("Step-by-step reasoning behind the script")

class RewrittenQuery(BaseModel):
    rewritten: str = Field(description="The rewritten user query")

class InitialRouting(BaseModel):
    """Determines the initial agent to handle the user's query."""
    next: Literal["sql_agent", "viz_agent"] = Field(
        description="The agent selected to process the user's question.",
        default="sql_agent"
    )
    reasoning: str = Field(
        description="Brief reasoning of why the agent was called"
    )

class PostSQLRouting(BaseModel):
    """Determines the next step after data retrieval by the sql_agent."""
    next: Literal["viz_agent", "process_answers"] = Field(
        description="The next step based on the retrieved data.",
        default="viz_agent"
    )
    reasoning: str = Field(
        description="Explanation for choosing the next step after data retrieval."
    )
