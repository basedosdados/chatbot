from typing import Any, Literal

from pydantic import BaseModel, Field


class InitialRouting(BaseModel):
    """Determines the initial agent to handle the user's query."""
    agent: Literal["sql_agent", "viz_agent"] = Field(description="The agent selected to handle the user's question")
    reasoning: str = Field(
        description="Brief reasoning for the routing decision, explaining what the user wants and why this agent was chosen"
    )
    data_turn_ids: list[int] | None = Field(
        default=None,
        description="A list of all turn numbers from the chat history that contains the relevant data when routing to the viz_agent"
    )
    question_for_viz_agent: str | None = Field(
        default=None,
        description="A rephrased, self-contained question for the viz_agent, including all necessary context from the conversation"
    )

class PostSQLRouting(BaseModel):
    """Determines the next action after data retrieval by the sql_agent."""
    action: Literal["trigger_visualization", "skip_visualization"] = Field(description="The next action to take")
    reasoning: str = Field(
        description="Brief reasoning for the decision, explaining why this action was chosen"
    )
    question_for_viz_agent: str | None = Field(
        default=None,
        description="A rephrased, self-contained question for the viz_agent, including all necessary context from the conversation"
    )

class RewrittenQuery(BaseModel):
    rewritten: str = Field(description="The rewritten user query")

class VizScript(BaseModel):
    script: str = Field(description="A complete and executable Python script that generates a Plotly figure")
    reasoning: str = Field(description="A detailed, step-by-step explanation of the choices made during the script's creation. This includes the choice of visualization, data transformations, and any calculations performed")
    insights: str = Field(description="A concise, business-friendly paragraph that describes the generated visualization and its key takeaways, highlighting the most important trends or patterns in the data")

class Visualization(VizScript):
    data: list[dict[str, Any]]
    data_placeholder: Literal["INPUT_DATA"] = "INPUT_DATA"
