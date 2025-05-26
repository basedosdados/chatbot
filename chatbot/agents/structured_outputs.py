from typing import Any, Literal

from pydantic import BaseModel, Field, Json, SerializeAsAny


class ChartData(BaseModel):
    data: Json[list[dict[str, Any]]] | None = Field(description="Preprocessed data", default=None)
    reasoning: str | None = Field(description="Brief explanation of how the query results were processed, including any assumptions made, conflicts resolved, or placeholders used", default=None)

class ChartMetadata(BaseModel):
    chart_type: Literal["bar", "horizontal_bar", "line", "pie", "scatter"] | None = Field(description="Name of the chart type", default=None)
    title: str | None = Field(description="Title of the chart", default=None)
    x_axis: str | None = Field(description="Name of the variable to be plotted on the x-axis", default=None)
    x_axis_title: str | None = Field(description="Human friendly label for the x-axis", default=None)
    y_axis: str | None = Field(description="Name of the variable to be plotted on the y-axis", default=None)
    y_axis_title: str | None = Field(description="Human friendly label for the y-axis", default=None)
    label: str | None = Field(description="Name of the variable to be used as label", default=None)
    label_title: str | None = Field(description="Human friendly name for the label", default=None)
    reasoning: str | None = Field(description="Brief explanation for your recommendation", default=None)

# LangGraph seems to convert nested Pydantic models to dictionaries during state management when a checkpointer is used. As a result, an annoying warning is displayed:
#
# UserWarning: Pydantic serializer warnings: Expected ... but got `dict` with value ...
#
# To suppress the warning, I'm using the SerializeAsAny annotation. When a field is annotated as SerializeAsAny[<SomeType>], it retains the same validation behavior as if it was annotated as <SomeType>, and type checkers will treat the attribute as having the appropriate type as well. However, during serialization, the field is treated as if its type hint was Any, hence the name. For more details, see: https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
class Chart(BaseModel):
    data: SerializeAsAny[ChartData]
    metadata: SerializeAsAny[ChartMetadata]
    is_valid: bool

class Rephrase(BaseModel):
    original: str = Field(description="The original user question")
    rephrased: str = Field(description="The rephrased user question")

class Route(BaseModel):
    next: Literal["sql_agent", "viz_agent"] = Field(description="The worker name", default="sql_agent")
    reasoning: str = Field(description="Brief reasoning of why the worker was called")

class Route2(BaseModel):
    next: Literal["viz_agent", "process_answers"] = Field(description="The next step", default="viz_agent")
    reasoning: str = Field(description="Brief reasoning of why the worker was called")
