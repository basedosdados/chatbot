from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field, Json, SerializeAsAny, model_validator


class ChartData(BaseModel):
    data: Json[Any] | None = Field(description="Preprocessed data", default=None)
    reasoning: str | None = Field(description="Brief explanation of how the query results were processed, including any assumptions made, conflicts resolved, or placeholders used", default=None)

    @model_validator(mode="after")
    def _merge_data(self) -> "ChartData":
        """Transforms a list of dictionaries into a dictionary of lists, padding with None values
        when the lists have different lengths

        Returns:
            ChartData: The updated ChartData object

        Examples:
            >>> list_of_dicts = [
            ...     {"k1": "v1", "k2": "v2"}
            ...     {"k1": "v3"}
            ... ]
            >>> _merge_data(list_of_dicts)
            {"k1": ["v1", "v3"], "k2": ["v2", None]}
        """
        if self.data is None:
            return self

        # the validator is also called when creating the Chart object, so this check is necessary
        # this is a pydantic bug, as stated in https://github.com/pydantic/pydantic/issues/8452
        if isinstance(self.data, dict):
            return self

        all_columns = set()

        for row in self.data:
            all_columns.update(row.keys())

        merged = defaultdict(list)

        for row in self.data:
            for column in all_columns:
                merged[column].append(row.get(column))

        # ensures we have the same number of data points for each variable
        assert len({len(l) for l in merged.values()}) == 1

        self.data = dict(merged)

        return self

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
