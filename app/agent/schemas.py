from pydantic import BaseModel, Field


class DataSource(BaseModel):
    """A Base dos Dados table the answer draws on or points the user to."""

    dataset_id: str = Field(
        description=(
            "Dataset UUID (the `dataset_id` field from `get_table_details` or the `id` "
            "field from `get_dataset_details`), not the BigQuery id (e.g. 'br_bd_diretorios')."
        )
    )
    table_id: str = Field(
        description=(
            "Table UUID (the `id` field from `get_table_details`, or a table's `id` from "
            "the tables list of `get_dataset_details`). Not the dataset UUID, not the BigQuery id."
        )
    )
    name: str = Field(description="Human-readable name of the table.")


class StructuredResponse(BaseModel):
    """The agent's structured response for the user interface."""

    response: str = Field(
        description=(
            "The prose answer in Markdown, written in the user's language: a direct answer to the question "
            "with the data obtained, plus analysis and context. Do NOT repeat the source, period, SQL, "
            "or suggestions here — each of those has its own dedicated field."
        )
    )
    data_sources: list[DataSource] | None = Field(
        default=None,
        description=(
            "The tables the answer draws on — those you queried, or specific tables you recommend "
            "on clarification turns. Leave empty (None) when no table is relevant."
        ),
    )
    follow_up_prompts: list[str] | None = Field(
        default=None,
        description=(
            "3 next prompts the user could send you to explore the data further, each written "
            "in the user's own voice — a message the user types to you, never a question you ask "
            "the user. In the user's language."
        ),
    )
