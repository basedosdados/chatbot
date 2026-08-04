from enum import Enum

from pydantic import BaseModel, Field


class TemporalGranularity(str, Enum):
    """Granularity of the data's temporal coverage."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class TemporalCoverage(BaseModel):
    """The interval the SQL query actually filtered on in the answer."""

    period_start: str = Field(
        description=(
            "Start of the interval filtered by the SQL query (e.g. '2010' for "
            "`ano = 2010` or `ano BETWEEN 2010 AND 2012`). Format it to match `granularity`: "
            "YYYY (year), YYYY-MM (month) or YYYY-MM-DD (day) — e.g. '2010', '2010-01', '2010-01-01'. "
            "May be narrower than the table's full coverage."
        )
    )
    period_end: str = Field(
        description=(
            "End of the interval filtered by the SQL query (e.g. '2010' for "
            "`ano = 2010`; '2012' for `ano BETWEEN 2010 AND 2012`). Format it to match `granularity`: "
            "YYYY (year), YYYY-MM (month) or YYYY-MM-DD (day) — e.g. '2012', '2012-01', '2012-01-01'. "
            "May be narrower than the table's full coverage."
        )
    )
    granularity: TemporalGranularity = Field(
        description=(
            "Granularity of `period_start`/`period_end`, matching their format: "
            "YYYY (year), YYYY-MM (month), YYYY-MM-DD (day)."
        )
    )


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
    temporal_coverage: TemporalCoverage | None = Field(
        default=None,
        description=(
            "The interval your SQL query actually filtered. Leave empty (None) when no query "
            "was run (e.g. a clarification turn) or the answer has no temporal dimension."
        ),
    )
    follow_up_questions: list[str] | None = Field(
        default=None,
        description=(
            "3 suggested follow-up questions (in the user's language) to explore the data further."
        ),
    )
