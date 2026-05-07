from pydantic import BaseModel, Field


class Column(BaseModel):
    """Complete column information."""

    name: str
    type: str
    description: str | None
    unit: str | None = Field(exclude_if=lambda v: v is None)
    reference_table_id: str | None = Field(exclude_if=lambda v: v is None)
    needs_decoding: bool


class TableOverview(BaseModel):
    """Basic table information without column details."""

    id: str
    gcp_id: str | None
    name: str
    description: str | None


class Table(TableOverview):
    """Complete table information including all its columns."""

    columns: list[Column]
    partitioned_by: list[str]
    period_start: str | None
    period_end: str | None


class DatasetOverview(BaseModel):
    """Basic dataset information without table details."""

    id: str
    name: str
    description: str | None
    organizations: list[str]
    tags: list[str]
    themes: list[str]


class Dataset(DatasetOverview):
    """Complete dataset information including all tables and columns."""

    tables: list[TableOverview]
    usage_guide: str | None
