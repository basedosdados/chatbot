from pydantic import BaseModel, Field


class Column(BaseModel):
    """Complete column information."""

    name: str
    type: str
    description: str | None
    unit: str | None = Field(exclude_if=lambda v: v is None)
    reference_table_id: str | None = Field(exclude_if=lambda v: v is None)


class TableOverview(BaseModel):
    """Basic table information without column details."""

    id: str
    gcp_id: str | None
    name: str
    slug: str | None
    description: str | None
    temporal_coverage: dict[str, str | None]


class Table(TableOverview):
    """Complete table information including all its columns."""

    columns: list[Column]


class DatasetOverview(BaseModel):
    """Basic dataset information without table details."""

    id: str
    name: str
    slug: str | None
    description: str | None
    tags: list[str]
    themes: list[str]
    organizations: list[str]


class Dataset(DatasetOverview):
    """Complete dataset information including all tables and columns."""

    tables: list[TableOverview]
    usage_guide: str | None
