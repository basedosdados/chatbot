import json
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnMetadata:
    """One column's translation-relevant metadata."""

    name: str
    needs_decoding: bool
    reference_table_id: str | None  # a directory table UUID to JOIN, or None

    @property
    def is_coded(self) -> bool:
        """Whether the column holds opaque values that must be translated before use.

        True when the column carries a code dictionary (`needs_decoding`) or points
        to a directory table (`reference_table_id`) — the two cases the prompt's
        "Coded columns" rule covers.
        """
        return self.needs_decoding or self.reference_table_id is not None


@dataclass(frozen=True)
class TableMetadata:
    """A table's metadata as retrieved by `get_table_details` during a run."""

    gcp_id: str | None
    table_id: str
    dataset_id: str
    name: str
    partitioned_by: tuple[str, ...]
    period_start: str | None
    period_end: str | None
    columns: tuple[ColumnMetadata, ...]

    @property
    def is_partitioned(self) -> bool:
        """Whether the table has any partition column."""
        return bool(self.partitioned_by)

    @property
    def coded_columns(self) -> tuple[ColumnMetadata, ...]:
        """The columns that must be translated before they are filtered, grouped or shown."""
        return tuple(column for column in self.columns if column.is_coded)

    def column(self, name: str) -> ColumnMetadata | None:
        """The column with this exact name, or `None` if the table has no such column."""
        return next((column for column in self.columns if column.name == name), None)


def parse_table_details(raw: str | dict) -> TableMetadata:
    """Parse one `get_table_details` output into a :class:`TableMetadata`.

    Args:
        raw: The tool output — either its JSON string or an already-decoded dict.

    Returns:
        The parsed table metadata.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw

    columns = tuple(
        ColumnMetadata(
            name=column["name"],
            needs_decoding=column["needs_decoding"],
            reference_table_id=column.get("reference_table_id"),
        )
        for column in data.get("columns") or []
    )

    return TableMetadata(
        gcp_id=data["gcp_id"],
        table_id=data["id"],
        dataset_id=data["dataset_id"],
        name=data["name"],
        partitioned_by=tuple(data["partitioned_by"]),
        period_start=data["period_start"],
        period_end=data["period_end"],
        columns=columns,
    )


def index_by_gcp_id(tables: Iterable[TableMetadata]) -> dict[str, TableMetadata]:
    """Index tables by their `gcp_id` so a query's table refs resolve to their metadata.

    Tables with no `gcp_id` (unmaterialized) are skipped, since a query can't reference
    them. When the same table is fetched more than once in a run, the last one wins.

    Args:
        tables: The parsed table metadata to index.

    Returns:
        A `gcp_id -> TableMetadata` mapping.
    """
    return {table.gcp_id: table for table in tables if table.gcp_id}
