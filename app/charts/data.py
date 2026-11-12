import asyncio
import json
from decimal import Decimal
from functools import cache
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import bigquery as bq

from app.db.database import AsyncDatabase, sessionmaker
from app.db.models import QueryHandle
from app.exports import ResultTableExpired, is_result_expired
from app.settings import settings


class ChartHandleNotFound(Exception):
    """No query result exists for the given query_ref in this thread."""


class ChartResultTooLarge(Exception):
    """The result has more rows than a chart should bind."""


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BILLING_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def _json_default(value: Any) -> Any:
    """Coerce a BigQuery value that JSON can't render natively into a JSON-native one.

    `Decimal` (NUMERIC/BIGNUMERIC) becomes a number so quantitative encodings and color
    scales see a number, not a string. Everything else (date/datetime/…) falls back to
    its string form, which Vega-Lite parses for temporal encodings.

    Args:
        value (Any): The value json.dumps could not serialize.

    Returns:
        Any: A JSON-native replacement.
    """
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return str(value)


def _fetch_rows(
    destination_table: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fetch a result table's rows for charting, capped at settings.CHART_MAX_BYTES.

    Fetches row by row and stops as soon as the JSON that would be bound into the spec
    exceeds the budget, so a huge result is rejected without materializing all of it.

    Args:
        destination_table (dict[str, Any]): `TableReference.to_api_repr()` of the result table.

    Returns:
        tuple[dict[str, str], list[dict[str, Any]]]: Column names mapped to their BigQuery
            types, in schema order, and row dicts.

    Raises:
        ChartResultTooLarge: The bound data would exceed settings.CHART_MAX_BYTES.
        ResultTableExpired: The result table no longer exists (~24h TTL).
    """
    table_ref = bq.TableReference.from_api_repr(destination_table)

    try:
        table = _bq_client().get_table(table_ref)

        columns = {field.name: field.field_type for field in table.schema}

        rows = []
        size = 0
        for row in _bq_client().list_rows(table_ref):
            # Serialize once to both measure the row and coerce BigQuery types
            # (date/datetime/Decimal/…) to JSON-native values. Raw objects are
            # not valid JSON and would fail serialization when the spec is bound.
            serialized = json.dumps(
                dict(row), ensure_ascii=False, default=_json_default
            )
            size += len(serialized.encode())
            if size > settings.CHART_MAX_BYTES:
                raise ChartResultTooLarge(
                    f"The result is too large to chart (over "
                    f"{settings.CHART_MAX_BYTES // (1024 * 1024)} MB of data). Aggregate "
                    "or summarize it in SQL first, then chart the smaller result."
                )
            rows.append(json.loads(serialized))
    except NotFound as e:
        raise ResultTableExpired(str(e)) from e

    return columns, rows


async def fetch_chart_data(
    query_ref: str, thread_id: str
) -> tuple[QueryHandle, dict[str, str], list[dict[str, Any]]]:
    """Resolve a chartable result: authorize the handle, then read its capped rows.

    Args:
        query_ref (str): The handle of the result to chart.
        thread_id (str): The thread the handle must belong to (authorization).

    Returns:
        tuple[QueryHandle, dict[str, str], list[dict[str, Any]]]: The handle,
            its columns (mapped to their BigQuery types), and rows.

    Raises:
        ChartHandleNotFound: No such result in this thread.
        ResultTableExpired: The result expired (by age or missing table).
    """
    async with sessionmaker() as session:
        handle = await AsyncDatabase(session).get_query_handle_from_thread(
            query_ref, thread_id
        )

    if handle is None:
        raise ChartHandleNotFound(
            f"No query result found for '{query_ref}'. "
            "Call list_query_results to see the available results."
        )

    if is_result_expired(handle.created_at):
        raise ResultTableExpired(
            f"The result for '{query_ref}' has expired (results are kept ~24h). "
            "Re-run the query, then chart the new result."
        )

    columns, rows = await asyncio.to_thread(_fetch_rows, handle.destination_table)

    return handle, columns, rows


def spread_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Pick `size` rows evenly spaced across the result, first and last included.

    A result is usually sorted by its grouping columns, so the first rows often
    all share one group; an even spread shows every part of it.

    Args:
        rows (list[dict[str, Any]]): The result rows.
        size (int): How many rows to pick.

    Returns:
        list[dict[str, Any]]: The sample, in result order.
    """
    if len(rows) <= size:
        return rows
    step = (len(rows) - 1) / (size - 1)
    return [rows[round(i * step)] for i in range(size)]
