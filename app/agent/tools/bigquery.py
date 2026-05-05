import inspect
import json
from functools import cache

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery as bq
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.tools.exceptions import handle_tool_errors
from app.settings import settings

MAX_BYTES_BILLED = 10 * 10**9


@cache
def _get_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BIGQUERY_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


@tool
@handle_tool_errors
def execute_bigquery_sql(sql_query: str, config: RunnableConfig) -> str:
    """Execute a SQL query against BigQuery tables from the Base dos Dados database.

    Use AFTER identifying the right datasets and understanding tables structure.
    It includes a 10GB processing limit for safety.

    Args:
        sql_query (str): Standard GoogleSQL query. Must reference tables using their full `gcp_id` from `get_dataset_details()`.

    Rules:
        - Use fully qualified names: `project.dataset.table`.
        - Select only needed columns, don't use `SELECT *`.
        - Always filter by partitioned columns when present (see `partitioned_by` in `get_table_details` results). In `JOIN` queries, each partitioned table needs its own partition filter.
        - Order by relevant columns.
        - Use `LIMIT` for exploration.
        - Use appropriate data types in comparisons.
        - Only `SELECT` statements are allowed.

    Returns:
        str: Query results as JSON array.
    """
    client = _get_client()

    dry_run = client.query(
        sql_query, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False)
    )

    if dry_run.statement_type != "SELECT":
        raise ValueError(
            f"Only SELECT statements are allowed, got {dry_run.statement_type}."
        )

    labels = {
        "thread_id": config.get("configurable", {}).get("thread_id", "unknown"),
        "user_id": config.get("configurable", {}).get("user_id", "unknown"),
        "tool_name": inspect.currentframe().f_code.co_name,
    }

    try:
        job = client.query(
            sql_query,
            job_config=bq.QueryJobConfig(
                maximum_bytes_billed=MAX_BYTES_BILLED, labels=labels
            ),
        )
        results = [dict(row) for row in job.result()]

        if not results:
            results = "Query returned 0 rows. Review the filters per the empty-result protocol."

    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "bytesBilledLimitExceeded":
            raise ValueError(
                f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. Filter by partitioned columns."
            ) from e
        raise

    return json.dumps(results, ensure_ascii=False, default=str)


@tool
@handle_tool_errors
def decode_table_values(
    table_gcp_id: str, config: RunnableConfig, column_name: str | None = None
) -> str:
    """Fetch the dictionary mapping (code -> human-readable value) for a coded column.

    REQUIRED whenever a column has `needs_decoding: true` in `get_table_details`,
    BEFORE writing any SQL that filters, joins, or displays that column.

    Returns pairs of `chave` (the literal value stored in the table) and `valor` (its meaning).

    Args:
        table_gcp_id (str): Full BigQuery table reference (`project.dataset.table`).
        column_name (str | None, optional): The specific column to decode. Always
            provide this when you know which column you need; passing None returns
            the entire dictionary for the table and wastes tokens.

    Returns:
        str: JSON array of {nome_coluna, chave, valor} entries.
    """
    if "`" in table_gcp_id:
        table_gcp_id = table_gcp_id.replace("`", "")

    try:
        project_name, dataset_name, table_name = table_gcp_id.split(".")
    except ValueError:
        raise ValueError(
            f"Invalid table reference: '{table_gcp_id}'. Expected format: project.dataset.table"
        )

    dict_table_id = f"`{project_name}.{dataset_name}.dicionario`"

    search_query = f"""
        SELECT nome_coluna, chave, valor
        FROM {dict_table_id}
        WHERE id_tabela = @table_name
    """

    query_params = [
        bq.ScalarQueryParameter("table_name", "STRING", table_name),
    ]

    if column_name is not None:
        search_query += "AND nome_coluna = @column_name\n"
        query_params.append(
            bq.ScalarQueryParameter("column_name", "STRING", column_name),
        )

    search_query += "ORDER BY nome_coluna, chave"

    labels = {
        "thread_id": config.get("configurable", {}).get("thread_id", "unknown"),
        "user_id": config.get("configurable", {}).get("user_id", "unknown"),
        "tool_name": inspect.currentframe().f_code.co_name,
    }

    try:
        client = _get_client()
        job = client.query(
            search_query,
            job_config=bq.QueryJobConfig(query_parameters=query_params, labels=labels),
        )
        results = [dict(row) for row in job.result()]
    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "notFound":
            raise ValueError("Dictionary table not found for this dataset.") from e
        raise

    return json.dumps(results, ensure_ascii=False, default=str)
