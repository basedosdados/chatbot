import inspect
import itertools
import json
import uuid
from functools import cache
from typing import Any

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import bigquery as bq
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.agent.context import AgentContext
from app.agent.tools.exceptions import handle_tool_errors
from app.settings import settings

MAX_BYTES_BILLED = 10 * 10**9

# Cap on how many rows are serialized into the agent's context. The full result set
# is still materialized in BigQuery's destination table, so downloads (via query_ref)
# get every row regardless — this only keeps a large result from blowing up context.
MAX_CONTEXT_ROWS = 1000


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BILLING_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
def execute_bigquery_sql(
    sql_query: str, slug: str, runtime: ToolRuntime[AgentContext]
) -> tuple[str, dict[str, Any] | None]:
    """Run one read-only GoogleSQL query against Base dos Dados (10GB scan limit).

    Args:
        sql_query (str): The query. Follow the SQL rules in the system prompt.
        slug (str): Short, filename-safe, lowercase_with_underscores name for this result's
            download, in the user's language. Must be distinct from the other queries in the
            current request — each slug names a separate download.

    Returns:
        JSON object with `row_count` and `rows`.
    """
    client = _bq_client()

    dry_run = client.query(
        sql_query, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False)
    )

    if dry_run.statement_type != "SELECT":
        raise ValueError(
            f"Only SELECT statements are allowed, got {dry_run.statement_type}."
        )

    labels = {
        "thread_id": runtime.context.thread_id,
        "user_id": runtime.context.user_id,
        "tool_name": inspect.currentframe().f_code.co_name,
    }

    try:
        job = client.query(
            sql_query,
            job_config=bq.QueryJobConfig(
                maximum_bytes_billed=MAX_BYTES_BILLED,
                labels=labels,
            ),
        )
        result = job.result()
        total_rows = result.total_rows
        rows = [dict(row) for row in itertools.islice(result, MAX_CONTEXT_ROWS)]
    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "bytesBilledLimitExceeded":
            raise ValueError(
                f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. Filter by partitioned columns."
            ) from e
        raise

    payload = {"row_count": total_rows, "rows": rows}

    # Surface truncation only when it actually happened, so the agent knows the rows
    # it sees are a subset — and that the full set is still available for download.
    if total_rows > len(rows):
        payload["truncated"] = True
        payload["truncation_note"] = (
            f"Only the first {len(rows)} of {total_rows} rows are shown here to keep "
            "the context small. The full result can still be downloaded from the interface."
        )

    content = json.dumps(payload, ensure_ascii=False, default=str)

    # No rows -> nothing to download
    if not rows:
        return content, None

    # Server-minted handle for the anonymous result table BigQuery already materialized
    # (~24h TTL), so a later export hands back exactly these rows without re-running.
    query_ref = f"qr_{uuid.uuid4().hex}"

    artifact = {
        "type": "query_result",
        "query_ref": query_ref,
        "slug": slug,
        "destination_table": job.destination.to_api_repr(),
    }

    return content, artifact


@tool
@handle_tool_errors
def decode_table_values(
    table_gcp_id: str,
    runtime: ToolRuntime[AgentContext],
    column_name: str | None = None,
) -> str:
    """Fetch the code->label dictionary for a coded (`needs_decoding`) column.

    Args:
        table_gcp_id (str): Full table reference (`project.dataset.table`).
        column_name (str | None, optional): The column to decode. Omit only to fetch the
            whole table's dictionary, which costs more tokens.

    Returns:
        JSON array of {nome_coluna, chave (stored value), valor (meaning)}.
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
        "thread_id": runtime.context.thread_id,
        "user_id": runtime.context.user_id,
        "tool_name": inspect.currentframe().f_code.co_name,
    }

    try:
        client = _bq_client()
        job = client.query(
            search_query,
            job_config=bq.QueryJobConfig(query_parameters=query_params, labels=labels),
        )
        results = [dict(row) for row in job.result()]
    except NotFound as e:
        raise ValueError("Dictionary table not found for this dataset.") from e

    return json.dumps(results, ensure_ascii=False, indent=2, default=str)
