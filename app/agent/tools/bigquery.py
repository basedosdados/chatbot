import inspect
import json
import re
import uuid
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery as bq
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.tools.exceptions import handle_tool_errors
from app.artifacts import Artifact, ArtifactMetadata, RemoteObjectSource
from app.settings import settings
from app.storage import get_object_size

type ExportFormat = Literal["AVRO", "CSV", "JSON", "PARQUET"]


@dataclass(frozen=True, slots=True)
class ExportSpec:
    extension: str
    mime_type: str
    dest: str


EXPORT_FORMATS = {
    "AVRO": ExportSpec(
        extension="avro",
        mime_type="application/avro",
        dest=bq.DestinationFormat.AVRO,
    ),
    "CSV": ExportSpec(
        extension="csv",
        mime_type="text/csv",
        dest=bq.DestinationFormat.CSV,
    ),
    "JSON": ExportSpec(
        extension="jsonl",
        mime_type="application/jsonl",
        dest=bq.DestinationFormat.NEWLINE_DELIMITED_JSON,
    ),
    "PARQUET": ExportSpec(
        extension="parquet",
        mime_type="application/vnd.apache.parquet",
        dest=bq.DestinationFormat.PARQUET,
    ),
}

EXPORT_FILENAME_MAX_LEN = 64

EXPORT_FILENAME_PATTERN = re.compile(r"^[\w\-. ]+$")

MAX_BYTES_BILLED = 10 * 10**9


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
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
        sql_query (str): Standard GoogleSQL query. Must reference
            tables using their full `gcp_id` from `get_dataset_details()`.

    Best practices:
        - Use fully qualified names: `project.dataset.table`
        - Select only needed columns, avoid `SELECT *`
        - Add `LIMIT` for exploration
        - Filter early with `WHERE` clauses
        - Order by relevant columns
        - Never use DDL/DML commands
        - Use appropriate data types in comparisons

    Returns:
        str: Query results as JSON array. Empty results return "[]".
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
        "thread_id": config.get("configurable", {}).get("thread_id", "unknown"),
        "user_id": config.get("configurable", {}).get("user_id", "unknown"),
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
        results = [dict(row) for row in job.result()]
    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "bytesBilledLimitExceeded":
            raise ValueError(
                f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. "
                "Add WHERE filters or select fewer columns."
            ) from e
        raise

    return json.dumps(results, ensure_ascii=False, indent=2, default=str)


@tool
@handle_tool_errors
def decode_table_values(
    table_gcp_id: str, config: RunnableConfig, column_name: str | None = None
) -> str:
    """Decode coded values from a table using its dataset's `dicionario` table.

    Use when column values appear to be codes (e.g., 1,2,3 or A,B,C) and the
    column does NOT have a `reference_table_id` in `get_table_details()` metadata.

    Args:
        table_gcp_id (str): Full BigQuery table reference.
        column_name (str | None, optional): Column with coded values. If `None`,
            all columns will be used. Defaults to `None`.

    Returns:
        str: JSON array with chave (code) and valor (meaning) mappings.
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
        client = _bq_client()
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

    return json.dumps(results, ensure_ascii=False, indent=2, default=str)


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
def export_query_results(
    sql_query: str,
    filename: str,
    config: RunnableConfig,
    file_format: ExportFormat = "CSV",
) -> tuple[str, dict[str, Any]]:
    """Export the results of a SELECT query to a single downloadable file in Google Cloud Storage.

    Call this with the SAME SQL you previously ran via `execute_bigquery_sql`
    when the user asks to download, export, or save the data. Exports are
    capped at ~1GB — if the results exceeds that, the export will fail
    and you should ask the user to narrow the query (add filters, select fewer columns, limit rows).

    A download link is surfaced to the user by the application. You will not see the URL yourself.

    Args:
        sql_query (str): Standard GoogleSQL SELECT query.
        filename (str): Short, human-readable base name for the file (no extension, no path separators).
        file_format (str): Output format. One of "AVRO", "CSV", "JSON", "PARQUET". Defaults to "CSV".

    Returns:
        str: Confirmation that the export suceeded and the object was created.
    """
    if len(filename) > EXPORT_FILENAME_MAX_LEN:
        raise ValueError(
            f"`filename` must be at most {EXPORT_FILENAME_MAX_LEN} characters."
        )

    if not EXPORT_FILENAME_PATTERN.match(filename):
        raise ValueError(
            "`filename` must contain only letters, digits, hyphens, underscores, "
            "dots, and spaces (no path separators)."
        )

    client = _bq_client()

    dry_run = client.query(
        sql_query, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False)
    )

    if dry_run.statement_type != "SELECT":
        raise ValueError(
            f"Only SELECT statements are allowed, got {dry_run.statement_type}."
        )

    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    object_id = uuid.uuid4().hex

    extension = EXPORT_FORMATS[file_format].extension
    object_key = f"exports/{thread_id}/{object_id}.{extension}"
    gcs_uri = f"gs://{settings.GOOGLE_GCS_BUCKET}/{object_key}"

    labels = {
        "thread_id": thread_id,
        "user_id": config.get("configurable", {}).get("user_id", "unknown"),
        "tool_name": inspect.currentframe().f_code.co_name,
    }

    # Query results land in an anonymous temp table with ~24h TTL and no cleanup needed.
    try:
        query_job = client.query(
            sql_query,
            job_config=bq.QueryJobConfig(
                maximum_bytes_billed=MAX_BYTES_BILLED,
                labels=labels,
            ),
        )
        query_job.result()
    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "bytesBilledLimitExceeded":
            raise ValueError(
                f"Export exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. "
                "Add WHERE filters or select fewer columns."
            ) from e
        raise

    # The temp table is then extracted to Google Cloud Storage.
    try:
        client.extract_table(
            query_job.destination,
            destination_uris=[gcs_uri],
            job_config=bq.ExtractJobConfig(
                destination_format=EXPORT_FORMATS[file_format].dest,
                labels=labels,
            ),
        ).result()
    except GoogleAPICallError as e:
        msg = e.errors[0].get("message", "") if getattr(e, "errors", None) else ""
        if "too large to be exported to a single file" in msg:
            raise ValueError(
                "Result set is too large to export as a single file. "
                "Add WHERE filters, select fewer columns, or limit the number of rows."
            ) from e
        raise

    size_bytes = get_object_size(settings.GOOGLE_GCS_BUCKET, object_key)

    if size_bytes is None:
        raise RuntimeError("Export completed but no file was written to GCS.")

    content = json.dumps(
        {
            "status": "success",
            "filename": f"{filename}.{extension}",
        }
    )

    artifact = Artifact(
        source=RemoteObjectSource(
            bucket=settings.GOOGLE_GCS_BUCKET,
            object_key=object_key,
        ),
        metadata=ArtifactMetadata(
            filename=f"{filename}.{extension}",
            mime_type=EXPORT_FORMATS[file_format].mime_type,
            size_bytes=size_bytes,
        ),
    ).model_dump(mode="json")

    return content, artifact
