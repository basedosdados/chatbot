import inspect
import json
import re
import uuid
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Literal

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import bigquery as bq
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.agent.tools.exceptions import handle_tool_errors
from app.artifacts import Artifact, ArtifactMetadata, RemoteObjectSource
from app.settings import settings
from app.storage import get_object_size

type ExportFormat = Literal["AVRO", "CSV", "JSONL", "PARQUET"]


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
    "JSONL": ExportSpec(
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


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
def execute_bigquery_sql(
    sql_query: str, config: RunnableConfig
) -> tuple[str, dict[str, Any] | None]:
    """Execute a SQL query against BigQuery tables from the Base dos Dados database.

    PRECONDITION — only call this when the question is already specific enough to
    answer with data. For a broad/exploratory question (a bare topic) or one that
    references an entity the user did not name, do NOT call this tool: explore the
    metadata and ask the user to refine the question first.

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
        str: A JSON object with `query_ref` (an opaque handle you can later pass to
            `export_query_results` so the user can download exactly these rows),
            `row_count`, and `results` (the rows as a JSON array). If the query
            returns no rows, a short message string is returned instead.
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
        rows = [dict(row) for row in job.result()]
    except GoogleAPICallError as e:
        reason = e.errors[0].get("reason") if getattr(e, "errors", None) else None
        if reason == "bytesBilledLimitExceeded":
            raise ValueError(
                f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. Filter by partitioned columns."
            ) from e
        raise

    if not rows:
        message = (
            "Query returned 0 rows. Review the filters per the empty-result protocol."
        )
        return json.dumps(message, ensure_ascii=False), None

    # Reference the anonymous result table BigQuery already materialized (~24h TTL)
    # so export_query_results can hand back exactly these rows without re-running.
    query_ref = f"q_{uuid.uuid4().hex[:16]}"

    content = json.dumps(
        {"query_ref": query_ref, "row_count": len(rows), "results": rows},
        ensure_ascii=False,
        default=str,
    )

    artifact = {
        "type": "query_result",
        "query_ref": query_ref,
        "destination_table": job.destination.to_api_repr(),
    }

    return content, artifact


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
        client = _bq_client()
        job = client.query(
            search_query,
            job_config=bq.QueryJobConfig(query_parameters=query_params, labels=labels),
        )
        results = [dict(row) for row in job.result()]
    except NotFound as e:
        raise ValueError("Dictionary table not found for this dataset.") from e

    return json.dumps(results, ensure_ascii=False, indent=2, default=str)


def _resolve_destination_table(state: dict, query_ref: str) -> dict | None:
    """Look up the BigQuery result table for a prior `execute_bigquery_sql` result.

    Scans the thread's message history (most recent first) for the result whose
    artifact carries `query_ref` and returns its serialized table reference (the
    `TableReference.to_api_repr()` dict), or None if no matching result is still
    in the agent's context.

    Args:
        state (dict): The injected agent state, containing `messages`.
        query_ref (str): The handle minted by `execute_bigquery_sql`.

    Returns:
        dict | None: The destination table reference, or None if not found.
    """
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact
        if isinstance(artifact, dict) and artifact.get("query_ref") == query_ref:
            return artifact.get("destination_table")
    return None


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
def export_query_results(
    query_ref: str,
    filename: str,
    config: RunnableConfig,
    state: Annotated[dict, InjectedState],
    file_format: ExportFormat = "CSV",
) -> tuple[str, dict[str, Any]]:
    """Export the results of a previously executed query to a single downloadable file in Google Cloud Storage.

    Call this when the user asks to download, export, or save the data. Pass the
    `query_ref` returned by the `execute_bigquery_sql` call whose results the user
    wants — the exact rows from that query are exported, so run the query with
    `execute_bigquery_sql` first. Do NOT rewrite or re-run the SQL here.

    Exports are capped at ~1GB — if the results exceed that, the export will fail
    and you should ask the user to narrow the query (add filters, select fewer columns, limit rows).

    A download link is surfaced to the user by the application. You will not see the URL yourself.

    Args:
        query_ref (str): The `query_ref` handle from a prior `execute_bigquery_sql` result.
        filename (str): Short, human-readable base name for the file (no extension, no path separators).
        file_format (str): Output format. One of "AVRO", "CSV", "JSONL", "PARQUET". Defaults to "CSV".

    Returns:
        str: Confirmation that the export succeeded and the object was created.
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

    destination_table = _resolve_destination_table(state, query_ref)

    if destination_table is None:
        raise ValueError(
            f"No query results found for query_ref '{query_ref}'. Run the query with "
            "`execute_bigquery_sql` first, then export using the `query_ref` it returns."
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

    client = _bq_client()

    # Extract the result table BigQuery already materialized for the referenced
    # query straight to GCS. No re-run: the file is byte-identical to what the
    # model saw, and no query bytes are billed again.
    try:
        client.extract_table(
            bq.TableReference.from_api_repr(destination_table),
            destination_uris=[gcs_uri],
            job_config=bq.ExtractJobConfig(
                destination_format=EXPORT_FORMATS[file_format].dest,
                labels=labels,
            ),
        ).result()
    except NotFound as e:
        # The result table lives only ~24h; once BigQuery expires it (or the query
        # was never actually run) the extract job 404s. Caught by type rather than
        # by the raw `reason` string, which is exactly what the client maps to 404.
        raise ValueError(
            "Those query results are no longer available to export (results are "
            "kept only temporarily). Please re-run the query and try again."
        ) from e
    except GoogleAPICallError as e:
        # "Too large for a single file" is a generic 400 with no dedicated exception
        # type or reason code, so a message match is the only signal available.
        errors = getattr(e, "errors", None) or []
        message = errors[0].get("message", "") if errors else ""
        if "too large to be exported to a single file" in message:
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
