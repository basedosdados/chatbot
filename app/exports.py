from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, TypedDict

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import bigquery as bq
from pydantic import JsonValue

from app.settings import settings
from app.storage import get_object_size

type ExportFormat = Literal["AVRO", "CSV", "JSONL", "PARQUET"]


class QueryResultDownload(TypedDict):
    """A downloadable query result: its handle and the formats it can be exported in."""

    type: Literal["query_result"]
    query_ref: str
    formats: list[ExportFormat]


# One download offered on a message. Other kinds join here, discriminated by `type`.
type Download = QueryResultDownload

# A BigQuery result table as `TableReference.to_api_repr()`
type DestinationTable = dict[str, Any]


@dataclass(frozen=True)
class ExportSpec:
    extension: str
    mime_type: str
    dest: str


@dataclass(frozen=True)
class ExportedFile:
    """A materialized download: a single GCS object ready to be signed.

    The return of `materialize_export` — a transient description of the object
    just extracted (or reused via its deterministic key). Downloads are stateless,
    so this is deliberately not persisted; an exported file is not an artifact.
    """

    bucket: str
    object_key: str
    filename: str
    mime_type: str
    size_bytes: int


class ResultTableExpired(Exception):
    """The BigQuery result table backing a query_ref no longer exists (~24h TTL).

    A typed signal for the endpoint to map to a 410 — it carries the raw BigQuery
    cause for logs, not user-facing text (the endpoint owns that).
    """


class ResultTooLarge(Exception):
    """The result set is too big to export to a single file.

    A typed signal for the endpoint to map to a 400 — carries the raw cause for logs,
    not user-facing text (the endpoint owns that).
    """


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

# Formats the download offers. Every one is materializable on demand,
# so this is simply the set of supported export formats.
SUPPORTED_EXPORT_FORMATS: list[ExportFormat] = list(EXPORT_FORMATS)


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BIGQUERY_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def _extract_table_to_gcs(
    destination_table: DestinationTable,
    *,
    object_key: str,
    file_format: ExportFormat,
) -> int:
    """Extract a materialized BigQuery result table to a single GCS object.

    Byte-identical to the referenced table — no query is re-run.

    Args:
        destination_table (DestinationTable): `TableReference.to_api_repr()` of the result table.
        object_key (str): Destination object key within the export bucket.
        file_format (ExportFormat): The output format.

    Returns:
        int: The size in bytes of the written object.

    Raises:
        ResultTableExpired: If the result table no longer exists (~24h TTL).
        ResultTooLarge: If the result set is too big for a single file.
        RuntimeError: If the extract reports success but no object was written.
    """
    bucket = settings.GOOGLE_GCS_BUCKET
    gcs_uri = f"gs://{bucket}/{object_key}"

    # Extract the result table BigQuery already materialized straight to GCS. No re-run:
    # the file is byte-identical to what the model saw, and no query bytes are re-billed.
    try:
        _bq_client().extract_table(
            bq.TableReference.from_api_repr(destination_table),
            destination_uris=[gcs_uri],
            job_config=bq.ExtractJobConfig(
                destination_format=EXPORT_FORMATS[file_format].dest,
            ),
        ).result()
    except NotFound as e:
        # The result table lives only ~24h; once BigQuery expires it (or the query was
        # never actually run) the extract job 404s. Caught by type rather than by the
        # raw `reason` string. Surfaced as a typed signal the caller maps to a 410.
        raise ResultTableExpired(str(e)) from e
    except GoogleAPICallError as e:
        # "Too large for a single file" is a generic 400 with no dedicated exception
        # type or reason code, so a message match is the only signal available.
        errors = getattr(e, "errors", None) or []
        message = errors[0].get("message", "") if errors else ""
        if "too large to be exported to a single file" in message:
            raise ResultTooLarge(str(e)) from e
        raise

    size_bytes = get_object_size(bucket, object_key)

    if size_bytes is None:
        raise RuntimeError("Export completed but no file was written to GCS.")

    return size_bytes


def materialize_export(
    *,
    query_ref: str,
    destination_table: DestinationTable,
    file_format: ExportFormat,
    filename: str,
    thread_id: str,
) -> ExportedFile:
    """Materialize a query handle as a downloadable GCS object.

    Extracts the already-materialized result table to a single GCS object,
    byte-identical to what the model saw — no query is ever re-run. The object key is
    deterministic (`exports/{thread_id}/{query_ref}.{ext}`), so a repeat download of the
    same (query_ref, format) reuses the existing object instead of re-extracting, and a
    click after the object was lifecycle-deleted re-extracts it in place — as long as the
    result table still exists, otherwise `ResultTableExpired` (mapped to a 410).

    Args:
        query_ref (str): The handle whose result table is exported.
        destination_table (DestinationTable): `TableReference.to_api_repr()` of that table.
        file_format (ExportFormat): The output format.
        filename (str): Base name for the file (the extension is appended).
        thread_id (str): Owning thread, used for the deterministic object key.

    Returns:
        ExportedFile: The GCS object (bucket, key, filename, mime type, size) to sign.

    Raises:
        ResultTableExpired: the result table no longer exists (~24h TTL).
        ResultTooLarge: the result set is too big for a single file.
    """
    bucket = settings.GOOGLE_GCS_BUCKET
    extension = EXPORT_FORMATS[file_format].extension
    object_key = f"exports/{thread_id}/{query_ref}.{extension}"

    # Reuse the object if it's already there (a prior download of this query+format);
    # a missing object (never made, or lifecycle-deleted) is (re-)extracted in place.
    size_bytes = get_object_size(bucket, object_key)

    if size_bytes is None:
        size_bytes = _extract_table_to_gcs(
            destination_table,
            object_key=object_key,
            file_format=file_format,
        )

    return ExportedFile(
        bucket=bucket,
        object_key=object_key,
        filename=f"{filename}.{extension}",
        mime_type=EXPORT_FORMATS[file_format].mime_type,
        size_bytes=size_bytes,
    )


def collect_query_handles(
    artifacts: Iterable[JsonValue | None],
) -> dict[str, DestinationTable]:
    """Map `query_ref` -> `destination_table` from a run's tool-output artifacts.

    Picks out the `query_result` handles minted by `execute_bigquery_sql` tool.
    The `query_ref`s that *actually executed* this run are used both to sanitise the
    refs the model reported and to persist the handle a later download materializes from.

    Args:
        artifacts (Iterable[JsonValue | None]): The run's tool-output artifacts.

    Returns:
        dict[str, DestinationTable]: Each executed `query_ref` mapped to its result table.
    """
    handles: dict[str, DestinationTable] = {}
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("type") == "query_result"
            and artifact.get("query_ref") is not None
            and artifact.get("destination_table") is not None
        ):
            handles[artifact["query_ref"]] = artifact["destination_table"]
    return handles


def sanitize_sql_query_refs(
    structured_response: dict[str, Any] | None, executed_refs: set[str]
) -> dict[str, Any] | None:
    """Return a copy of `structured_response` with unexecuted `sql_queries[].query_ref`s nulled.

    Args:
        structured_response (dict[str, Any] | None): The answer's structured response.
        executed_refs (set[str]): The `query_ref`s that actually executed this run.

    Returns:
        dict[str, Any] | None: A sanitized copy of `structured_response` with unexecuted refs nulled,
            or the input unchanged when it has no `sql_queries` to clean.
    """
    if not isinstance(structured_response, dict):
        return structured_response

    sql_queries = structured_response.get("sql_queries")
    if not sql_queries:
        return structured_response

    sanitized = [
        {**query, "query_ref": None}
        if isinstance(query, dict) and query.get("query_ref") not in executed_refs
        else query
        for query in sql_queries
    ]

    return {**structured_response, "sql_queries": sanitized}


def derive_downloads(
    structured_response: dict[str, Any] | None,
) -> list[Download]:
    """Derive the downloads offered for an answer — one per backing query.

    Each backing query carrying a sanitized `query_ref` becomes a `query_result`
    download in every supported format.

    Args:
        structured_response (dict[str, Any] | None): The answer's structured response.

    Returns:
        list[Download]: One download per backing query; empty when none is downloadable.
    """
    if not isinstance(structured_response, dict):
        return []

    return [
        {
            "type": "query_result",
            "query_ref": query["query_ref"],
            "formats": SUPPORTED_EXPORT_FORMATS,
        }
        for query in structured_response.get("sql_queries") or []
        if isinstance(query, dict) and query.get("query_ref")
    ]
