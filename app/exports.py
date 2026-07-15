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
    just extracted or reused; downloads are stateless, so it is not persisted.
    """

    bucket: str
    object_key: str
    filename: str
    mime_type: str
    size_bytes: int


class ResultTableExpired(Exception):
    """The BigQuery result table backing a query_ref no longer exists (~24h TTL).
    A typed signal for the endpoint to map to a 410.
    """


class ResultTooLarge(Exception):
    """The result set is too big to export to a single file.
    A typed signal for the endpoint to map to a 400.
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

    # Extract straight from the table BigQuery already materialized.
    try:
        _bq_client().extract_table(
            bq.TableReference.from_api_repr(destination_table),
            destination_uris=[gcs_uri],
            job_config=bq.ExtractJobConfig(
                destination_format=EXPORT_FORMATS[file_format].dest,
            ),
        ).result()
    except NotFound as e:
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
    message_id: str,
) -> ExportedFile:
    """Materialize a query handle as a downloadable GCS object.

    Extracts the result table (see `_extract_table_to_gcs`) to a deterministic,
    message-scoped key (`exports/{message_id}/{query_ref}.{ext}`; `query_ref` is
    only unique within its run). The determinism makes downloads idempotent: a repeat
    reuses the existing object, and a lifecycle-deleted one is re-extracted in place.

    Args:
        query_ref (str): The handle whose result table is exported.
        destination_table (DestinationTable): `TableReference.to_api_repr()` of that table.
        file_format (ExportFormat): The output format.
        filename (str): Base name for the file (the extension is appended).
        message_id (str): The owning message/run, used for the deterministic object key.

    Returns:
        ExportedFile: The GCS object (bucket, key, filename, mime type, size) to sign.

    Raises:
        ResultTableExpired: the result table no longer exists (~24h TTL).
        ResultTooLarge: the result set is too big for a single file.
    """
    bucket = settings.GOOGLE_GCS_BUCKET
    extension = EXPORT_FORMATS[file_format].extension
    object_key = f"exports/{message_id}/{query_ref}.{extension}"

    # Reuse a previously extracted object; (re-)extract only when it is absent.
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
