import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cache
from typing import Any, Literal, TypedDict

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import bigquery as bq
from pydantic import JsonValue

from app.settings import settings
from app.storage import get_object_size

type ExportFormat = Literal["AVRO", "CSV", "JSONL", "PARQUET"]


@dataclass(frozen=True)
class CollectedQueryHandle:
    """An executed query's handle, gathered from a tool-output artifact."""

    query_ref: str
    slug: str
    destination_table: dict[str, Any]


class QueryResultDownload(TypedDict):
    """A downloadable query result: its handle, slug, and export formats."""

    type: Literal["query_result"]
    query_ref: str
    slug: str
    formats: list[ExportFormat]


class ResultTableExpired(Exception):
    """The BigQuery result table backing a query_ref no longer exists (~24h TTL).
    A typed signal for the endpoint to map to a 410.
    """


class ResultTooLarge(Exception):
    """The result set is too big to export, either over our own
    `MAX_EXPORT_BYTES` budget or over BigQuery's single-file limit.
    A typed signal for the endpoint to map to a 400.
    """


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


# Formats offered to the frontend, each materializable on demand via a BigQuery extract job.
OFFERED_EXPORT_FORMATS: list[ExportFormat] = ["AVRO", "CSV", "JSONL", "PARQUET"]

# BigQuery's anonymous result tables live ~24h.
RESULT_TABLE_TTL = timedelta(hours=24)


def is_result_expired(created_at: datetime) -> bool:
    """Whether a query handle is old enough that its result table has likely expired."""
    return datetime.now(timezone.utc) - created_at >= RESULT_TABLE_TTL


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BILLING_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def _extract_table_to_gcs(
    destination_table: dict[str, Any],
    *,
    object_key: str,
    file_format: ExportFormat,
) -> int:
    """Extract a materialized BigQuery result table to a single GCS object.

    Args:
        destination_table (dict[str, Any]): `TableReference.to_api_repr()` of the result table.
        object_key (str): Destination object key within the export bucket.
        file_format (ExportFormat): The output format.

    Returns:
        int: Size in bytes of the written object.

    Raises:
        ResultTableExpired: The result table no longer exists (~24h TTL).
        ResultTooLarge: The result set is over `MAX_EXPORT_BYTES` or too big for one file.
        RuntimeError: The extract reported success but no object was written.
    """
    bucket = settings.GOOGLE_GCS_BUCKET
    gcs_uri = f"gs://{bucket}/{object_key}"
    table_ref = bq.TableReference.from_api_repr(destination_table)
    bytes_limit = settings.MAX_EXPORT_BYTES

    # Extract straight from the table BigQuery already materialized.
    try:
        # num_bytes is BigQuery's logical size and only approximates the exported file
        num_bytes = _bq_client().get_table(table_ref).num_bytes

        if num_bytes is not None and num_bytes > bytes_limit:
            raise ResultTooLarge(
                f"Result table is {num_bytes} bytes, over the {bytes_limit} bytes export limit."
            )

        _bq_client().extract_table(
            table_ref,
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
    destination_table: dict[str, Any],
    file_format: ExportFormat,
    filename: str,
    message_id: str,
) -> ExportedFile:
    """Materialize a query handle as a downloadable GCS object.

    The object key is deterministic and message-scoped, so a repeat reuses it.

    Args:
        query_ref (str): The handle whose result table is exported.
        destination_table (dict[str, Any]): `TableReference.to_api_repr()` of that table.
        file_format (ExportFormat): The output format.
        filename (str): Base name for the file (the extension is appended).
        message_id (str): The owning message/run, used for the deterministic object key.

    Returns:
        ExportedFile: The GCS object (bucket, key, filename, mime type, size) to sign.

    Raises:
        ResultTableExpired: The result table no longer exists (~24h TTL).
        ResultTooLarge: The result set is too big for a single file.
    """
    bucket = settings.GOOGLE_GCS_BUCKET
    extension = EXPORT_FORMATS[file_format].extension
    object_key = f"query_results/{message_id}/{query_ref}.{extension}"

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


def sanitize_export_filename(slug: str, fallback: str) -> str:
    """Sanitize a query's slug into a safe base filename (no extension).

    Args:
        slug (str): The query's slug.
        fallback (str): Base name to use when the slug yields nothing filesystem-safe.

    Returns:
        str: A filesystem-safe base filename, without extension.
    """
    filename = re.sub(r"[^\w-]+", "_", slug).strip("_")
    return filename or fallback


def collect_query_handles(
    artifacts: Iterable[JsonValue | None],
) -> list[CollectedQueryHandle]:
    """Return the `query_result` handles found in a run's tool-output artifacts.

    Picks out the handles minted by the `execute_bigquery_sql` tool (each with its
    `query_ref`, display `slug`, and result table).

    Args:
        artifacts (Iterable[JsonValue | None]): The run's tool-output artifacts.

    Returns:
        list[CollectedQueryHandle]: The query handles found, in artifact order.
    """
    return [
        CollectedQueryHandle(
            query_ref=artifact["query_ref"],
            slug=artifact["slug"],
            destination_table=artifact["destination_table"],
        )
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("type") == "query_result"
    ]


def query_result_download(query_ref: str, slug: str) -> QueryResultDownload:
    """Build the download offered for one executed query.

    Args:
        query_ref (str): The query's handle.
        slug (str): The query's slug.

    Returns:
        QueryResultDownload: The download descriptor.
    """
    return {
        "type": "query_result",
        "query_ref": query_ref,
        "slug": slug,
        "formats": OFFERED_EXPORT_FORMATS,
    }
