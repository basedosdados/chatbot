"""Load the question/thread rows into a standing BigQuery dataset.

Both tables are fully replaced each run (WRITE_TRUNCATE); the dataset is small.
"""

import json
import tempfile
from datetime import datetime

from google.cloud import bigquery as bq

from analytics.extract import EXPORT_FORMATS

QUESTIONS_TABLE = "questions"
THREADS_TABLE = "threads"

# Explicit schemas keep column types stable across runs: autodetect could infer a
# different type on a day when some column happens to be all-null and then fail
# the load into the existing table.
_QUESTIONS_SCHEMA = [
    bq.SchemaField("user_id", "STRING"),
    bq.SchemaField("is_team", "INTEGER"),
    bq.SchemaField("thread_id", "STRING"),
    bq.SchemaField("thread_title", "STRING"),
    bq.SchemaField("language", "STRING"),
    bq.SchemaField("thread_deleted", "INTEGER"),
    bq.SchemaField("turn_index", "INTEGER"),
    bq.SchemaField("asked_at_utc", "TIMESTAMP"),
    bq.SchemaField("question", "STRING"),
    bq.SchemaField("answer", "STRING"),
    bq.SchemaField("answer_status", "STRING"),
    bq.SchemaField("model_uri", "STRING"),
    bq.SchemaField("downloads_offered", "INTEGER"),
    bq.SchemaField("charts_rendered", "INTEGER"),
    bq.SchemaField("exports_made", "INTEGER"),
    *(bq.SchemaField(f"exports_{fmt.lower()}", "INTEGER") for fmt in EXPORT_FORMATS),
    bq.SchemaField("feedback", "INTEGER"),
    bq.SchemaField("feedback_comment", "STRING"),
    bq.SchemaField("question_id", "STRING"),
]

_THREADS_SCHEMA = [
    bq.SchemaField("user_id", "STRING"),
    bq.SchemaField("is_team", "INTEGER"),
    bq.SchemaField("thread_id", "STRING"),
    bq.SchemaField("thread_title", "STRING"),
    bq.SchemaField("language", "STRING"),
    bq.SchemaField("deleted", "INTEGER"),
    bq.SchemaField("created_at_utc", "TIMESTAMP"),
    bq.SchemaField("num_questions", "INTEGER"),
    bq.SchemaField("charts_rendered", "INTEGER"),
    bq.SchemaField("exports_made", "INTEGER"),
    *(bq.SchemaField(f"exports_{fmt.lower()}", "INTEGER") for fmt in EXPORT_FORMATS),
]


def _write_ndjson(rows: list[dict], schema: list[bq.SchemaField], sink) -> None:
    """Write `rows` to `sink` as newline-delimited JSON, one row at a time.

    Streaming rather than building a second list keeps peak memory at one record:
    the export already holds every row once, and load_table_from_json would have
    copied the lot. Only datetimes need converting — BigQuery's JSON loader wants
    ISO strings for TIMESTAMP columns, and ours are naive UTC (which BigQuery reads
    as UTC). Python None (e.g. an unrated `feedback`) loads as NULL.
    """
    ts_fields = {f.name for f in schema if f.field_type == "TIMESTAMP"}
    for row in rows:
        record = {
            name: value.isoformat()
            if name in ts_fields and isinstance(value, datetime)
            else value
            for name, value in row.items()
        }
        sink.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        sink.write(b"\n")


def load_to_bigquery(
    dataset: str,
    question_rows: list[dict],
    thread_rows: list[dict],
    *,
    project: str | None = None,
) -> None:
    """Replace the `questions` and `threads` tables in `dataset` with these rows.

    Authenticates with Application Default Credentials (`gcloud auth` locally, or
    the job's service account in Cloud Run). The dataset must already exist — the
    script doesn't create it, so it makes no guess about its location.
    """
    client = bq.Client(project=project)
    dataset_ref = bq.DatasetReference(project or client.project, dataset)

    for table_name, schema, rows in (
        (QUESTIONS_TABLE, _QUESTIONS_SCHEMA, question_rows),
        (THREADS_TABLE, _THREADS_SCHEMA, thread_rows),
    ):
        # The optional `email` column (see fetch_rows --include-emails) sits right
        # after user_id; add it to the schema only when the rows actually carry it.
        if rows and "email" in rows[0]:
            schema = [schema[0], bq.SchemaField("email", "STRING"), *schema[1:]]

        table_ref = dataset_ref.table(table_name)
        job_config = bq.LoadJobConfig(
            schema=schema,
            write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
            source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        # A spooled file: small loads stay in memory, larger ones roll over to disk
        # instead of holding the whole serialized payload.
        with tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024) as payload:
            _write_ndjson(rows, schema, payload)
            payload.seek(0)
            job = client.load_table_from_file(payload, table_ref, job_config=job_config)
            job.result()  # wait for the load; raises on failure
        print(f"Loaded {len(rows)} rows into {table_ref}.")
