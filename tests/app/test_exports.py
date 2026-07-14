from typing import get_args
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from pytest_mock import MockerFixture

from app.exports import (
    EXPORT_FORMATS,
    SUPPORTED_EXPORT_FORMATS,
    ExportFormat,
    ResultTableExpired,
    ResultTooLarge,
    collect_query_handles,
    derive_downloads,
    materialize_export,
    sanitize_sql_query_refs,
)
from app.settings import settings

DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}


def test_export_formats_match_the_advertised_literal():
    """Every ExportFormat the API accepts must have an EXPORT_FORMATS entry.

    Guards the Literal and the lookup table against drifting apart, which would turn a
    valid file_format into a KeyError at materialisation.
    """
    assert set(get_args(ExportFormat.__value__)) == set(EXPORT_FORMATS)


def test_collect_query_handles_maps_only_query_result_artifacts():
    """query_result artifacts become {query_ref: destination_table}; the rest are ignored."""
    artifacts = [
        {"type": "query_result", "query_ref": "qr_1", "destination_table": DESTINATION},
        None,  # a tool with no artifact
        {"type": "file", "id": "x"},  # some other artifact kind
        {
            "type": "query_result",
            "query_ref": "qr_2",
        },  # no destination_table -> skipped
    ]

    assert collect_query_handles(artifacts) == {"qr_1": DESTINATION}


class TestSanitizeSqlQueryRefs:
    """Only refs that actually executed this run may survive as download handles."""

    def test_keeps_executed_refs_and_nulls_the_rest(self):
        structured = {
            "sql_queries": [
                {"sql": "a", "query_ref": "qr_1"},
                {"sql": "b", "query_ref": "qr_hallucinated"},
            ]
        }

        cleaned = sanitize_sql_query_refs(structured, {"qr_1"})

        # Pure: the input is left untouched ...
        assert structured["sql_queries"][1]["query_ref"] == "qr_hallucinated"
        # ... and the cleaned copy nulls the ref that did not execute.
        assert cleaned["sql_queries"][0]["query_ref"] == "qr_1"
        assert cleaned["sql_queries"][1]["query_ref"] is None

    def test_none_passes_through(self):
        assert sanitize_sql_query_refs(None, {"qr_1"}) is None

    def test_nothing_to_clean_returns_input_unchanged(self):
        no_queries = {"response": "x"}
        empty = {"sql_queries": []}

        assert sanitize_sql_query_refs(no_queries, set()) is no_queries
        assert sanitize_sql_query_refs(empty, set()) is empty


class TestDeriveDownloads:
    """The download affordance is one item per backing query with a real ref."""

    @staticmethod
    def _query_result(query_ref: str) -> dict:
        return {
            "type": "query_result",
            "query_ref": query_ref,
            "formats": SUPPORTED_EXPORT_FORMATS,
        }

    def test_one_download_per_backing_query(self):
        structured = {
            "sql_queries": [
                {"sql": "a", "query_ref": "qr_1"},
                {"sql": "b", "query_ref": "qr_2"},
            ]
        }

        assert derive_downloads(structured) == [
            self._query_result("qr_1"),
            self._query_result("qr_2"),
        ]

    def test_ignores_null_refs(self):
        structured = {
            "sql_queries": [
                {"sql": "a", "query_ref": "qr_1"},
                {"sql": "b", "query_ref": None},
            ]
        }

        assert derive_downloads(structured) == [self._query_result("qr_1")]

    def test_empty_when_nothing_downloadable(self):
        assert derive_downloads({"sql_queries": []}) == []
        assert (
            derive_downloads({"sql_queries": [{"sql": "a", "query_ref": None}]}) == []
        )
        assert derive_downloads(None) == []


class TestMaterializeExport:
    """The shared service behind the download endpoint."""

    @staticmethod
    def _not_found() -> NotFound:
        return NotFound(
            "Not found: Table p:d.t",
            errors=[{"reason": "notFound", "message": "Not found: Table p:d.t"}],
        )

    def _materialize(self, **overrides):
        return materialize_export(
            **{
                "query_ref": "qr_1",
                "destination_table": DESTINATION,
                "file_format": "CSV",
                "filename": "resultados",
                "thread_id": "t1",
                **overrides,
            }
        )

    def test_extracts_to_deterministic_key(self, mocker: MockerFixture):
        """First download extracts the exact table to a deterministic per-query key."""
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value = MagicMock()
        mocker.patch("app.exports._bq_client", return_value=client)
        # Object absent on the reuse check, then sized once the extract has written it.
        mocker.patch("app.exports.get_object_size", side_effect=[None, 1024])

        exported = self._materialize()

        assert exported.bucket == settings.GOOGLE_GCS_BUCKET
        assert exported.object_key == "exports/t1/qr_1.csv"
        assert exported.filename == "resultados.csv"
        assert exported.mime_type == "text/csv"
        assert exported.size_bytes == 1024

        # Extracts the exact referenced table, never re-running SQL.
        assert client.query.call_count == 0
        source = client.extract_table.call_args.args[0]
        assert source == bq.TableReference.from_api_repr(DESTINATION)
        job_config = client.extract_table.call_args.kwargs["job_config"]
        assert job_config.destination_format == bq.DestinationFormat.CSV

    def test_reuses_existing_object(self, mocker: MockerFixture):
        """A repeat download of the same query+format reuses the object, no extract."""
        client = MagicMock(spec=bq.Client)
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=512)

        exported = self._materialize(file_format="PARQUET")

        assert exported.object_key == "exports/t1/qr_1.parquet"
        assert exported.mime_type == "application/vnd.apache.parquet"
        assert exported.size_bytes == 512
        client.extract_table.assert_not_called()

    def test_expired_table_raises_result_table_expired(self, mocker: MockerFixture):
        """An expired result table surfaces as ResultTableExpired (endpoint -> 410)."""
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value.result.side_effect = self._not_found()
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(ResultTableExpired):
            self._materialize()

        assert client.query.call_count == 0

    def test_too_large_raises_result_too_large(self, mocker: MockerFixture):
        """A too-large result set surfaces as ResultTooLarge (endpoint -> 400)."""
        error = BadRequest(
            "Table too large",
            errors=[
                {
                    "reason": "invalid",
                    "message": "...table too large to be exported to a single file...",
                }
            ],
        )
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value.result.side_effect = error
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(ResultTooLarge):
            self._materialize()

    def test_other_extract_error_is_reraised(self, mocker: MockerFixture):
        """A non-'too large' GoogleAPICallError from the extract is re-raised as-is."""
        error = BadRequest(
            "Some other error",
            errors=[{"reason": "otherReason", "message": "Some other error"}],
        )
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value.result.side_effect = error
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(BadRequest):
            self._materialize()

    def test_object_missing_after_extract_raises_runtime_error(
        self, mocker: MockerFixture
    ):
        """An extract that reports success but writes nothing is a RuntimeError."""
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value = MagicMock()
        mocker.patch("app.exports._bq_client", return_value=client)
        # Absent before the extract and still absent after (nothing written).
        mocker.patch("app.exports.get_object_size", side_effect=[None, None])

        with pytest.raises(RuntimeError, match="no file was written"):
            self._materialize()
