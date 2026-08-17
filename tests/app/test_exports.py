from typing import get_args
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from pytest_mock import MockerFixture

from app.exports import (
    EXPORT_FORMATS,
    CollectedQueryHandle,
    ExportFormat,
    ResultTableExpired,
    ResultTooLarge,
    collect_query_handles,
    materialize_export,
    query_result_download,
    sanitize_export_filename,
)
from app.settings import settings

DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}


def test_export_formats_match_the_advertised_literal():
    """Every ExportFormat the API accepts must have an EXPORT_FORMATS entry.

    Guards the Literal and the lookup table against drifting apart, which would turn a
    valid file_format into a KeyError at materialisation.
    """
    assert set(get_args(ExportFormat.__value__)) == set(EXPORT_FORMATS)


def test_query_result_download_shape():
    """An executed query becomes one download offering the user-facing formats."""
    assert query_result_download("qr_1", "slug") == {
        "type": "query_result",
        "query_ref": "qr_1",
        "slug": "slug",
        "formats": ["AVRO", "CSV", "JSONL", "PARQUET"],
    }


class TestCollectQueryHandles:
    def test_collect_query_handles_returns_only_query_result_artifacts(self):
        """query_result artifacts are returned as handles; other artifacts are ignored."""
        artifacts = [
            {
                "type": "query_result",
                "query_ref": "qr_1",
                "slug": "slug",
                "destination_table": DESTINATION,
            },
            None,  # a tool with no artifact
            {"type": "file", "id": "x"},  # some other artifact kind
        ]

        assert collect_query_handles(artifacts) == [
            CollectedQueryHandle(
                query_ref="qr_1", slug="slug", destination_table=DESTINATION
            )
        ]

    def test_collect_query_handles_raises_on_malformed_query_result(self):
        """A query_result artifact missing a field is a producer bug — fail loud, don't skip."""
        with pytest.raises(KeyError):
            collect_query_handles([{"type": "query_result", "query_ref": "qr_1"}])


class TestMaterializeExport:
    """The shared service behind the download endpoint."""

    @staticmethod
    def _not_found() -> NotFound:
        return NotFound(
            "Not found: Table p:d.t",
            errors=[{"reason": "notFound", "message": "Not found: Table p:d.t"}],
        )

    @staticmethod
    def _client(num_bytes: int = 1024) -> MagicMock:
        """A BigQuery client whose result table reports `num_bytes`."""
        client = MagicMock(spec=bq.Client)
        client.get_table.return_value.num_bytes = num_bytes
        return client

    def _materialize(self, **overrides):
        return materialize_export(
            **{
                "query_ref": "qr_1",
                "destination_table": DESTINATION,
                "file_format": "CSV",
                "filename": "resultados",
                "message_id": "m1",
                **overrides,
            }
        )

    def test_extracts_to_deterministic_key(self, mocker: MockerFixture):
        """First download extracts the exact table to a deterministic per-query key."""
        client = self._client()
        client.extract_table.return_value = MagicMock()
        mocker.patch("app.exports._bq_client", return_value=client)
        # Object absent on the reuse check, then sized once the extract has written it.
        mocker.patch("app.exports.get_object_size", side_effect=[None, 1024])

        exported = self._materialize()

        assert exported.bucket == settings.GOOGLE_GCS_BUCKET
        assert exported.object_key == "query_results/m1/qr_1.csv"
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
        client = self._client()
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=512)

        exported = self._materialize(file_format="PARQUET")

        assert exported.object_key == "query_results/m1/qr_1.parquet"
        assert exported.mime_type == "application/vnd.apache.parquet"
        assert exported.size_bytes == 512
        client.extract_table.assert_not_called()

    def test_expired_table_raises_result_table_expired(self, mocker: MockerFixture):
        """An expired result table surfaces as ResultTableExpired (endpoint -> 410)."""
        client = self._client()
        client.extract_table.return_value.result.side_effect = self._not_found()
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(ResultTableExpired):
            self._materialize()

        assert client.query.call_count == 0

    def test_oversized_table_is_rejected_before_extracting(self, mocker: MockerFixture):
        """A table over MAX_EXPORT_BYTES is refused upfront, without an extract job."""
        client = self._client(num_bytes=settings.MAX_EXPORT_BYTES + 1)
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(ResultTooLarge):
            self._materialize()

        client.extract_table.assert_not_called()

    def test_expired_table_on_the_size_check_raises_result_table_expired(
        self, mocker: MockerFixture
    ):
        """The size lookup touches the expired table first — still a 410, not a 500."""
        client = self._client()
        client.get_table.side_effect = self._not_found()
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(ResultTableExpired):
            self._materialize()

        client.extract_table.assert_not_called()

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
        client = self._client()
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
        client = self._client()
        client.extract_table.return_value.result.side_effect = error
        mocker.patch("app.exports._bq_client", return_value=client)
        mocker.patch("app.exports.get_object_size", return_value=None)

        with pytest.raises(BadRequest):
            self._materialize()

    def test_object_missing_after_extract_raises_runtime_error(
        self, mocker: MockerFixture
    ):
        """An extract that reports success but writes nothing is a RuntimeError."""
        client = self._client()
        client.extract_table.return_value = MagicMock()
        mocker.patch("app.exports._bq_client", return_value=client)
        # Absent before the extract and still absent after (nothing written).
        mocker.patch("app.exports.get_object_size", side_effect=[None, None])

        with pytest.raises(RuntimeError, match="no file was written"):
            self._materialize()


class TestSanitizeExportFilename:
    """The download filename guard, shared by the export tool and the endpoint."""

    @pytest.mark.parametrize(
        ("slug", "expected"),
        [
            # A clean slug (what the model is asked to produce) passes through unchanged.
            ("vendas_por_ano", "vendas_por_ano"),
            # Hyphens and digits are allowed.
            ("ideb-2021", "ideb-2021"),
            # Spaces and punctuation collapse to a single underscore.
            ("Vendas por ano", "Vendas_por_ano"),
            ("a   b", "a_b"),
            ("café & leite!", "café_leite"),
            # Leading/trailing separators are stripped, not left dangling.
            ("_vendas_", "vendas"),
            ("  vendas  ", "vendas"),
            # Path separators and traversal are neutralized (no slashes or dots survive).
            ("../../etc/passwd", "etc_passwd"),
            ("relatorio/2021", "relatorio_2021"),
            # File extensions are neutralized.
            ("vendas_por_ano.csv", "vendas_por_ano_csv"),
            # Accented word characters are preserved (\w is unicode).
            ("população", "população"),
        ],
    )
    def test_sanitizes_slug(self, slug: str, expected: str):
        assert sanitize_export_filename(slug, "resultados") == expected

    @pytest.mark.parametrize("slug", ["", "   ", "!!!", "/", "..."])
    def test_falls_back_when_nothing_usable_remains(self, slug: str):
        """A slug that sanitizes to empty falls back to the provided fallback, never ''."""
        assert sanitize_export_filename(slug, "resultados") == "resultados"
