import json
import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from pytest_mock import MockerFixture

from app.agent.tools.bigquery import (
    EXPORT_FILENAME_MAX_LEN,
    MAX_BYTES_BILLED,
    decode_table_values,
    execute_bigquery_sql,
    export_query_results,
)
from app.settings import settings


@pytest.fixture
def mock_config() -> dict:
    return {"configurable": {"thread_id": "test-thread", "user_id": "test-user"}}


class TestExecuteBigQuerySQL:
    """Tests for execute_bigquery_sql tool."""

    def test_successful_query(self, mocker: MockerFixture, mock_config: dict):
        """Test successful SELECT query execution."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [{"col1": "value1"}, {"col1": "value2"}]

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [
            mock_dry_run_query_job,
            mock_query_job,
        ]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "SELECT * FROM project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output == [{"col1": "value1"}, {"col1": "value2"}]

    def test_forbidden_statement_type(self, mocker: MockerFixture, mock_config: dict):
        """Test error when statement is not SELECT."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "DELETE"

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_dry_run_query_job

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "DELETE FROM project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "Only SELECT statements are allowed, got DELETE."

    def test_bytes_billed_limit_exceeded(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """Test error when query exceeds bytes billed limit."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        error = BadRequest(
            message="Query limit exceeded",
            errors=[
                {"reason": "bytesBilledLimitExceeded", "message": "Limit exceeded"}
            ],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [mock_dry_run_query_job, error]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "SELECT * FROM project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == (
            f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. "
            "Add WHERE filters or select fewer columns."
        )

    def test_google_api_error_reraise(self, mocker: MockerFixture, mock_config: dict):
        """Test that non-bytesBilledLimitExceeded GoogleAPICallError is re-raised."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        error = BadRequest(
            message="Syntax error",
            errors=[{"reason": "testReason", "message": "Test message"}],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [mock_dry_run_query_job, error]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "SELECT * FROM project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"


class TestDecodeTableValues:
    """Tests for decode_table_values tool."""

    def test_decode_all_columns(self, mocker: MockerFixture, mock_config: dict):
        """Test decoding all columns from a table."""
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [
            {"nome_coluna": "col1", "chave": "1", "valor": "Value 1"},
            {"nome_coluna": "col2", "chave": "2", "valor": "Value 2"},
        ]

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_query_job

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert len(output) == 2

        call_args = mock_bigquery_client.query.call_args[0][0]
        assert "id_tabela = @table_name" in call_args
        assert "nome_coluna = @column_name" not in call_args

        job_config = mock_bigquery_client.query.call_args[1]["job_config"]
        param_names = {p.name for p in job_config.query_parameters}
        assert "table_name" in param_names
        assert "nome_coluna = " not in param_names

    def test_decode_all_columns_with_backticks(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """Test decoding all columns from a table with backticks in its name."""
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [
            {"nome_coluna": "col1", "chave": "1", "valor": "Value 1"},
            {"nome_coluna": "col2", "chave": "2", "valor": "Value 2"},
        ]

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_query_job

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "`project.dataset.table`", "config": mock_config}
        )

        output = json.loads(result)

        assert len(output) == 2

        call_args = mock_bigquery_client.query.call_args[0][0]
        assert "id_tabela = @table_name" in call_args
        assert "nome_coluna = @column_name" not in call_args

        job_config = mock_bigquery_client.query.call_args[1]["job_config"]
        param_names = {p.name for p in job_config.query_parameters}
        assert "table_name" in param_names
        assert "nome_coluna = " not in param_names

    def test_decode_specific_column(self, mocker: MockerFixture, mock_config: dict):
        """Test decoding a specific column."""
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [
            {"nome_coluna": "col1", "chave": "1", "valor": "Value 1"},
            {"nome_coluna": "col1", "chave": "2", "valor": "Value 2"},
        ]

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_query_job

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {
                "table_gcp_id": "project.dataset.table",
                "column_name": "col1",
                "config": mock_config,
            }
        )

        output = json.loads(result)

        assert len(output) == 2

        call_args = mock_bigquery_client.query.call_args[0][0]
        assert "id_tabela = @table_name" in call_args
        assert "nome_coluna = @column_name" in call_args

        job_config = mock_bigquery_client.query.call_args[1]["job_config"]
        param_names = {p.name for p in job_config.query_parameters}
        assert "table_name" in param_names
        assert "column_name" in param_names

    def test_dictionary_not_found(self, mocker: MockerFixture, mock_config: dict):
        """Test error when dictionary table doesn't exist."""
        error = NotFound(
            message="Table not found",
            errors=[{"reason": "notFound", "message": "Test message"}],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = error

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "Dictionary table not found for this dataset."

    def test_invalid_table_reference(self, mock_config: dict):
        """Test error when table reference format is invalid."""
        result = decode_table_values.invoke(
            {"table_gcp_id": "table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert (
            output["message"]
            == "Invalid table reference: 'table'. Expected format: project.dataset.table"
        )

    def test_google_api_error_reraise(self, mocker: MockerFixture, mock_config: dict):
        """Test that non-notFound GoogleAPICallError is re-raised."""
        error = BadRequest(
            message="Syntax error",
            errors=[{"reason": "testReason", "message": "Test message"}],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = error

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"


class TestExportQueryResults:
    """Tests for export_query_results tool."""

    @pytest.fixture
    def mock_bq_client(self, mocker: MockerFixture):
        """BQ client wired for a successful export path."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        mock_query_job = MagicMock()
        mock_query_job.destination = MagicMock()

        client = MagicMock(spec=bq.Client)
        client.query.side_effect = [mock_dry_run_query_job, mock_query_job]
        client.extract_table.return_value = MagicMock()

        mocker.patch("app.agent.tools.bigquery._bq_client", return_value=client)
        return client

    def _invoke(self, args: dict, config: dict | None = None) -> Any:
        return export_query_results.invoke(
            {
                "id": "1",
                "args": args,
                "type": "tool_call",
                "name": "export_query_results",
            },
            config=config,
        )

    def test_successful_export(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Test successful export returns content and artifact."""
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=1024)

        message = self._invoke(
            args={
                "sql_query": "SELECT * FROM project.dataset.table",
                "filename": "test-file",
            },
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "success"
        assert output["filename"] == "test-file.csv"
        assert message.artifact["metadata"]["filename"] == "test-file.csv"
        assert message.artifact["metadata"]["mime_type"] == "text/csv"
        assert message.artifact["metadata"]["size_bytes"] == 1024
        assert message.artifact["source"]["bucket"] == settings.GOOGLE_GCS_BUCKET
        assert re.fullmatch(
            r"exports/test-thread/[0-9a-f]{32}\.csv",
            message.artifact["source"]["object_key"],
        )

    def test_successful_export_parquet(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Test successful export with PARQUET format."""
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=512)

        message = self._invoke(
            args={
                "sql_query": "SELECT * FROM project.dataset.table",
                "filename": "test-file",
                "file_format": "PARQUET",
            },
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "success"
        assert output["filename"] == "test-file.parquet"
        assert message.artifact["metadata"]["filename"] == "test-file.parquet"
        assert (
            message.artifact["metadata"]["mime_type"]
            == "application/vnd.apache.parquet"
        )
        assert message.artifact["metadata"]["size_bytes"] == 512
        assert message.artifact["source"]["bucket"] == settings.GOOGLE_GCS_BUCKET
        assert re.fullmatch(
            r"exports/test-thread/[0-9a-f]{32}\.parquet",
            message.artifact["source"]["object_key"],
        )

    def test_filename_too_long(self, mock_config: dict):
        """Test error when filename exceeds max length."""
        message = self._invoke(
            args={"sql_query": "SELECT 1", "filename": "a" * 65},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            f"`filename` must be at most {EXPORT_FILENAME_MAX_LEN} characters."
        )
        assert message.artifact is None

    def test_filename_invalid_characters(self, mock_config: dict):
        """Test error when filename contains invalid characters."""
        message = self._invoke(
            args={"sql_query": "SELECT 1", "filename": "../etc/test-file"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            "`filename` must contain only letters, digits, hyphens, underscores, "
            "dots, and spaces (no path separators)."
        )
        assert message.artifact is None

    def test_forbidden_statement_type(self, mocker: MockerFixture, mock_config: dict):
        """Test error when statement is not SELECT."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "DELETE"

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_dry_run_query_job

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = self._invoke(
            args={"sql_query": "DELETE FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "Only SELECT statements are allowed, got DELETE."
        assert message.artifact is None

    def test_bytes_billed_limit_exceeded(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """Test error when query exceeds bytes billed limit."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        error = BadRequest(
            message="Query limit exceeded",
            errors=[
                {"reason": "bytesBilledLimitExceeded", "message": "Limit exceeded"}
            ],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [mock_dry_run_query_job, error]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = self._invoke(
            args={"sql_query": "SELECT * FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            f"Export exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. "
            "Add WHERE filters or select fewer columns."
        )
        assert message.artifact is None

    def test_result_too_large_for_single_file(
        self, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Test error when extract_table fails because result set is too large."""
        error = BadRequest(
            message="Table too large",
            errors=[
                {
                    "reason": "invalid",
                    "message": "...table too large to be exported to a single file...",
                }
            ],
        )
        mock_bq_client.extract_table.return_value.result.side_effect = error

        message = self._invoke(
            args={"sql_query": "SELECT * FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            "Result set is too large to export as a single file. "
            "Add WHERE filters, select fewer columns, or limit the number of rows."
        )
        assert message.artifact is None

    def test_gcs_object_not_written(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Test error when GCS object is missing after extract completes."""
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=None)

        message = self._invoke(
            args={"sql_query": "SELECT * FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "Export completed but no file was written to GCS."
        assert message.artifact is None

    def test_query_google_api_error_reraise(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """Test that non-bytesBilledLimitExceeded GoogleAPICallError is re-raised."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        error = BadRequest(
            message="Syntax error",
            errors=[{"reason": "testReason", "message": "Test message"}],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [mock_dry_run_query_job, error]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = self._invoke(
            args={"sql_query": "SELECT * FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"
        assert message.artifact is None

    def test_extract_google_api_error_reraise(
        self, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Test that non-'too large' GoogleAPICallError from extract is re-raised."""
        error = BadRequest(
            message="Some other error",
            errors=[{"reason": "testReason", "message": "Test message"}],
        )
        mock_bq_client.extract_table.return_value.result.side_effect = error

        message = self._invoke(
            args={"sql_query": "SELECT * FROM t", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "400 Some other error"
        assert message.artifact is None
