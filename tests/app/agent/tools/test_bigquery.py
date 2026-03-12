import json
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from pytest_mock import MockerFixture

from app.agent.tools.bigquery import (
    MAX_BYTES_BILLED,
    decode_table_values,
    execute_bigquery_sql,
)


class TestExecuteBigQuerySQL:
    """Tests for execute_bigquery_sql tool."""

    @pytest.fixture
    def mock_config(self) -> dict:
        return {"configurable": {"thread_id": "test-thread", "user_id": "test-user"}}

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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "SELECT * FROM project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"


class TestDecodeTableValues:
    """Tests for decode_table_values tool."""

    @pytest.fixture
    def mock_config(self) -> dict:
        return {"configurable": {"thread_id": "test-thread", "user_id": "test-user"}}

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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert len(output) == 2

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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
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
        # Verify column filter was added to query
        call_args = mock_bigquery_client.query.call_args[0][0]
        assert "nome_coluna = 'col1'" in call_args

    def test_dictionary_not_found(self, mocker: MockerFixture, mock_config: dict):
        """Test error when dictionary table doesn't exist."""
        error = NotFound(
            message="Table not found",
            errors=[{"reason": "notFound", "message": "Test message"}],
        )

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = error

        mocker.patch(
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
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
            "app.agent.tools.bigquery._get_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = json.loads(result)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"
