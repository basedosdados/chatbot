import json
import re
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from langchain_core.messages import ToolMessage
from pytest_mock import MockerFixture

from app.agent.tools.bigquery import (
    MAX_BYTES_BILLED,
    decode_table_values,
    execute_bigquery_sql,
)


@pytest.fixture
def mock_config() -> dict:
    return {"configurable": {"thread_id": "test-thread", "user_id": "test-user"}}


def _invoke_tool(tool, args: dict, config: dict | None = None) -> ToolMessage:
    """Invoke a tool the way the agent's ToolNode does, returning the ToolMessage.

    The tool-call form (rather than a plain args dict) exercises the real
    content+artifact packaging, so the ToolMessage exposes both `.content` and,
    for content_and_artifact tools, `.artifact`.
    """
    return tool.invoke(
        {"type": "tool_call", "id": "1", "name": tool.name, "args": args},
        config=config,
    )


class TestExecuteBigQuerySQL:
    """Tests for execute_bigquery_sql tool."""

    def test_successful_query(self, mocker: MockerFixture, mock_config: dict):
        """Test successful SELECT query returns rows plus a query_ref handle."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [{"col1": "value1"}, {"col1": "value2"}]
        mock_query_job.destination.to_api_repr.return_value = {
            "projectId": "p",
            "datasetId": "d",
            "tableId": "t",
        }

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [
            mock_dry_run_query_job,
            mock_query_job,
        ]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "SELECT * FROM project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["results"] == [{"col1": "value1"}, {"col1": "value2"}]
        assert output["row_count"] == 2
        assert re.fullmatch(r"qr_[0-9a-f]{8}", output["query_ref"])

    def test_successful_query_exposes_destination_table_on_artifact(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """The result table reference is carried on the artifact, not the content."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [{"col1": "value1"}]
        mock_query_job.destination.to_api_repr.return_value = {
            "projectId": "p",
            "datasetId": "d",
            "tableId": "t",
        }

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [
            mock_dry_run_query_job,
            mock_query_job,
        ]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "SELECT * FROM project.dataset.table"},
            config=mock_config,
        )

        content_query_ref = json.loads(message.content)["query_ref"]

        assert message.artifact["type"] == "query_result"
        assert message.artifact["query_ref"] == content_query_ref
        assert message.artifact["destination_table"] == {
            "projectId": "p",
            "datasetId": "d",
            "tableId": "t",
        }

    def test_successful_query_empty_result(
        self, mocker: MockerFixture, mock_config: dict
    ):
        """A query with no rows returns a message and no downloadable handle."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "SELECT"

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.side_effect = [
            mock_dry_run_query_job,
            mock_query_job,
        ]

        mocker.patch(
            "app.agent.tools.bigquery._bq_client", return_value=mock_bigquery_client
        )

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "SELECT * FROM project.dataset.table"},
            config=mock_config,
        )

        assert (
            json.loads(message.content)
            == "Query returned 0 rows. Review the filters per the empty-result protocol."
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

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "DELETE FROM project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "SELECT * FROM project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            f"Query exceeds the {MAX_BYTES_BILLED // 10**9}GB processing limit. "
            "Filter by partitioned columns."
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

        message = _invoke_tool(
            execute_bigquery_sql,
            {"sql_query": "SELECT * FROM project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            decode_table_values,
            {"table_gcp_id": "project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            decode_table_values,
            {"table_gcp_id": "`project.dataset.table`"},
            config=mock_config,
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            decode_table_values,
            {"table_gcp_id": "project.dataset.table", "column_name": "col1"},
            config=mock_config,
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            decode_table_values,
            {"table_gcp_id": "project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "Dictionary table not found for this dataset."

    def test_invalid_table_reference(self, mock_config: dict):
        """Test error when table reference format is invalid."""
        message = _invoke_tool(
            decode_table_values, {"table_gcp_id": "table"}, config=mock_config
        )

        output = json.loads(message.content)

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

        message = _invoke_tool(
            decode_table_values,
            {"table_gcp_id": "project.dataset.table"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "400 Syntax error"
