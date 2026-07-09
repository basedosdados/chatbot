import json
import re
from typing import get_args
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pytest_mock import MockerFixture

from app.agent.tools.bigquery import (
    EXPORT_FILENAME_MAX_LEN,
    EXPORT_FORMATS,
    MAX_BYTES_BILLED,
    ExportFormat,
    decode_table_values,
    execute_bigquery_sql,
    export_query_results,
)
from app.settings import settings


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


def test_export_formats_match_the_advertised_literal():
    """Every ExportFormat the model may pass must have an EXPORT_FORMATS entry.

    Guards against the schema (from the Literal) and the lookup table drifting
    apart, which would turn a valid file_format into a KeyError.
    """
    assert set(get_args(ExportFormat.__value__)) == set(EXPORT_FORMATS)


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
        assert re.fullmatch(r"q_[0-9a-f]{16}", output["query_ref"])

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


class TestExportQueryResults:
    """Tests for export_query_results tool."""

    DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}

    @pytest.fixture
    def mock_bq_client(self, mocker: MockerFixture):
        """BQ client wired for a successful extract path."""
        client = MagicMock(spec=bq.Client)
        client.extract_table.return_value = MagicMock()

        mocker.patch("app.agent.tools.bigquery._bq_client", return_value=client)
        return client

    def _state(
        self, query_ref: str = "q_test", destination: dict | None = None
    ) -> dict:
        """Build agent state holding one execute_bigquery_sql result."""
        return {
            "messages": [
                ToolMessage(
                    content="{}",
                    tool_call_id="prev",
                    name="execute_bigquery_sql",
                    artifact={
                        "type": "query_result",
                        "query_ref": query_ref,
                        "destination_table": destination or self.DESTINATION,
                    },
                )
            ]
        }

    def _invoke(
        self, args: dict, config: dict | None = None, state: dict | None = None
    ) -> ToolMessage:
        return _invoke_tool(
            export_query_results,
            {**args, "state": state if state is not None else self._state()},
            config=config,
        )

    def test_successful_export(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """A successful export extracts the referenced table and returns an artifact."""
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=1024)

        message = self._invoke(
            args={"query_ref": "q_test", "filename": "test-file"},
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

        # It extracts the exact table from the referenced query, without re-running SQL.
        assert mock_bq_client.query.call_count == 0
        source_table = mock_bq_client.extract_table.call_args.args[0]
        assert source_table == bq.TableReference.from_api_repr(self.DESTINATION)
        job_config = mock_bq_client.extract_table.call_args.kwargs["job_config"]
        assert job_config.destination_format == bq.DestinationFormat.CSV

    def test_successful_export_parquet(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """Export honours the requested file format."""
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=512)

        message = self._invoke(
            args={
                "query_ref": "q_test",
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
        job_config = mock_bq_client.extract_table.call_args.kwargs["job_config"]
        assert job_config.destination_format == bq.DestinationFormat.PARQUET

    def test_filename_too_long(self, mock_config: dict):
        """Test error when filename exceeds max length."""
        message = self._invoke(
            args={"query_ref": "q_test", "filename": "a" * 65},
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
            args={"query_ref": "q_test", "filename": "../etc/test-file"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            "`filename` must contain only letters, digits, hyphens, underscores, "
            "dots, and spaces (no path separators)."
        )
        assert message.artifact is None

    def test_unknown_query_ref(self, mock_config: dict, mock_bq_client: MagicMock):
        """Exporting a query_ref that isn't in state is a clear, actionable error."""
        message = self._invoke(
            args={"query_ref": "q_missing", "filename": "test"},
            config=mock_config,
            state=self._state(query_ref="q_other"),
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            "No query results found for query_ref 'q_missing'. Run the query with "
            "`execute_bigquery_sql` first, then export using the `query_ref` it returns."
        )
        assert message.artifact is None
        mock_bq_client.extract_table.assert_not_called()

    def test_source_table_expired(self, mock_config: dict, mock_bq_client: MagicMock):
        """A missing (expired) result table maps to a re-run-the-query message."""
        error = NotFound(
            message="Not found: Table p:d.t",
            errors=[{"reason": "notFound", "message": "Not found: Table p:d.t"}],
        )
        mock_bq_client.extract_table.return_value.result.side_effect = error

        message = self._invoke(
            args={"query_ref": "q_test", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == (
            "Those query results are no longer available to export (results are "
            "kept only temporarily). Please re-run the query and try again."
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
            args={"query_ref": "q_test", "filename": "test"},
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
            args={"query_ref": "q_test", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "Export completed but no file was written to GCS."
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
            args={"query_ref": "q_test", "filename": "test"},
            config=mock_config,
        )

        output = json.loads(message.content)

        assert output["status"] == "error"
        assert output["message"] == "400 Some other error"
        assert message.artifact is None

    def test_state_is_injected_by_the_tool_node(
        self, mocker: MockerFixture, mock_config: dict, mock_bq_client: MagicMock
    ):
        """End-to-end: LangGraph's ToolNode injects `state`; export resolves + extracts.

        The AIMessage tool call carries only the model-facing args (no `state`), so a
        pass here proves the query handle is wired through to the tool by the real
        injection path — the piece the other tests simulate by pre-merging `state`.
        """
        mocker.patch("app.agent.tools.bigquery.get_object_size", return_value=2048)

        prior_result = self._state()["messages"][0]
        model_turn = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "export_query_results",
                    "args": {"query_ref": "q_test", "filename": "my-data"},
                    "id": "call1",
                    "type": "tool_call",
                }
            ],
        )

        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([export_query_results]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        app = graph.compile()

        out = app.invoke(
            {"messages": [prior_result, model_turn]},
            config=mock_config,
        )

        message = out["messages"][-1]

        assert message.artifact["type"] == "file"
        assert message.artifact["metadata"]["filename"] == "my-data.csv"
        assert re.fullmatch(
            r"exports/test-thread/[0-9a-f]{32}\.csv",
            message.artifact["source"]["object_key"],
        )

        # ToolNode injected `state` (the tool call carried none), so resolution found
        # the exact result table and no SQL was re-run.
        source_table = mock_bq_client.extract_table.call_args.args[0]
        assert source_table == bq.TableReference.from_api_repr(self.DESTINATION)
        assert mock_bq_client.query.call_count == 0
