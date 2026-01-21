import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery as bq
from pydantic import ValidationError
from pytest_mock import MockerFixture

from app.agent.tools import (
    BDToolkit,
    ToolError,
    ToolOutput,
    decode_table_values,
    execute_bigquery_sql,
    get_dataset_details,
    handle_tool_errors,
    search_datasets,
)
from app.settings import settings


class TestHandleToolErrors:
    """Tests for handle_tool_errors decorator."""

    def test_decorator_passes_through_success(self):
        """Test decorator returns function result on success."""

        @handle_tool_errors
        def successful_function():
            return '{"status": "success", "results": "test results"}'

        output = ToolOutput.model_validate(json.loads(successful_function()))

        assert output.status == "success"
        assert output.results == "test results"
        assert output.error_details is None

    def test_decorator_catches_google_api_error(self):
        """Test decorator catches GoogleAPICallError."""

        @handle_tool_errors
        def failing_function():
            error = BadRequest(
                message="Some bad request",
                errors=[{"reason": "testReason", "message": "Test message"}],
            )
            raise error

        output = ToolOutput.model_validate(json.loads(failing_function()))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "testReason"
        assert output.error_details.message == "Test message"
        assert output.error_details.instructions is None

    def test_decorator_catches_google_api_error_without_errors(self):
        """Test decorator catches GoogleAPICallError."""

        @handle_tool_errors
        def failing_function():
            error = BadRequest(message="Some bad request")
            raise error

        output = ToolOutput.model_validate(json.loads(failing_function()))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type is None
        assert output.error_details.message == f"{BadRequest.code} Some bad request"
        assert output.error_details.instructions is None

    def test_decorator_catches_tool_error(self):
        """Test decorator catches ToolError."""

        @handle_tool_errors
        def failing_function():
            raise ToolError(
                "Custom error", error_type="CUSTOM", instructions="Try again"
            )

        output = ToolOutput.model_validate(json.loads(failing_function()))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "CUSTOM"
        assert output.error_details.message == "Custom error"
        assert output.error_details.instructions == "Try again"

    def test_decorator_catches_unexpected_exception(self):
        """Test decorator catches unexpected exceptions."""

        @handle_tool_errors
        def failing_function():
            raise ValueError("This is a value error")

        output = ToolOutput.model_validate(json.loads(failing_function()))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type is None
        assert output.error_details.message == "Unexpected error: This is a value error"
        assert output.error_details.instructions is None

    def test_decorator_with_custom_instructions(self):
        """Test decorator with custom instructions mapping."""

        @handle_tool_errors(instructions={"testReason": "Custom instruction"})
        def failing_function():
            error = BadRequest(
                message="Some bad request",
                errors=[{"reason": "testReason", "message": "Test message"}],
            )
            raise error

        output = ToolOutput.model_validate(json.loads(failing_function()))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "testReason"
        assert output.error_details.message == "Test message"
        assert output.error_details.instructions == "Custom instruction"


class TestToolOutput:
    """Tests for ToolOutput model validation."""

    def test_valid_success_output(self):
        """Test valid success output with results."""
        output = ToolOutput(status="success", results={"data": "test"})

        assert output.status == "success"
        assert output.results == {"data": "test"}
        assert output.error_details is None

    def test_valid_error_output(self):
        """Test valid success output with results."""
        from app.agent.tools import ErrorDetails

        error_details = ErrorDetails(message="error")

        output = ToolOutput(status="error", error_details=error_details)

        assert output.status == "error"
        assert output.results is None
        assert output.error_details == error_details

    def test_invalid_both_results_and_error(self):
        """Test validation fails when both results and error_details are set."""
        from app.agent.tools import ErrorDetails

        with pytest.raises(ValidationError):
            ToolOutput(
                status="error",
                results={"data": "test"},
                error_details=ErrorDetails(message="error"),
            )

    def test_invalid_neither_results_nor_error(self):
        """Test validation fails when neither results nor error_details are set."""
        with pytest.raises(ValidationError):
            ToolOutput(status="success", results=None, error_details=None)


class TestSearchDatasets:
    """Tests for search_datasets tool."""

    SEARCH_ENDPOINT = f"{settings.BASEDOSDADOS_BASE_URL}/search/"

    @respx.mock
    def test_search_datasets_returns_overviews(self):
        """Test successful dataset search."""
        mock_response = {
            "results": [
                {
                    "id": "dataset-1",
                    "name": "Test Dataset",
                    "slug": "test_dataset",
                    "description": "Dataset description",
                    "tags": [{"name": "tag1"}, {"name": "tag2"}],
                    "themes": [{"name": "theme1"}, {"name": "theme2"}],
                    "organizations": [{"name": "org1"}],
                }
            ]
        }

        respx.get(self.SEARCH_ENDPOINT).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = search_datasets.invoke({"query": "test"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert len(output.results) == 1

        dataset = output.results[0]

        assert dataset["id"] == "dataset-1"
        assert dataset["name"] == "Test Dataset"
        assert dataset["slug"] == "test_dataset"
        assert dataset["description"] == "Dataset description"
        assert dataset["tags"] == ["tag1", "tag2"]
        assert dataset["themes"] == ["theme1", "theme2"]
        assert dataset["organizations"] == ["org1"]

        assert output.error_details is None

    @respx.mock
    def test_search_datasets_returns_empty_results(self):
        """Test successful dataset search with no results."""
        respx.get(self.SEARCH_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        result = search_datasets.invoke({"query": "nonexistent"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert output.results == []
        assert output.error_details is None


class TestGetDatasetDetails:
    """Tests for get_dataset_details tool."""

    GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

    @pytest.fixture
    def mock_response(self):
        return {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "dataset-1",
                                "name": "Test Dataset",
                                "slug": "test_dataset",
                                "description": "Dataset description",
                                "tags": {"edges": [{"node": {"name": "tag1"}}]},
                                "themes": {"edges": [{"node": {"name": "theme1"}}]},
                                "organizations": {
                                    "edges": [
                                        {"node": {"name": "org1", "slug": "org1_slug"}}
                                    ]
                                },
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "table-1",
                                                "name": "Test Table",
                                                "slug": "test_table",
                                                "description": "Table description",
                                                "temporalCoverage": {
                                                    "start": "2020",
                                                    "end": "2023",
                                                },
                                                "cloudTables": {
                                                    "edges": [
                                                        {
                                                            "node": {
                                                                "gcpProjectId": "basedosdados",
                                                                "gcpDatasetId": "test_dataset",
                                                                "gcpTableId": "test_table",
                                                            }
                                                        }
                                                    ]
                                                },
                                                "columns": {
                                                    "edges": [
                                                        {
                                                            "node": {
                                                                "id": "col-1",
                                                                "name": "column_name",
                                                                "description": "Column description",
                                                                "bigqueryType": {
                                                                    "name": "COLUMN_TYPE"
                                                                },
                                                            }
                                                        }
                                                    ]
                                                },
                                            }
                                        }
                                    ]
                                },
                            }
                        }
                    ]
                }
            }
        }

    @respx.mock
    def test_get_dataset_details_success(self, mock_response):
        """Test successful dataset details retrieval."""
        # Mock graphql endpoint
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        # Mock usage guide (not found)
        respx.get(url__startswith="https://raw.githubusercontent.com").mock(
            return_value=httpx.Response(404)
        )

        result = get_dataset_details.invoke({"dataset_id": "dataset-1"})
        output = ToolOutput.model_validate(json.loads(result))

        dataset = output.results

        assert output.status == "success"
        assert dataset["id"] == "dataset-1"
        assert dataset["name"] == "Test Dataset"
        assert dataset["slug"] == "test_dataset"
        assert dataset["description"] == "Dataset description"
        assert dataset["tags"] == ["tag1"]
        assert dataset["themes"] == ["theme1"]
        assert dataset["organizations"] == ["org1"]
        assert dataset["usage_guide"] is None

        assert len(dataset["tables"]) == 1

        table = dataset["tables"][0]

        assert table["id"] == "table-1"
        assert table["gcp_id"] == "basedosdados.test_dataset.test_table"
        assert table["name"] == "Test Table"
        assert table["slug"] == "test_table"
        assert table["description"] == "Table description"
        assert table["temporal_coverage"] == {"start": "2020", "end": "2023"}

        assert len(table["columns"]) == 1

        column = table["columns"][0]

        assert column["name"] == "column_name"
        assert column["type"] == "COLUMN_TYPE"
        assert column["description"] == "Column description"

        assert output.error_details is None

    @respx.mock
    def test_get_dataset_details_success_with_usage_guide(self, mock_response):
        """Test dataset details with usage guide available."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        respx.get(url__startswith="https://raw.githubusercontent.com").mock(
            return_value=httpx.Response(200, text="# This is a usage guide.")
        )

        result = get_dataset_details.invoke({"dataset_id": "dataset-1"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert output.results["usage_guide"] == "# This is a usage guide."
        assert output.error_details is None

    @respx.mock
    def test_table_without_tags_themes_orgs(self):
        """Test dataset with table that has no tags, themes and orgs."""
        mock_response = {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "dataset-1",
                                "name": "Test Dataset",
                                "slug": "test_dataset",
                                "description": "Dataset description",
                                "tags": {"edges": [{"node": {}}]},
                                "themes": {"edges": [{"node": {}}]},
                                "organizations": {"edges": [{"node": {}}]},
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "table-1",
                                                "name": "Test Table",
                                                "slug": "test_table",
                                                "description": "Table description",
                                                "temporalCoverage": {
                                                    "start": "2020",
                                                    "end": "2023",
                                                },
                                                "cloudTables": {
                                                    "edges": [
                                                        {
                                                            "node": {
                                                                "gcpProjectId": "basedosdados",
                                                                "gcpDatasetId": "test_dataset",
                                                                "gcpTableId": "test_table",
                                                            }
                                                        }
                                                    ]
                                                },
                                                "columns": {
                                                    "edges": [
                                                        {
                                                            "node": {
                                                                "id": "col-1",
                                                                "name": "column_name",
                                                                "description": "Column description",
                                                                "bigqueryType": {
                                                                    "name": "COLUMN_TYPE"
                                                                },
                                                            }
                                                        }
                                                    ]
                                                },
                                            }
                                        }
                                    ]
                                },
                            }
                        }
                    ]
                }
            }
        }

        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        respx.get(url__startswith="https://raw.githubusercontent.com").mock(
            return_value=httpx.Response(200)
        )

        result = get_dataset_details.invoke({"dataset_id": "dataset-1"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert output.results["tags"] == []
        assert output.results["themes"] == []
        assert output.results["organizations"] == []
        assert output.error_details is None

    @respx.mock
    def test_table_without_cloud_tables(self):
        """Test dataset with table that has no cloud tables."""
        mock_response = {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "dataset-1",
                                "name": "Test Dataset",
                                "slug": "test_dataset",
                                "description": "Dataset description",
                                "tags": {"edges": [{"node": {"name": "tag1"}}]},
                                "themes": {"edges": [{"node": {"name": "theme1"}}]},
                                "organizations": {
                                    "edges": [
                                        {"node": {"name": "org1", "slug": "org1_slug"}}
                                    ]
                                },
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "table-1",
                                                "name": "Test Table",
                                                "slug": "test_table",
                                                "description": "Table description",
                                                "temporalCoverage": {
                                                    "start": "2020",
                                                    "end": "2023",
                                                },
                                                "cloudTables": {"edges": []},
                                                "columns": {
                                                    "edges": [
                                                        {
                                                            "node": {
                                                                "id": "col-1",
                                                                "name": "column_name",
                                                                "description": "Column description",
                                                                "bigqueryType": {
                                                                    "name": "COLUMN_TYPE"
                                                                },
                                                            }
                                                        }
                                                    ]
                                                },
                                            }
                                        }
                                    ]
                                },
                            }
                        }
                    ]
                }
            }
        }

        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = get_dataset_details.invoke({"dataset_id": "dataset-1"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert output.results["tables"][0]["gcp_id"] is None
        assert output.results["usage_guide"] is None
        assert output.error_details is None

    @respx.mock
    def test_get_dataset_details_dataset_not_found(self):
        """Test error when dataset is not found."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json={"data": {"allDataset": {"edges": []}}}
            )
        )

        result = get_dataset_details.invoke({"dataset_id": "nonexistent"})
        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.message == "Dataset nonexistent not found"
        assert output.error_details.error_type == "DATASET_NOT_FOUND"
        assert (
            output.error_details.instructions
            == "Verify the dataset ID from `search_datasets` results"
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
            "app.agent.tools.get_bigquery_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "SELECT * FROM project.dataset.table", "config": mock_config}
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert output.results == [{"col1": "value1"}, {"col1": "value2"}]
        assert output.error_details is None

    def test_forbidden_statement_type(self, mocker: MockerFixture, mock_config: dict):
        """Test error when statement is not SELECT."""
        mock_dry_run_query_job = MagicMock()
        mock_dry_run_query_job.statement_type = "DELETE"

        mock_bigquery_client = MagicMock(spec=bq.Client)
        mock_bigquery_client.query.return_value = mock_dry_run_query_job

        mocker.patch(
            "app.agent.tools.get_bigquery_client", return_value=mock_bigquery_client
        )

        result = execute_bigquery_sql.invoke(
            {"sql_query": "DELETE FROM project.dataset.table", "config": mock_config}
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "FORBIDDEN_STATEMENT"
        assert (
            output.error_details.message
            == "Query aborted: Statement DELETE is forbidden."
        )
        assert (
            output.error_details.instructions
            == "Your access is strictly read-only. Use only SELECT statements."
        )


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
            "app.agent.tools.get_bigquery_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
        assert len(output.results) == 2
        assert output.error_details is None

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
            "app.agent.tools.get_bigquery_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {
                "table_gcp_id": "project.dataset.table",
                "column_name": "col1",
                "config": mock_config,
            }
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "success"
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
            "app.agent.tools.get_bigquery_client", return_value=mock_bigquery_client
        )

        result = decode_table_values.invoke(
            {"table_gcp_id": "project.dataset.table", "config": mock_config}
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "notFound"
        assert output.error_details.message == "Test message"
        assert (
            output.error_details.instructions
            == "Dictionary table not found for this dataset."
        )

    def test_invalid_table_reference(self, mock_config: dict):
        """Test error when table reference format is invalid."""
        result = decode_table_values.invoke(
            {"table_gcp_id": "table", "config": mock_config}
        )

        output = ToolOutput.model_validate(json.loads(result))

        assert output.status == "error"
        assert output.results is None
        assert output.error_details.error_type == "INVALID_TABLE_REFERENCE"
        assert output.error_details.message == "Invalid table reference: 'table'"
        assert (
            output.error_details.instructions
            == "Provide a valid table reference in the format `project.dataset.table`"
        )


class TestBDToolkit:
    """Tests for BDToolkit class."""

    def test_get_tools_returns_all_tools(self):
        """Test that get_tools returns all expected tools."""
        tools = BDToolkit.get_tools()

        assert len(tools) == 4

        tool_names = [tool.name for tool in tools]

        assert "search_datasets" in tool_names
        assert "get_dataset_details" in tool_names
        assert "execute_bigquery_sql" in tool_names
        assert "decode_table_values" in tool_names
