import json

import httpx
import pytest
import respx
from langchain.tools import ToolRuntime

from app.agent.context import AgentContext
from app.agent.tools.api import (
    BASE_USAGE_GUIDE_URL,
    SKIP_DIRECTORY_DATASETS,
    _fetch_usage_guide,
    get_dataset_details,
    get_table_details,
    search_datasets,
)
from app.settings import settings


def _build_tool_runtime(context: AgentContext) -> ToolRuntime[AgentContext]:
    """Build a ToolRuntime the way the agent's ToolNode injects it into a tool.

    Only `context` varies between tests; the other slots are inert stand-ins for
    graph plumbing the tools under test never read.
    """
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=None,
        tool_call_id="test-tool-call",
        store=None,
    )


def _runtime(language: str = "pt") -> ToolRuntime[AgentContext]:
    """A ToolRuntime for the given language (thread/user ids are irrelevant here)."""
    return _build_tool_runtime(
        AgentContext(
            thread_id="test-thread",
            user_id="test-user",
            language=language,
        )
    )


class TestSearchDatasets:
    """Tests for search_datasets tool."""

    SEARCH_ENDPOINT = f"{settings.BASEDOSDADOS_BASE_URL}/search/"

    @respx.mock
    async def test_search_datasets_returns_overviews(self):
        """Test successful dataset search."""
        # The /search/ endpoint is locale-aware server-side, so the response
        # already carries name/description/tags/themes/organizations in the
        # requested locale — the tool reads them verbatim.
        mock_response = {
            "results": [
                {
                    "id": "dataset-1",
                    "name": "Test Dataset",
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

        result = await search_datasets.ainvoke({"query": "test", "runtime": _runtime()})
        output = json.loads(result)

        assert len(output) == 1

        dataset = output[0]

        assert dataset["id"] == "dataset-1"
        assert dataset["name"] == "Test Dataset"
        assert dataset["description"] == "Dataset description"
        assert dataset["tags"] == ["tag1", "tag2"]
        assert dataset["themes"] == ["theme1", "theme2"]
        assert dataset["organizations"] == ["org1"]

    @respx.mock
    async def test_search_datasets_returns_empty_results(self):
        """Test successful dataset search with no results."""
        respx.get(self.SEARCH_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        result = await search_datasets.ainvoke(
            {"query": "nonexistent", "runtime": _runtime()}
        )
        output = json.loads(result)

        assert output == []

    @respx.mock
    async def test_forwards_context_language_as_locale(self):
        """The thread's language is sent as the `locale` so the API returns localized text."""
        route = respx.get(self.SEARCH_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        await search_datasets.ainvoke({"query": "x", "runtime": _runtime("es")})

        assert route.calls.last.request.url.params["locale"] == "es"


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
                                "id": "DatasetNode:dataset-1",
                                "namePt": "Test Dataset",
                                "descriptionPt": "Dataset description",
                                "tags": {"edges": [{"node": {"namePt": "tag1"}}]},
                                "themes": {"edges": [{"node": {"namePt": "theme1"}}]},
                                "organizations": {
                                    "edges": [
                                        {
                                            "node": {
                                                "namePt": "org1",
                                                "slug": "org1_slug",
                                            }
                                        }
                                    ]
                                },
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "TableNode:table-1",
                                                "namePt": "Test Table",
                                                "descriptionPt": "Table description",
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
    async def test_get_dataset_details_success(self, mock_response):
        """Test successful dataset details retrieval."""
        # Mock graphql endpoint
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        # Mock usage guide (not found)
        respx.get(url__startswith="https://raw.githubusercontent.com").mock(
            return_value=httpx.Response(404)
        )

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "dataset-1", "runtime": _runtime()}
        )
        dataset = json.loads(result)

        assert dataset["id"] == "dataset-1"
        assert dataset["name"] == "Test Dataset"
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
        assert table["description"] == "Table description"

    @respx.mock
    async def test_get_dataset_details_success_with_usage_guide(self, mock_response):
        """Test dataset details with usage guide available."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        respx.get(url__startswith="https://raw.githubusercontent.com").mock(
            return_value=httpx.Response(200, text="# This is a usage guide.")
        )

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "dataset-1", "runtime": _runtime()}
        )
        dataset = json.loads(result)

        assert dataset["usage_guide"] == "# This is a usage guide."

    @respx.mock
    async def test_get_dataset_details_without_tags_themes_orgs(self):
        """Test dataset details that has no tags, themes and orgs."""
        mock_response = {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "dataset-1",
                                "namePt": "Test Dataset",
                                "descriptionPt": "Dataset description",
                                "tags": {"edges": [{"node": {}}]},
                                "themes": {"edges": [{"node": {}}]},
                                "organizations": {"edges": [{"node": {}}]},
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "table-1",
                                                "namePt": "Test Table",
                                                "descriptionPt": "Table description",
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

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "dataset-1", "runtime": _runtime()}
        )
        dataset = json.loads(result)

        assert dataset["tags"] == []
        assert dataset["themes"] == []
        assert dataset["organizations"] == []

    @respx.mock
    async def test_table_without_cloud_tables(self):
        """Test dataset details with table that has no cloud tables."""
        mock_response = {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "dataset-1",
                                "namePt": "Test Dataset",
                                "slug": "test_dataset",
                                "descriptionPt": "Dataset description",
                                "tags": {"edges": [{"node": {"namePt": "tag1"}}]},
                                "themes": {"edges": [{"node": {"namePt": "theme1"}}]},
                                "organizations": {
                                    "edges": [
                                        {
                                            "node": {
                                                "namePt": "org1",
                                                "slug": "org1_slug",
                                            }
                                        }
                                    ]
                                },
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "table-1",
                                                "namePt": "Test Table",
                                                "slug": "test_table",
                                                "descriptionPt": "Table description",
                                                "temporalCoverage": {
                                                    "start": "2020",
                                                    "end": "2023",
                                                },
                                                "cloudTables": {"edges": []},
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

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "dataset-1", "runtime": _runtime()}
        )
        dataset = json.loads(result)

        assert dataset["tables"][0]["gcp_id"] is None
        assert dataset["usage_guide"] is None

    @respx.mock
    async def test_get_dataset_details_dataset_not_found(self):
        """Test error when dataset is not found."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json={"data": {"allDataset": {"edges": []}}}
            )
        )

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "nonexistent", "runtime": _runtime()}
        )
        output = json.loads(result)

        assert output["status"] == "error"
        assert (
            output["message"]
            == "Dataset 'nonexistent' not found. Verify the dataset ID from search_datasets results."
        )

    @respx.mock
    async def test_localizes_metadata_to_context_language(self):
        """Each field is picked in the thread's language, falling back to pt per field."""
        mock_response = {
            "data": {
                "allDataset": {
                    "edges": [
                        {
                            "node": {
                                "id": "DatasetNode:dataset-1",
                                "namePt": "Conjunto",
                                "nameEn": "Dataset",
                                # description has no English value -> must fall back to pt
                                "descriptionPt": "Descrição",
                                "descriptionEn": "",
                                "tags": {
                                    "edges": [
                                        {
                                            "node": {
                                                "namePt": "saúde",
                                                "nameEn": "health",
                                            }
                                        }
                                    ]
                                },
                                "themes": {"edges": []},
                                "organizations": {"edges": []},
                                "tables": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "TableNode:table-1",
                                                "namePt": "Tabela",
                                                "nameEn": "Table",
                                                "descriptionPt": "Desc",
                                                "descriptionEn": "Desc EN",
                                                "cloudTables": {"edges": []},
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

        result = await get_dataset_details.ainvoke(
            {"dataset_id": "dataset-1", "runtime": _runtime("en")}
        )
        dataset = json.loads(result)

        assert dataset["name"] == "Dataset"  # English picked
        assert dataset["description"] == "Descrição"  # English empty -> pt fallback
        assert dataset["tags"] == ["health"]  # English picked
        assert dataset["tables"][0]["name"] == "Table"  # English picked


class TestGetTableDetails:
    """Tests for get_table_details tool."""

    GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

    @pytest.fixture
    def mock_response(self):
        return {
            "data": {
                "allTable": {
                    "edges": [
                        {
                            "node": {
                                "id": "TableNode:table-1",
                                "namePt": "Test Table",
                                "descriptionPt": "Table description",
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
                                                "name": "peso_liquido",
                                                "descriptionPt": "Peso líquido",
                                                "measurementUnit": "kg",
                                                "coveredByDictionary": False,
                                                "isPartition": False,
                                                "bigqueryType": {"name": "FLOAT64"},
                                                "directoryPrimaryKey": None,
                                            }
                                        },
                                        {
                                            "node": {
                                                "id": "col-2",
                                                "name": "status",
                                                "descriptionPt": "Status",
                                                "measurementUnit": None,
                                                "coveredByDictionary": True,
                                                "isPartition": False,
                                                "bigqueryType": {"name": "STRING"},
                                                "directoryPrimaryKey": None,
                                            }
                                        },
                                        {
                                            "node": {
                                                "id": "col-3",
                                                "name": "id_municipio",
                                                "descriptionPt": "ID do município",
                                                "measurementUnit": None,
                                                "coveredByDictionary": False,
                                                "isPartition": False,
                                                "bigqueryType": {"name": "STRING"},
                                                "directoryPrimaryKey": {
                                                    "table": {
                                                        "id": "TableNode:dir-table-1",
                                                        "cloudTables": {
                                                            "edges": [
                                                                {
                                                                    "node": {
                                                                        "gcpDatasetId": "directory_dataset",
                                                                        "gcpTableId": "directory_table",
                                                                    }
                                                                }
                                                            ]
                                                        },
                                                    }
                                                },
                                            }
                                        },
                                        {
                                            "node": {
                                                "id": "col-4",
                                                "name": "ano",
                                                "descriptionPt": "Ano",
                                                "measurementUnit": None,
                                                "coveredByDictionary": False,
                                                "isPartition": True,
                                                "bigqueryType": {"name": "INT64"},
                                                "directoryPrimaryKey": {
                                                    "table": {
                                                        "id": "TableNode:dir-table-2",
                                                        "cloudTables": {
                                                            "edges": [
                                                                {
                                                                    "node": {
                                                                        "gcpDatasetId": next(
                                                                            iter(
                                                                                SKIP_DIRECTORY_DATASETS
                                                                            )
                                                                        ),
                                                                        "gcpTableId": "ano",
                                                                    }
                                                                }
                                                            ]
                                                        },
                                                    }
                                                },
                                            }
                                        },
                                    ]
                                },
                                "dataset": {"id": "DatasetNode:dataset-1"},
                            }
                        }
                    ]
                }
            }
        }

    @respx.mock
    async def test_get_table_details_success(self, mock_response):
        """Test successful table details retrieval."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await get_table_details.ainvoke(
            {"table_id": "table-1", "runtime": _runtime()}
        )
        table = json.loads(result)

        assert table["id"] == "table-1"
        assert table["dataset_id"] == "dataset-1"
        assert table["gcp_id"] == "basedosdados.test_dataset.test_table"
        assert table["name"] == "Test Table"
        assert table["description"] == "Table description"
        assert table["period_start"] == "2020"
        assert table["period_end"] == "2023"
        assert table["partitioned_by"] == ["ano"]

        assert len(table["columns"]) == 4

        assert table["columns"][0]["name"] == "peso_liquido"
        assert table["columns"][0]["type"] == "FLOAT64"
        assert table["columns"][0]["description"] == "Peso líquido"
        assert table["columns"][0]["unit"] == "kg"
        assert table["columns"][0]["needs_decoding"] is False
        assert "reference_table_id" not in table["columns"][0]

        assert table["columns"][1]["name"] == "status"
        assert table["columns"][1]["type"] == "STRING"
        assert table["columns"][1]["description"] == "Status"
        assert table["columns"][1]["needs_decoding"] is True
        assert "unit" not in table["columns"][1]
        assert "reference_table_id" not in table["columns"][1]

        assert table["columns"][2]["name"] == "id_municipio"
        assert table["columns"][2]["type"] == "STRING"
        assert table["columns"][2]["description"] == "ID do município"
        assert table["columns"][2]["reference_table_id"] == "dir-table-1"
        assert table["columns"][2]["needs_decoding"] is False
        assert "unit" not in table["columns"][2]

        assert table["columns"][3]["name"] == "ano"
        assert table["columns"][3]["type"] == "INT64"
        assert table["columns"][3]["description"] == "Ano"
        assert table["columns"][3]["needs_decoding"] is False
        assert "reference_table_id" not in table["columns"][3]
        assert "unit" not in table["columns"][3]

    @respx.mock
    async def test_get_table_details_null_temporal_coverage(self, mock_response):
        """Test table details when temporalCoverage is null."""
        mock_response["data"]["allTable"]["edges"][0]["node"]["temporalCoverage"] = None

        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await get_table_details.ainvoke(
            {"table_id": "table-1", "runtime": _runtime()}
        )
        table = json.loads(result)

        assert table["period_start"] is None
        assert table["period_end"] is None

    @respx.mock
    async def test_get_table_details_without_cloud_tables(self, mock_response):
        """Test table details when no cloud tables exist."""
        mock_response["data"]["allTable"]["edges"][0]["node"]["cloudTables"] = {
            "edges": []
        }

        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await get_table_details.ainvoke(
            {"table_id": "table-1", "runtime": _runtime()}
        )
        table = json.loads(result)

        assert table["gcp_id"] is None

    @respx.mock
    async def test_get_table_details_not_found(self):
        """Test error when table is not found."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json={"data": {"allTable": {"edges": []}}})
        )

        result = await get_table_details.ainvoke(
            {"table_id": "nonexistent", "runtime": _runtime()}
        )
        output = json.loads(result)

        assert output["status"] == "error"
        assert (
            output["message"]
            == "Table 'nonexistent' not found. Verify the table ID from get_dataset_details results."
        )

    @respx.mock
    async def test_localizes_metadata_to_context_language(self):
        """Table/column metadata is picked in the thread's language, pt as per-field fallback."""
        mock_response = {
            "data": {
                "allTable": {
                    "edges": [
                        {
                            "node": {
                                "id": "TableNode:table-1",
                                "namePt": "Tabela",
                                "nameEn": "Table",
                                # no English description -> must fall back to pt
                                "descriptionPt": "Descrição",
                                "descriptionEn": "",
                                "temporalCoverage": None,
                                "cloudTables": {"edges": []},
                                "columns": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "col-1",
                                                "name": "status",
                                                "descriptionPt": "Situação",
                                                "descriptionEn": "Status",
                                                "measurementUnit": None,
                                                "coveredByDictionary": False,
                                                "isPartition": False,
                                                "bigqueryType": {"name": "STRING"},
                                                "directoryPrimaryKey": None,
                                            }
                                        }
                                    ]
                                },
                                "dataset": {"id": "DatasetNode:dataset-1"},
                            }
                        }
                    ]
                }
            }
        }

        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await get_table_details.ainvoke(
            {"table_id": "table-1", "runtime": _runtime("en")}
        )
        table = json.loads(result)

        assert table["name"] == "Table"  # English picked
        assert table["description"] == "Descrição"  # English empty -> pt fallback
        assert table["columns"][0]["description"] == "Status"  # English picked


class TestFetchUsageGuide:
    """The localized usage-guide fetch: requested language, then the default, deduped."""

    GCP_DATASET_ID = "test_dataset"
    FILENAME = "test-dataset"  # underscores become dashes in the guide filename

    def _url(self, locale: str) -> str:
        return f"{BASE_USAGE_GUIDE_URL}/{locale}/{self.FILENAME}.md"

    @respx.mock
    async def test_returns_requested_language_guide(self):
        respx.get(self._url("en")).mock(
            return_value=httpx.Response(200, text="# EN guide")
        )

        assert await _fetch_usage_guide(self.GCP_DATASET_ID, "en") == "# EN guide"

    @respx.mock
    async def test_falls_back_to_default_when_localized_missing(self):
        respx.get(self._url("en")).mock(return_value=httpx.Response(404))
        pt = respx.get(self._url("pt")).mock(
            return_value=httpx.Response(200, text="# PT guide")
        )

        assert await _fetch_usage_guide(self.GCP_DATASET_ID, "en") == "# PT guide"
        assert pt.called

    @respx.mock
    async def test_default_language_fetches_once(self):
        pt = respx.get(self._url("pt")).mock(
            return_value=httpx.Response(200, text="# PT guide")
        )
        en = respx.get(self._url("en")).mock(
            return_value=httpx.Response(200, text="# EN guide")
        )

        assert await _fetch_usage_guide(self.GCP_DATASET_ID, "pt") == "# PT guide"
        assert pt.call_count == 1
        assert (
            not en.called
        )  # dedupe: no second fetch when language is already the default

    @respx.mock
    async def test_returns_none_when_no_guide_exists(self):
        respx.get(url__startswith=BASE_USAGE_GUIDE_URL).mock(
            return_value=httpx.Response(404)
        )

        assert await _fetch_usage_guide(self.GCP_DATASET_ID, "en") is None
