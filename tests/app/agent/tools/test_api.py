import json

import httpx
import pytest
import respx

from app.agent.tools.api import (
    SKIP_DIRECTORY_DATASETS,
    get_dataset_details,
    get_table_details,
    search_datasets,
)
from app.settings import settings


class TestSearchDatasets:
    """Tests for search_datasets tool."""

    SEARCH_ENDPOINT = f"{settings.BASEDOSDADOS_BASE_URL}/search/"

    @respx.mock
    async def test_search_datasets_returns_overviews(self):
        """Test successful dataset search."""
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

        result = await search_datasets.ainvoke({"query": "test"})
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

        result = await search_datasets.ainvoke({"query": "nonexistent"})
        output = json.loads(result)

        assert output == []


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
                                "name": "Test Dataset",
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
                                                "id": "TableNode:table-1",
                                                "name": "Test Table",
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

        result = await get_dataset_details.ainvoke({"dataset_id": "dataset-1"})
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

        result = await get_dataset_details.ainvoke({"dataset_id": "dataset-1"})
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
                                "name": "Test Dataset",
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

        result = await get_dataset_details.ainvoke({"dataset_id": "dataset-1"})
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

        result = await get_dataset_details.ainvoke({"dataset_id": "dataset-1"})
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

        result = await get_dataset_details.ainvoke({"dataset_id": "nonexistent"})
        output = json.loads(result)

        assert output["status"] == "error"
        assert (
            output["message"]
            == "Dataset 'nonexistent' not found. Verify the dataset ID from search_datasets results."
        )


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
                                "name": "Test Table",
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
                                                "name": "peso_liquido",
                                                "description": "Peso líquido",
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
                                                "description": "Status",
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
                                                "description": "ID do município",
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
                                                "description": "Ano",
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

        result = await get_table_details.ainvoke({"table_id": "table-1"})
        table = json.loads(result)

        assert table["id"] == "table-1"
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

        result = await get_table_details.ainvoke({"table_id": "table-1"})
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

        result = await get_table_details.ainvoke({"table_id": "table-1"})
        table = json.loads(result)

        assert table["gcp_id"] is None

    @respx.mock
    async def test_get_table_details_not_found(self):
        """Test error when table is not found."""
        respx.post(self.GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json={"data": {"allTable": {"edges": []}}})
        )

        result = await get_table_details.ainvoke({"table_id": "nonexistent"})
        output = json.loads(result)

        assert output["status"] == "error"
        assert (
            output["message"]
            == "Table 'nonexistent' not found. Verify the table ID from get_dataset_details results."
        )
