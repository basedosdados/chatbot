from typing import Iterator
from unittest.mock import MagicMock

import pytest
from google.cloud.bigquery.dataset import Dataset, DatasetListItem
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.table import Row, Table, TableListItem
from pydantic import BaseModel

from chatbot.contexts import BigQueryContextProvider

# constants for the number of datasets and tables created
# in the `list_datasets` and `list_tables` mock functions
N_DATASETS = 2
N_TABLES = 2
N_FIELDS = 3

# ============================== fixtures ==============================
@pytest.fixture
def mock_bigquery_client():
    class MockBigQueryClient:
        # it should return a Dataset object
        # https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.dataset.Dataset
        def get_dataset(self, dataset_id: str, timeout: float|None = None):
            dataset = MagicMock(spec=Dataset)
            dataset.dataset_id = dataset_id
            dataset.description = f"mock description for dataset {dataset_id}"
            return dataset

        # it should return an Iterator of DatasetListItem objects
        # https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.dataset.DatasetListItem
        def list_datasets(self, project: str, timeout: float|None = None, max_results: int|None = None) -> Iterator:
            dataset_list = [MagicMock(spec=DatasetListItem) for _ in range(N_DATASETS)]
            for i, dataset_list_item in enumerate(dataset_list):
                dataset_list_item.dataset_id = f"mock_dataset_{i+1}"
            return iter(dataset_list)

        # it should return a Table object
        # https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.Table
        def get_table(self, table_list_item: TableListItem, timeout: float|None = None):
            table = MagicMock(spec=Table)

            table.dataset_id = table_list_item.dataset_id
            table.table_id = table_list_item.table_id
            table.full_table_id = table_list_item.full_table_id
            table.description=f"mock description for table {table.full_table_id}"

            schema = [MagicMock(spec=SchemaField) for _ in range(N_FIELDS)]

            for i, field in enumerate(schema):
                field.name = f"field_{i+1}"
                field.field_type = f"STRING"
                field.description = f"mock_field_{i+1}"

            table.schema = schema

            return table

        # it should return an Iterator of TableListItem objects
        # https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.TableListItem
        def list_tables(self, dataset_id: str, timeout: float|None = None, max_results: int|None = None):
            table_list = [MagicMock(spec=TableListItem) for _ in range(N_TABLES)]
            for i, table_list_item in enumerate(table_list):
                table_list_item.dataset_id=dataset_id
                table_list_item.full_table_id=f"mock_project:{dataset_id}.table_{i+1}"
            return iter(table_list)

        # it should return an Iterator of Rows when the return() method is called
        # https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.Row
        def query(self, query: str, project: str):
            query_job = MagicMock()

            query_job.query_results = iter([
                Row(
                    values=("value_1", "value_2", "value_3"),
                    field_to_index={"field_1": 0, "field_2": 1, "field_3": 2}
                )
            ])

            query_job.result.return_value = query_job.query_results

            return query_job

    return MockBigQueryClient()

@pytest.fixture
def context_provider(mock_bigquery_client, monkeypatch):
    monkeypatch.setattr(
        "google.cloud.bigquery.Client", lambda _: mock_bigquery_client
    )

    return BigQueryContextProvider()

@pytest.fixture
def dataset_list_item():
    dataset_id = "mock_dataset_id"
    dataset_list_item = MagicMock(spec=DatasetListItem)
    dataset_list_item.dataset_id = dataset_id
    dataset_list_item.description = f"mock description for dataset {dataset_id}"
    return dataset_list_item

@pytest.fixture
def table_list_item():
    dataset_id = "mock_dataset_id"
    full_table_id = f"mock_project:{dataset_id}.mock_table_id"
    table_list_item = MagicMock(spec=TableListItem)
    table_list_item.dataset_id = dataset_id
    table_list_item.full_table_id = full_table_id
    return table_list_item

# ============================== test cases ==============================
def test_fetch_dataset(context_provider: BigQueryContextProvider, dataset_list_item: MagicMock):
    dataset = context_provider._fetch_dataset(dataset_list_item)
    assert dataset.dataset_id == dataset_list_item.dataset_id
    assert dataset.description == dataset_list_item.description

def test_fetch_datasets(context_provider: BigQueryContextProvider):
    datasets = context_provider._fetch_datasets()
    assert isinstance(datasets, list)
    for dataset in datasets:
        assert hasattr(dataset, "dataset_id")
        assert hasattr(dataset, "description")

def test_fetch_table(context_provider: BigQueryContextProvider, table_list_item: MagicMock):
    table = context_provider._fetch_table(table_list_item)
    assert table.dataset_id == table_list_item.dataset_id
    assert table.full_table_id == table_list_item.full_table_id
    assert hasattr(table, "table_id")
    assert hasattr(table, "description")
    assert hasattr(table, "schema")

def test_fetch_tables(context_provider: BigQueryContextProvider):
    tables = context_provider._fetch_tables("mock_dataset_id")
    assert isinstance(tables, list)
    for table in tables:
        assert table.dataset_id == "mock_dataset_id"
        assert hasattr(table, "table_id")
        assert hasattr(table, "full_table_id")
        assert hasattr(table, "description")
        assert hasattr(table, "schema")

def test_cache_data(context_provider: BigQueryContextProvider):
    context_provider._cache_data()

    assert isinstance(context_provider._cache, dict)

    for dataset_id, data in context_provider._cache.items():
        assert isinstance(dataset_id, str)
        assert isinstance(data, BaseModel)

        assert hasattr(data.dataset, "dataset_id")
        assert hasattr(data.dataset, "description")

        assert isinstance(data.tables, list)

        for table in data.tables:
            assert data.dataset.dataset_id == table.dataset_id
            assert hasattr(table, "table_id")
            assert hasattr(table, "full_table_id")
            assert hasattr(table, "description")
            assert hasattr(table, "schema")

def test_query(context_provider: BigQueryContextProvider):
    expected = '[{"field_1": "value_1", "field_2": "value_2", "field_3": "value_3"}]'
    query_results = context_provider.get_query_results("mock_query")
    assert query_results == expected
