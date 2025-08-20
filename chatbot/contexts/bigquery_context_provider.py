import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread
from typing import Literal

from google.cloud import bigquery as bq
from google.cloud.bigquery.dataset import (Dataset, DatasetListItem,
                                           DatasetReference)
from google.cloud.bigquery.table import Table, TableListItem, TableReference
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from chatbot.contexts.metadata_formatter import MetadataFormatterFactory

from .context_provider import BaseContextProvider

# cache living time (in seconds)
CACHE_TTL = 60*60

class Data(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset = Field(description="A BigQuery `Dataset` object.")
    tables: list[Table] = Field(description="A list of BigQuery `Table` objects.")

class BigQueryContextProvider(BaseContextProvider):
    """A BigQuery-backed database interface, for fetching and formatting data.

    Args:
        billing_project (str | None, optional):
            Project ID for the project which the client acts on behalf of.
            Will be used when creating a dataset/table/job. If not provided,
            falls back to the default project inferred from the environment.
        query_project (str | None, optional):
            Project ID for the project from which the datasets and tables
            will be fetched and in which SQL queries will be run. If not provided,
            falls back to the default project inferred from the environment.
        max_workers (int | None, optional):
            Maximum number of workers for `ThreadPoolExecutor` instances
            used during data fetching. Defaults to `4`.
        max_results (int | None, optional):
            Maximum number of results per page when fetching datasets and tables.
            Defaults to `1000`.
        timeout (float | None, optional):
            The number of seconds to wait for an HTTP response before retrying.
            Defaults to `30.0`.
        metadata_format (Literal["markdown", "xml"], optional):
            The format in which fetched metadata should be returned. Accepted
            values are `"markdown"` and `"xml"`. Defaults to `"markdown"`.
    """

    def __init__(
        self,
        billing_project: str | None = None,
        query_project: str | None = None,
        max_workers: int | None = 4,
        max_results: int | None = 1000,
        timeout: float | None = 30.0,
        metadata_format: Literal["markdown", "xml"] = "markdown",
    ):
        billing_project = billing_project or os.getenv("BILLING_PROJECT_ID")
        query_project = query_project or os.getenv("QUERY_PROJECT_ID")

        self._client = bq.Client(billing_project)
        self._project = query_project
        self._max_workers = max_workers
        self._max_results = max_results
        self._timeout = timeout

        self._formatter = MetadataFormatterFactory.get_metadata_formatter(
            format=metadata_format
        )

        self._cache: dict[str, Data] = {}
        self._cache_lock = Lock()
        self._cache_thread = Thread(target=self._run_cache_data, daemon=True)
        self._cache_data()
        self._cache_thread.start()

    def _run_cache_data(self):
        """Caches data every `CACHE_TTL` seconds. Should be used as a background function
        to periodically update the cache, as it is a time-consuming operation

        Raises:
            e: Any exception
        """
        while True:
            time.sleep(CACHE_TTL)
            try:
                self._cache_data()
            except Exception as e:
                logger.exception(f"Error on data caching:")
                raise e

    def _fetch_dataset(
        self,
        dataset: DatasetReference | DatasetListItem
    ) -> Dataset:
        """Fetches a single BigQuery dataset referenced by `dataset`

        Args:
            dataset (DatasetReference | DatasetListItem): A reference to the dataset

        Returns:
            Dataset: A `Dataset` instance
        """
        # Added a timeout to the get_dataset api request as a precaution. See the comment below for more details.
        return self._client.get_dataset(
            dataset.dataset_id,
            timeout=self._timeout
        )

    def _fetch_datasets(self) -> list[Dataset]:
        """Fetches all datasets from a BigQuery project

        Returns:
            list[Dataset]: A list of `Dataset` instances
        """
        dataset_list = list(
            self._client.list_datasets(
                self._project,
                timeout=self._timeout,
                max_results=self._max_results
            )
        )

        with ThreadPoolExecutor(self._max_workers) as executor:
            datasets = executor.map(self._fetch_dataset, dataset_list)

        return list(datasets)

    def _fetch_table(
        self,
        table: Table | TableListItem | TableReference | str
    ) -> Table:
        """Fetches a single table from a BigQuery dataset, referenced by `table`

        Args:
            table (Table | TableListItem | TableReference | str): A reference to the table

        Returns:
            Table: A `Table` instance
        """
        # Sometimes, the get_table api request hangs indefinitely, so a timeout was added to force the request to return.
        # So far, the table has always been returned correctly after the timeout, which indicates that the data
        # is being fetched but there is an issue with returning it.
        return self._client.get_table(
            table,
            timeout=self._timeout
        )

    def _fetch_tables(
        self,
        dataset: Dataset | DatasetReference | DatasetListItem | str
    ) -> list[Table]:
        """Fetches all tables from a BigQuery dataset

        Args:
            dataset (Dataset | DatasetReference | DatasetListItem | str): A dataset reference

        Returns:
            list[Table]: A list of `Table` instances
        """
        table_list = list(
            self._client.list_tables(
                dataset,
                timeout=self._timeout,
                max_results=self._max_results
            )
        )

        with ThreadPoolExecutor(self._max_workers) as executor:
            tables = executor.map(self._fetch_table, table_list)

        return list(tables)

    def _cache_data(self):
        """Caches all BigQuery datasets and its tables in memory
        """
        logger.info(f"Caching data")

        datasets = self._fetch_datasets()

        data = {
            dataset.dataset_id: Data(
                dataset=dataset,
                tables=self._fetch_tables(dataset.dataset_id)
            )
            for dataset in datasets
        }

        with self._cache_lock:
            self._cache = data

        logger.success(f"Data successfully cached")

    def _format_table_metadata(self, table: Table) -> str:
        """Formats the metadata of a BigQuery table

        Args:
            table (Table): A BigQuery table

        Raises:
            query_exception: If a query failed for any reason

        Returns:
            str: The formatted metadata
        """
        full_table_name = table.full_table_id.replace(":", ".")

        sample_query_filter = f"WHERE {table.schema[0].name} IS NOT NULL " + " ".join(
            [f"AND {field.name} IS NOT NULL" for field in table.schema[1:]]
        )

        sample_query = f"SELECT * FROM `{full_table_name}` {sample_query_filter} LIMIT 3"

        try:
            sample_rows = self._client.query(
                query=sample_query,
                project=self._project
            ).result()
            if sample_rows.total_rows == 0:
                sample_rows = self._client.query(
                    query=f"SELECT * FROM `{full_table_name}` LIMIT 3",
                    project=self._project
                ).result()
            sample_rows = [dict(row) for row in sample_rows]
        except Exception as query_exception:
            logger.exception("Error on getting table info:")
            raise query_exception

        return self._formatter.format_table_metadata(table, sample_rows)

    def _get_tables_metadata(self, dataset_name: str) -> str:
        """Gets the formatted metadata of all the tables of a single BigQuery dataset

        Args:
            dataset_name (str): Dataset name

        Returns:
            str: The formatted metadata
        """
        with self._cache_lock:
            tables = self._cache[dataset_name].tables

        with ThreadPoolExecutor(self._max_workers) as executor:
            tables_metadata = executor.map(self._format_table_metadata, tables)

        return "\n\n".join(tables_metadata)

    def get_datasets_info(self, query: str) -> str:
        """Gets the formatted metadata of all the datasets of a BigQuery project

        Args:
            query (str): A natural language user message. It is unused in this implementation.

        Returns:
            str: The formatted metadata
        """
        with self._cache_lock:
            datasets_info = [
                self._formatter.format_dataset_metadata(data.dataset, data.tables)
                for data in self._cache.values()
            ]

        return "\n\n---\n\n".join(datasets_info)

    def get_tables_info(self, dataset_names: str) -> str:
        """Gets the formatted metadata of all the tables of one or more BigQuery datasets

        Args:
            dataset_names (str): A comma-separated list of dataset names

        Returns:
            str: The formatted metadata
        """
        tables_info = []

        dataset_names = [name.strip() for name in dataset_names.split(",")]

        for dataset_name in dataset_names:
            dataset_tables_info = self._get_tables_metadata(dataset_name)
            tables_info.append(dataset_tables_info)

        return "\n\n---\n\n".join(tables_info)

    def get_query_results(self, sql_query: str) -> str:
        """Run a SQL query in BigQuery.

        Args:
            query (sql_query): A valid GoogleSQL query statement.

        Raises:
            query_exception: If the query failed for any reason.

        Returns:
            str: The execution results, formatted as a string.
        """
        try:
            rows = self._client.query(sql_query, project=self._project).result()

            results = [dict(row) for row in rows]

            if results:
                return json.dumps(results, ensure_ascii=False, default=str)
            return ""
        except Exception as query_exception:
            logger.exception("Error on querying table:")
            raise query_exception
