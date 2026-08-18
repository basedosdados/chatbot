from langchain_core.tools import BaseTool

from app.agent.tools.api import get_dataset_details, get_table_details, search_datasets
from app.agent.tools.bigquery import decode_table_values, execute_bigquery_sql


class BDToolkit:
    @staticmethod
    def get_tools() -> list[BaseTool]:
        """Return all available tools for Base dos Dados database interaction.

        This function provides a complete set of tools for discovering, exploring,
        and querying Brazilian public datasets through the Base dos Dados platform.

        Returns:
            list[BaseTool]: Tools in suggested usage order:
                - search_datasets: Find datasets using keywords.
                - get_dataset_details: Get comprehensive dataset information.
                - get_table_details: Get comprehensive table information.
                - execute_bigquery_sql: Execute SQL queries against BigQuery tables.
                - decode_table_values: Decode coded values using dictionary tables.
        """
        return [
            search_datasets,
            get_dataset_details,
            get_table_details,
            execute_bigquery_sql,
            decode_table_values,
        ]


__all__ = ["BDToolkit"]
