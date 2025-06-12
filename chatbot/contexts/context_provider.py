from abc import ABC, abstractmethod


class BaseContextProvider(ABC):
    """Base class defining methods for providing contextual information
    to support SQL query generation and execution based on a user query.
    """

    @abstractmethod
    def get_datasets_info(self, query: str) -> str:
        """Retrieve metadata about datasets relevant to a given user query.

        Args:
            query (str): A natural language user message.

        Returns:
            str: A formatted summary of datasets (names, descriptions, schemas, etc.)
                 that match or relate to the query.
        """
        ...

    @abstractmethod
    def get_tables_info(self, dataset_names: str) -> str:
        """Retrieve metadata about tables within specified datasets.

        Args:
            dataset_names (str): One or more dataset identifiers, separated by commas.

        Returns:
            str: A detailed overview of table names, descriptions and their schemas
                 for each specified dataset.
        """
        ...

    @abstractmethod
    def get_query_results(self, sql_query: str) -> str:
        """Execute a raw SQL statement.

        Args:
            sql_query (str): A valid SQL statement to execute.

        Returns:
            str: The execution results, formatted as a string.
        """
        ...
