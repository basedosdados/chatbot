from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SQLExample(BaseModel):
    """A single example pairing a natural language question with its corresponding SQL query.

    Attributes:
        question (str): The natural language question.
        query (str): The SQL query that answers the question.
    """
    question: str
    query: str


@runtime_checkable
class ContextProvider(Protocol):
    """Protocol defining methods for providing contextual information and examples
    to support SQL generation and execution based on a user query.
    """

    def get_datasets_info(self, query: str) -> str:
        """Retrieve metadata about datasets relevant to a given user query.

        Args:
            query (str): A natural language user message.

        Returns:
            str: A formatted summary of datasets (names, descriptions, schemas, etc.)
                 that match or relate to the query.
        """
        ...

    def get_tables_info(self, dataset_names: str) -> str:
        """Retrieve metadata about tables within specified datasets.

        Args:
            dataset_names (str): One or more dataset identifiers, separated by commas.

        Returns:
            str: A detailed overview of table names, descriptions and their schemas
                 for each specified dataset.
        """
        ...

    def get_sql_examples(self, query: str) -> list[SQLExample]:
        """Fetch example SQL queries relevant to a given user message
        for few-shot prompting building.

        Args:
            query (str): A natural language user message.

        Returns:
            list[SQLExample]: A list of `SQLExample` instances,
                              each containing a sample question and its SQL query.
        """
        ...

    async def aget_sql_examples(self, query: str) -> list[SQLExample]:
        """Asynchronously fetch example SQL queries relevant to a given user message
        for few-shot prompting building.

        Args:
            query (str): A natural language user message.

        Returns:
            list[SQLExample]: A list of `SQLExample` instances,
                              each containing a sample question and its SQL query.
        """
        ...

    def query(self, sql_query: str) -> str:
        """Execute a raw SQL statement.

        Args:
            sql_query (str): A valid SQL statement to execute.

        Returns:
            str: The execution results, formatted as a string.
        """
        ...
