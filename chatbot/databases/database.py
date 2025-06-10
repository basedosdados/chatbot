from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SQLExample(BaseModel):
    """Represents a single few-shot example."""
    question: str
    query: str


@runtime_checkable
class ContextProvider(Protocol):
    def get_datasets_info(self, query: str) -> str:
        ...

    def get_tables_info(self, dataset_names: str) -> str:
        ...

    def get_sql_examples(self, query: str) -> list[SQLExample]:
        ...

    async def aget_sql_examples(self, query: str) -> list[SQLExample]:
        ...

    def query(self, query: str) -> str:
        ...
