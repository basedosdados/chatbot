from typing import Protocol, runtime_checkable


@runtime_checkable
class Database(Protocol):
    def get_datasets_info(self) -> str:
        ...

    def get_tables_info(self, dataset_names: str) -> str:
        ...

    def query(self, query: str) -> str:
        ...
