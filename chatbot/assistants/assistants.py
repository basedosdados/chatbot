from typing import Any, Protocol


class Assistant(Protocol):
    def invoke(self, message: str, config: dict|None=None) -> Any:
        ...

    def stream(
        self,
        message: str,
        config: dict|None=None,
        stream_mode: list[str]|None=None,
        subgraphs: bool=False,
        rewrite_query: bool=False
    ) -> Any:
        ...

    def clear_thread(self, thread_id: str):
        ...

class AsyncAssistant(Protocol):
    async def ainvoke(self, message: str, config: dict|None=None) -> Any:
        ...

    async def astream(
        self,
        message: str,
        config: dict|None=None,
        stream_mode: list[str]|None=None,
        subgraphs: bool=False,
        rewrite_query: bool=False
    ) -> Any:
        ...

    async def aclear_thread(self, thread_id: str):
        ...
