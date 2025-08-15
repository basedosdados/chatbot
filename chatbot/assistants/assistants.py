from typing import AsyncIterator, Iterator, Protocol, TypeVar

T = TypeVar("T", covariant=True)

class Assistant(Protocol[T]):
    def invoke(
        self,
        message: str,
        config: dict|None=None,
        rewrite_query: bool=False
    ) -> T:
        ...

    def stream(
        self,
        message: str,
        config: dict|None=None,
        stream_mode: list[str]|None=None,
        subgraphs: bool=False,
        rewrite_query: bool=False
    ) -> Iterator[dict|tuple]:
        ...

    def clear_thread(self, thread_id: str):
        ...

class AsyncAssistant(Protocol):
    async def ainvoke(
        self,
        message: str,
        config: dict|None=None,
        rewrite_query: bool=False
    ) -> T:
        ...

    async def astream(
        self,
        message: str,
        config: dict|None=None,
        stream_mode: list[str]|None=None,
        subgraphs: bool=False,
        rewrite_query: bool=False
    ) -> AsyncIterator[dict|tuple]:
        ...

    async def aclear_thread(self, thread_id: str):
        ...
