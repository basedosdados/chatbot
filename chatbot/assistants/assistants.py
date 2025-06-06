from typing import Any, Protocol


class Assistant(Protocol):
    def invoke(self, message: str, config: dict|None=None) -> Any:
        ...

    def clear_thread(self, thread_id: str):
        ...

class AsyncAssistant(Protocol):
    async def invoke(self, message: str, config: dict|None=None) -> Any:
        ...

    async def clear_thread(self, thread_id: str):
        ...
