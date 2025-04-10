from typing import Any, Protocol

from .datatypes import UserMessage


class Assistant(Protocol):
    @staticmethod
    def _format_response(response: dict[str, Any]) -> dict[str, Any]:
        ...

    def invoke(self, message: UserMessage) -> Any:
        ...

    async def ainvoke(self, message: UserMessage) -> Any:
        ...

    def clear_thread(self, thread_id: str):
        ...

    async def aclear_thread(self, thread_id: str):
        ...
