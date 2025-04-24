from typing import Any, Protocol

from .datatypes import UserMessage


class Assistant(Protocol):
    @staticmethod
    def _format_response(response: dict[str, Any]) -> dict[str, Any]:
        ...

    def invoke(self, message: UserMessage, thread_id: str) -> Any:
        ...

    def clear_thread(self, thread_id: str):
        ...

class AsyncAssistant(Protocol):
    @staticmethod
    def _format_response(response: dict[str, Any]) -> dict[str, Any]:
        ...

    async def invoke(self, message: UserMessage, thread_id: str) -> Any:
        ...

    async def clear_thread(self, thread_id: str):
        ...
