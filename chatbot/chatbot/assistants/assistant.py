from typing import Any, Protocol

from .datatypes import UserQuestion


class Assistant(Protocol):
    @staticmethod
    def _format_response(response: dict[str, Any]) -> dict[str, Any]:
        ...

    def ask(self, user_question: UserQuestion) -> Any:
        ...

    async def aask(self, user_question: UserQuestion) -> Any:
        ...

    def clear_memory(self, thread_id: str):
        ...

    async def aclear_memory(self, thread_id: str):
        ...
