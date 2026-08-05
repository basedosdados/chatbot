from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel


class ConfigDict(TypedDict):
    run_id: str
    configurable: dict[str, Any]
    metadata: NotRequired[dict[str, Any]]


class UserMessage(BaseModel):
    content: str
