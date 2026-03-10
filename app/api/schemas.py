from typing import Any, TypedDict

from pydantic import BaseModel


class ConfigDict(TypedDict):
    run_id: str
    configurable: dict[str, Any]


class UserMessage(BaseModel):
    content: str
