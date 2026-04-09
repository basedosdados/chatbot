import uuid
from typing import Any, Literal

from pydantic import BaseModel, JsonValue


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolOutput(BaseModel):
    status: Literal["error", "success"]
    tool_call_id: str
    tool_name: str
    content: str
    artifact: JsonValue | None = None
    metadata: JsonValue | None = None


EventType = Literal[
    "tool_call",
    "tool_output",
    "final_answer",
    "error",
    "complete",
]


class EventData(BaseModel):
    run_id: uuid.UUID | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_outputs: list[ToolOutput] | None = None
    error_details: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    type: EventType
    data: EventData

    def to_sse(self) -> str:
        return self.model_dump_json() + "\n\n"
