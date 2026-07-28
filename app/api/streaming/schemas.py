from typing import Any, Literal

from pydantic import BaseModel, JsonValue, field_serializer

from app.exports import query_result_download


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

    @field_serializer("artifact")
    def _redact_internal_artifact(self, value: Any) -> Any | None:
        """Surface client-facing artifacts and redact internal handles.

        A `query_result` handle is projected onto its download descriptor. The original handle
        stays in memory, where the server reads it off the tool output; it is dropped from
        every serialization, both the SSE stream and the persisted `events` list.

        Args:
            value (Any): The artifact being serialized.

        Returns:
            Any | None: The artifact, or the download descriptor for an internal handle.
        """
        if isinstance(value, dict) and value.get("type") == "query_result":
            return query_result_download(value["query_ref"], value["slug"])
        return value


EventType = Literal[
    "tool_call",
    "tool_output",
    "final_answer",
    "model_call_limit",
    "error",
    "complete",
]


class EventData(BaseModel):
    run_id: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_outputs: list[ToolOutput] | None = None
    structured_response: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    type: EventType
    data: EventData

    def to_sse(self) -> str:
        return self.model_dump_json() + "\n\n"
