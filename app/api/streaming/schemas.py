from typing import Any, Literal

from pydantic import BaseModel, JsonValue, field_serializer


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

        `query_result` carries `destination_table`, which must never reach the client.
        It's kept in memory (so the server can capture the handle off the tool output)
        but dropped from every serialization: the SSE stream and the persisted `events`
        that `list_messages` returns.

        Args:
            value (Any): The artifact being serialized.

        Returns:
            Any | None: The artifact, or None if it is an internal handle.
        """
        if isinstance(value, dict) and value.get("type") == "query_result":
            return None
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
    downloads: list[dict[str, Any]] | None = None
    error_details: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    type: EventType
    data: EventData

    def to_sse(self) -> str:
        return self.model_dump_json() + "\n\n"
