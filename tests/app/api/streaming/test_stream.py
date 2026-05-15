import asyncio
import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.api.streaming.agent_runner import (
    ErrorMessage,
    _parse_thinking,
    _process_chunk,
    _truncate_json,
)
from app.api.streaming.schemas import EventData, StreamEvent
from app.api.streaming.stream import sse_forwarder


class TestTruncateJSON:
    """Tests for _truncate_json function."""

    STR_MAX_LEN = 300
    STR_LONG_LEN = 400
    STR_REMAINING = STR_LONG_LEN - STR_MAX_LEN

    LIST_MAX_LEN = 10
    LIST_LONG_LEN = 15
    LIST_REMAINING = LIST_LONG_LEN - LIST_MAX_LEN

    @staticmethod
    def _format_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def test_truncate_json_long_string(self):
        """Test that long strings are truncated with a remaining count."""
        data = {"long_string": "a" * self.STR_LONG_LEN}
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string, max_str_len=self.STR_MAX_LEN)
        expected_str = (
            "a" * self.STR_MAX_LEN + f"... ({self.STR_REMAINING} more characters)"
        )
        expected_json = self._format_json({"long_string": expected_str})
        assert truncated == expected_json

    def test_truncate_json_long_list(self):
        """Test that long lists are truncated with a remaining count."""
        data = {"long_list": list(range(self.LIST_LONG_LEN))}
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string, max_list_len=self.LIST_MAX_LEN)
        expected_list = list(range(self.LIST_MAX_LEN)) + [
            f"... ({self.LIST_REMAINING} more items)"
        ]
        expected_json = self._format_json({"long_list": expected_list})
        assert truncated == expected_json

    def test_truncate_json_nested(self):
        """Test that nested structures have both strings and lists truncated."""
        data = {
            "short_string": "a" * 100,
            "nested_list": [
                {
                    "short_string": "b" * 100,
                    "long_string": "c" * self.STR_LONG_LEN,
                    "int": 1,
                    "float": 1.0,
                }
                for _ in range(self.LIST_LONG_LEN)
            ],
            "nested_dict": {"long_string": "d" * self.STR_LONG_LEN},
        }
        json_string = json.dumps(data)
        truncated = _truncate_json(
            json_string, max_list_len=self.LIST_MAX_LEN, max_str_len=self.STR_MAX_LEN
        )
        expected_data = {
            "short_string": "a" * 100,
            "nested_list": [
                {
                    "short_string": "b" * 100,
                    "long_string": "c" * self.STR_MAX_LEN
                    + f"... ({self.STR_REMAINING} more characters)",
                    "int": 1,
                    "float": 1.0,
                }
                for _ in range(self.LIST_MAX_LEN)
            ]
            + [f"... ({self.LIST_REMAINING} more items)"],
            "nested_dict": {
                "long_string": "d" * self.STR_MAX_LEN
                + f"... ({self.STR_REMAINING} more characters)"
            },
        }
        expected_json = self._format_json(expected_data)
        assert truncated == expected_json

    def test_truncate_json_not_dict(self):
        """Test that non-dict JSON is returned as-is."""
        data = list(range(self.LIST_LONG_LEN))
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string)
        assert truncated == json_string

    def test_truncate_json_not_needed(self):
        """Test that short strings and lists are not truncated."""
        data = {
            "short_string": "hello",
            "short_list": [1, 2, 3],
        }
        json_string = json.dumps(data)
        expected_json = self._format_json(data)
        assert _truncate_json(json_string) == expected_json

    def test_truncate_json_invalid(self):
        """Test that invalid JSON is returned as-is."""
        invalid_json_string = '{"key": "value"'
        assert _truncate_json(invalid_json_string) == invalid_json_string


class TestParseThinking:
    """Tests for _parse_thinking function."""

    def test_string_content_returns_none(self):
        """Test that plain string content returns None."""
        message = AIMessage(content="Hello, world!")
        assert _parse_thinking(message) is None

    def test_single_thinking_block(self):
        """Test extraction of a single thinking block."""
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Let me reason about this."},
                {"type": "text", "text": "Here is my answer."},
            ]
        )
        assert _parse_thinking(message) == "Let me reason about this."

    def test_multiple_thinking_blocks_are_concatenated(self):
        """Test that multiple thinking blocks are concatenated."""
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "First thought. "},
                {"type": "text", "text": "Some text."},
                {"type": "thinking", "thinking": "Second thought."},
            ]
        )
        assert _parse_thinking(message) == "First thought. Second thought."

    def test_no_thinking_blocks_returns_none(self):
        """Test that content with no thinking blocks returns None."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "Just text."},
            ]
        )
        assert _parse_thinking(message) is None

    def test_empty_thinking_block_returns_none(self):
        """Test that an empty thinking string returns None."""
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": ""},
            ]
        )
        assert _parse_thinking(message) is None

    def test_non_dict_blocks_are_skipped(self):
        """Test that non-dict items in content are safely skipped."""
        message = AIMessage(
            content=[
                "plain string block",
                {"type": "thinking", "thinking": "Actual thinking."},
            ]
        )
        assert _parse_thinking(message) == "Actual thinking."


class TestProcessChunk:
    """Tests for _process_chunk function."""

    def test_agent_chunk_with_tool_calls(self):
        """Test agent chunk with tool calls returns tool_call event."""
        chunk = {
            "model": {
                "messages": [
                    AIMessage(
                        content="Let me search for that.",
                        tool_calls=[
                            {
                                "id": "call_123",
                                "name": "search",
                                "args": {"query": "foo"},
                            }
                        ],
                    )
                ]
            }
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_call"
        assert event.data.run_id is None
        assert event.data.tool_outputs is None
        assert event.data.error_details is None
        assert event.data.content == "Let me search for that."
        assert len(event.data.tool_calls) == 1

        tool_call = event.data.tool_calls[0]

        assert tool_call.id == "call_123"
        assert tool_call.name == "search"
        assert tool_call.args == {"query": "foo"}

    def test_agent_chunk_with_multiple_tool_calls(self):
        """Test agent chunk with multiple parallel tool calls."""
        chunk = {
            "model": {
                "messages": [
                    AIMessage(
                        content="I'll search both.",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "search",
                                "args": {"query": "foo"},
                            },
                            {"id": "call_2", "name": "lookup", "args": {"id": "123"}},
                        ],
                    )
                ]
            }
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_call"
        assert len(event.data.tool_calls) == 2
        assert event.data.tool_calls[0].name == "search"
        assert event.data.tool_calls[1].name == "lookup"

    def test_agent_chunk_final_answer(self):
        """Test agent chunk without tool calls returns final_answer event."""
        chunk = {"model": {"messages": [AIMessage(content="Here is your answer.")]}}

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "final_answer"
        assert event.data.run_id is None
        assert event.data.tool_calls is None
        assert event.data.tool_outputs is None
        assert event.data.error_details is None
        assert event.data.content == "Here is your answer."

    def test_agent_chunk_empty_messages(self):
        """Test agent chunk with empty messages list returns empty final_answer."""
        chunk = {"model": {"messages": []}}

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "final_answer"
        assert event.data.content == ""

    def test_tools_chunk_single_tool(self):
        """Test tools chunk with single tool output (dict format)."""
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"result": "found"}',
                        tool_call_id="call_123",
                        name="search",
                        status="success",
                        artifact={"url": "http://example.com"},
                    )
                ]
            }
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_output"
        assert len(event.data.tool_outputs) == 1

        tool_output = event.data.tool_outputs[0]

        assert tool_output.status == "success"
        assert tool_output.tool_call_id == "call_123"
        assert tool_output.tool_name == "search"
        assert tool_output.content == '{\n  "result": "found"\n}'
        assert tool_output.artifact == {"url": "http://example.com"}
        assert tool_output.metadata is None

    def test_tools_chunk_multiple_parallel_tools(self):
        """Test tools chunk with multiple parallel tool outputs (list format)."""
        chunk = {
            "tools": [
                {
                    "messages": [
                        ToolMessage(
                            content='{"data": "foo"}',
                            tool_call_id="call_1",
                            name="search",
                            status="success",
                        )
                    ]
                },
                {
                    "messages": [
                        ToolMessage(
                            content='{"data": "bar"}',
                            tool_call_id="call_2",
                            name="lookup",
                            status="success",
                        )
                    ]
                },
            ]
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_output"
        assert len(event.data.tool_outputs) == 2
        assert event.data.tool_outputs[0].tool_call_id == "call_1"
        assert event.data.tool_outputs[1].tool_call_id == "call_2"

    def test_tools_chunk_with_error_status(self):
        """Test tools chunk with error status."""
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="Tool execution failed",
                        tool_call_id="call_123",
                        name="search",
                        status="error",
                    )
                ]
            }
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_output"
        assert event.data.tool_outputs[0].status == "error"

    def test_tools_chunk_unexpected_format(self):
        """Test tools chunk with unexpected format returns empty tool_outputs."""
        chunk = {"tools": "unexpected string"}

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "tool_output"
        assert event.data.tool_outputs == []

    def test_model_call_limit_triggered_chunk(self):
        """Test before_model chunk with jump_to=end yields final_answer event."""
        chunk = {
            "ModelCallLimitMiddleware.before_model": {
                "jump_to": "end",
                "messages": [AIMessage(content="Model call limits exceeded: ...")],
            }
        }

        event = _process_chunk(chunk)

        assert event is not None
        assert event.type == "final_answer"
        assert event.data.content == ErrorMessage.MODEL_CALL_LIMIT_REACHED

    def test_model_call_limit_passthrough_chunk_returns_none(self):
        """Test before_model passthrough chunk (None payload) returns None."""
        chunk = {"ModelCallLimitMiddleware.before_model": None}
        assert _process_chunk(chunk) is None

    def test_model_call_limit_passthrough_chunk_no_jump_returns_none(self):
        """Test before_model chunk without jump_to returns None."""
        chunk = {"ModelCallLimitMiddleware.before_model": {"messages": []}}
        assert _process_chunk(chunk) is None

    def test_unrecognized_chunk_returns_none(self):
        """Test unrecognized chunk returns None."""
        chunk = {"unknown_node": {"data": "something"}}
        event = _process_chunk(chunk)
        assert event is None

    def test_empty_chunk_returns_none(self):
        """Test empty chunk returns None."""
        chunk = {}
        event = _process_chunk(chunk)
        assert event is None


class TestSseForwarder:
    async def test_forwards_until_complete_then_exits(self):
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        run_id = str(uuid.uuid4())
        await queue.put(StreamEvent(type="final_answer", data=EventData(content="hi")))
        await queue.put(StreamEvent(type="complete", data=EventData(run_id=run_id)))

        out = []
        async for sse in sse_forwarder(queue):
            out.append(sse)

        assert len(out) == 2
        assert '"type":"final_answer"' in out[0]
        assert '"type":"complete"' in out[1]
        assert run_id in out[1]

    async def test_forwards_error_event_before_complete(self):
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        await queue.put(StreamEvent(type="error", data=EventData(content="bad")))
        await queue.put(
            StreamEvent(type="complete", data=EventData(run_id=str(uuid.uuid4())))
        )

        out = []
        async for sse in sse_forwarder(queue):
            out.append(sse)

        assert '"type":"error"' in out[0]
        assert '"type":"complete"' in out[1]
