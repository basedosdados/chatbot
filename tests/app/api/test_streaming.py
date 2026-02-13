import json
import uuid
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core import exceptions as google_api_exceptions
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pytest_mock import MockerFixture

from app.api.schemas import ConfigDict
from app.api.streaming import (
    ErrorMessage,
    _process_chunk,
    _truncate_json,
    stream_response,
)
from app.db.models import Message, MessageRole, MessageStatus


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
        data = {"long_string": "a" * self.STR_LONG_LEN}
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string, max_str_len=self.STR_MAX_LEN)
        expected_str = (
            "a" * self.STR_MAX_LEN + f"... ({self.STR_REMAINING} more characters)"
        )
        expected_json = self._format_json({"long_string": expected_str})
        assert truncated == expected_json

    def test_truncate_json_long_list(self):
        data = {"long_list": list(range(self.LIST_LONG_LEN))}
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string, max_list_len=self.LIST_MAX_LEN)
        expected_list = list(range(self.LIST_MAX_LEN)) + [
            f"... ({self.LIST_REMAINING} more items)"
        ]
        expected_json = self._format_json({"long_list": expected_list})
        assert truncated == expected_json

    def test_truncate_json_nested(self):
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
        data = list(range(self.LIST_LONG_LEN))
        json_string = json.dumps(data)
        truncated = _truncate_json(json_string)
        assert truncated == json_string

    def test_truncate_json_not_needed(self):
        data = {
            "short_string": "hello",
            "short_list": [1, 2, 3],
        }
        json_string = json.dumps(data)
        expected_json = self._format_json(data)
        assert _truncate_json(json_string) == expected_json

    def test_truncate_json_invalid(self):
        invalid_json_string = '{"key": "value"'
        assert _truncate_json(invalid_json_string) == invalid_json_string


class TestProcessChunk:
    """Tests for _process_chunk function."""

    def test_agent_chunk_with_tool_calls(self):
        """Test agent chunk with tool calls returns tool_call event."""
        chunk = {
            "agent": {
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
            "agent": {
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
        chunk = {"agent": {"messages": [AIMessage(content="Here is your answer.")]}}

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
        chunk = {"agent": {"messages": []}}

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


class TestStreamResponse:
    """Tests for stream_response function."""

    @pytest.fixture
    def mock_thread_id(self) -> str:
        return str(uuid.uuid4())

    @pytest.fixture
    def mock_model_uri(self) -> str:
        return "mock-model"

    @pytest.fixture
    def mock_config(self, mock_thread_id: str) -> ConfigDict:
        return {
            "run_id": uuid.uuid4(),
            "recursion_limit": 10,
            "configurable": {"thread_id": mock_thread_id},
        }

    @pytest.fixture
    def mock_user_message(self, mock_thread_id: str, mock_model_uri: str) -> Message:
        return Message(
            thread_id=mock_thread_id,
            model_uri=mock_model_uri,
            role=MessageRole.USER,
            content="Hello",
            status=MessageStatus.SUCCESS,
        )

    @pytest.fixture
    def mock_database(self, mock_config: ConfigDict, mock_user_message: Message):
        db = MagicMock()

        created_message = Message(
            id=mock_config["run_id"],
            thread_id=mock_user_message.thread_id,
            user_message_id=mock_user_message.id,
            model_uri=mock_user_message.model_uri,
            role=MessageRole.ASSISTANT,
            content="Mock response",
            status=MessageStatus.SUCCESS,
        )

        db.create_message = AsyncMock(return_value=created_message)

        return db

    @staticmethod
    async def _collect_events(async_gen: AsyncIterator[str]) -> list[str]:
        """Helper to collect all events from async generator."""
        events = []
        async for event in async_gen:
            events.append(event)
        return events

    async def test_stream_response_happy_path(
        self,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test successful streaming: skips 'values' mode, collects artifacts, yields all events."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="Let me search.",
                                tool_calls=[
                                    {"id": "call_1", "name": "search", "args": {}},
                                    {"id": "call_2", "name": "lookup", "args": {}},
                                ],
                            )
                        ]
                    }
                },
            )
            yield "values", {"messages": ["msg1"]}
            yield (
                "updates",
                {"unknown_node": {}},
            )  # Unrecognized chunk, _process_chunk returns None
            yield (
                "updates",
                {
                    "tools": [
                        {
                            "messages": [
                                ToolMessage(
                                    content='{"result": "data"}',
                                    tool_call_id="call_1",
                                    name="search",
                                    status="success",
                                    artifact={"url": "http://example.com"},
                                )
                            ]
                        },
                        {
                            "messages": [
                                ToolMessage(
                                    content='{"id": "123"}',
                                    tool_call_id="call_2",
                                    name="lookup",
                                    status="success",
                                    artifact=None,  # No artifact
                                )
                            ]
                        },
                    ]
                },
            )
            yield "values", {"messages": ["msg1", "msg2"]}
            yield (
                "updates",
                {"agent": {"messages": [AIMessage(content="Here is your answer.")]}},
            )
            yield "values", {"messages": ["msg1", "msg2", "msg3"]}

        mock_agent.astream = mock_astream

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 4
        assert '"type":"tool_call"' in events[0]
        assert '"type":"tool_output"' in events[1]
        assert '"type":"final_answer"' in events[2]
        assert '"type":"complete"' in events[3]

        mock_database.create_message.assert_called_once()
        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.artifacts == [
            {"url": "http://example.com"}
        ]  # Only one artifact collected

    async def test_stream_response_generic_exception(
        self,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test generic exception yields error event."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            raise RuntimeError("Something went wrong")
            yield  # Makes this an async generator

        mock_agent.astream = mock_astream

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 2
        assert '"type":"error"' in events[0]
        assert ErrorMessage.UNEXPECTED in events[0]
        assert '"type":"complete"' in events[1]

        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.status == MessageStatus.ERROR
        assert call_args.content == ErrorMessage.UNEXPECTED

    async def test_stream_response_graph_recursion_error(
        self,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test GraphRecursionError sets graceful message without error status."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            raise GraphRecursionError("Recursion limit reached")
            yield  # Makes this an async generator

        mock_agent.astream = mock_astream

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 2
        assert '"type":"final_answer"' in events[0]
        assert ErrorMessage.GRAPH_RECURSION_LIMIT_REACHED in events[0]
        assert '"type":"complete"' in events[1]

        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.status == MessageStatus.SUCCESS
        assert call_args.content == ErrorMessage.GRAPH_RECURSION_LIMIT_REACHED

    async def test_stream_response_google_api_error(
        self,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test Google API InvalidArgument yields error event."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            raise google_api_exceptions.InvalidArgument("Invalid request")
            yield  # Makes this an async generator

        mock_agent.astream = mock_astream

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 2
        assert ErrorMessage.UNEXPECTED in events[0]
        assert '"type":"complete"' in events[1]

        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.status == MessageStatus.ERROR
        assert call_args.content == ErrorMessage.UNEXPECTED

    async def test_stream_response_google_api_error_with_agent_state_below_limit(
        self,
        mocker: MockerFixture,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test Google API error with agent_state set but tokens below limit."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {"messages": ["msg1"]}  # Sets agent_state
            raise google_api_exceptions.InvalidArgument("Some other error")

        mock_agent.astream = mock_astream

        mock_model = MagicMock()
        mock_model.get_num_tokens_from_messages.return_value = 999  # Below limit
        mock_model.profile.get.return_value = 1_048_576  # Gemini context window
        mocker.patch("app.api.streaming.init_chat_model", return_value=mock_model)

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 2
        assert '"type":"error"' in events[0]
        assert ErrorMessage.UNEXPECTED in events[0]  # Not CONTEXT_OVERFLOW
        assert '"type":"complete"' in events[1]

        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.status == MessageStatus.ERROR
        assert call_args.content == ErrorMessage.UNEXPECTED

    async def test_stream_response_google_api_error_with_agent_state_context_overflow(
        self,
        mocker: MockerFixture,
        mock_database,
        mock_user_message,
        mock_config,
        mock_thread_id,
        mock_model_uri,
    ):
        """Test Google API error with context window exceeded."""
        mock_agent = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {"messages": ["msg1"]}  # Sets agent_state
            raise google_api_exceptions.InvalidArgument("Token limit exceeded")

        mock_agent.astream = mock_astream

        mock_model = MagicMock()
        mock_model.get_num_tokens_from_messages.return_value = (
            9_999_999  # Exceeds limit
        )
        mock_model.profile.get.return_value = 1_048_576  # Gemini context window
        mocker.patch("app.api.streaming.init_chat_model", return_value=mock_model)

        events = await self._collect_events(
            stream_response(
                database=mock_database,
                agent=mock_agent,
                user_message=mock_user_message,
                config=mock_config,
                thread_id=mock_thread_id,
                model_uri=mock_model_uri,
            )
        )

        assert len(events) == 2
        assert '"type":"error"' in events[0]
        assert ErrorMessage.CONTEXT_OVERFLOW in events[0]
        assert '"type":"complete"' in events[1]

        call_args = mock_database.create_message.call_args[0][0]
        assert call_args.status == MessageStatus.ERROR
        assert call_args.content == ErrorMessage.CONTEXT_OVERFLOW
