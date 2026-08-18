import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.context import AgentContext
from app.agent.schemas import (
    DataSource,
    StructuredResponse,
)
from app.api.schemas import ConfigDict
from app.api.streaming.agent_runner import (
    _process_chunk,
    _truncate_json,
    run_agent,
)
from app.api.streaming.schemas import StreamEvent
from app.db.models import Message, MessageCreate, MessageRole, MessageStatus
from app.exports import OFFERED_EXPORT_FORMATS
from app.i18n import MessageKey, translate

MODEL_URI = "mock-model"


@pytest.fixture
def thread_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def config(thread_id: str) -> ConfigDict:
    return ConfigDict(
        run_id=str(uuid.uuid4()),
        configurable={"thread_id": thread_id},
    )


@pytest.fixture
def mock_user_message(thread_id: str) -> Message:
    return Message(
        thread_id=thread_id,
        model_uri=MODEL_URI,
        role=MessageRole.USER,
        content="Mock user message",
        status=MessageStatus.SUCCESS,
    )


@pytest.fixture
def mock_database(
    monkeypatch: pytest.MonkeyPatch,
    config: ConfigDict,
    mock_user_message: Message,
) -> MagicMock:
    """Patch AsyncDatabase + sessionmaker so run_agent uses a mock instead
    of opening a real DB connection."""
    db = MagicMock()

    db.create_message = AsyncMock(
        return_value=Message(
            id=config["run_id"],
            thread_id=mock_user_message.thread_id,
            user_message_id=mock_user_message.id,
            model_uri=mock_user_message.model_uri,
            role=MessageRole.ASSISTANT,
            content="Mock assistant message",
            status=MessageStatus.SUCCESS,
        )
    )
    db.create_query_handles = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_sessionmaker():
        yield  # session is unused because AsyncDatabase is itself mocked

    monkeypatch.setattr(
        "app.api.streaming.agent_runner.sessionmaker", mock_sessionmaker
    )

    monkeypatch.setattr(
        "app.api.streaming.agent_runner.AsyncDatabase", lambda session: db
    )

    return db


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

        event = _process_chunk(chunk, "pt")

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

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "tool_call"
        assert len(event.data.tool_calls) == 2
        assert event.data.tool_calls[0].name == "search"
        assert event.data.tool_calls[1].name == "lookup"

    def test_agent_chunk_final_answer(self):
        """Test agent chunk without tool calls returns final_answer event."""
        chunk = {"model": {"messages": [AIMessage(content="Here is your answer.")]}}

        event = _process_chunk(chunk, "pt")

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

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "final_answer"
        assert event.data.content == ""

    def test_agent_chunk_structured_response(self):
        """A model chunk carrying `structured_response` yields a final_answer event
        whose content is the prose and whose structured_response holds all fields."""
        structured = StructuredResponse(
            response="Here is your answer.",
            data_sources=[
                DataSource(dataset_id="ds1", table_id="tb1", name="Tabela 1")
            ],
            follow_up_prompts=["E em 2026?", "Por estado?", "Por região?"],
        )

        # The model node sets `structured_response` alongside the internal
        # structured-output tool call; that tool call must NOT become a tool_call event.
        chunk = {
            "model": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_struct",
                                "name": "StructuredResponse",
                                "args": structured.model_dump(),
                            }
                        ],
                    )
                ],
                "structured_response": structured,
            }
        }

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "final_answer"
        assert event.data.tool_calls is None
        assert event.data.content == "Here is your answer."
        assert event.data.structured_response is not None
        assert event.data.structured_response["response"] == "Here is your answer."
        # `_process_chunk` dumps the model's fields as-is; the authoritative name is resolved
        # later in `run_agent` (see test_structured_response_is_emitted_and_persisted).
        assert event.data.structured_response["data_sources"] == [
            {"dataset_id": "ds1", "table_id": "tb1", "name": "Tabela 1"}
        ]
        assert event.data.structured_response["follow_up_prompts"] == [
            "E em 2026?",
            "Por estado?",
            "Por região?",
        ]

    def test_agent_chunk_structured_response_sanitizes_links(self):
        """The prose in a structured response has its markdown links sanitized."""
        structured = StructuredResponse(response="See [evil](http://evil.com).")

        chunk = {
            "model": {
                "messages": [AIMessage(content="")],
                "structured_response": structured,
            }
        }

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "final_answer"
        assert "http://evil.com" not in event.data.content
        assert event.data.content == event.data.structured_response["response"]

    def test_tools_chunk_single_tool(self):
        """Test tools chunk with single tool output (dict format)."""
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"result": "found"}',
                        tool_call_id="call_123",
                        name="search_datasets",
                        status="success",
                    )
                ]
            }
        }

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "tool_output"
        assert len(event.data.tool_outputs) == 1

        tool_output = event.data.tool_outputs[0]

        assert tool_output.status == "success"
        assert tool_output.tool_call_id == "call_123"
        assert tool_output.tool_name == "search_datasets"
        assert tool_output.content == '{\n  "result": "found"\n}'
        # search_datasets carries no artifact, so the output has none.
        assert tool_output.artifact is None
        assert tool_output.metadata is None

    def test_tools_chunk_projects_query_result_artifact_on_serialization(self):
        """The internal query_result handle stays in memory (so the server can capture
        it) but every serialization — the SSE stream and the persisted events that
        list_messages returns — sees only its download descriptor."""
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"row_count": 1, "rows": [{"col1": "value1"}]}',
                        tool_call_id="call_123",
                        name="execute_bigquery_sql",
                        status="success",
                        artifact={
                            "type": "query_result",
                            "query_ref": "q_abc",
                            "slug": "slug",
                            "destination_table": {"projectId": "p"},
                        },
                    )
                ]
            }
        }

        event = _process_chunk(chunk, "pt")
        output = event.data.tool_outputs[0]

        # In memory the handle is present (run_agent reads destination_table off it) ...
        assert output.artifact["destination_table"] == {"projectId": "p"}
        # ... but the client sees only what it needs to render a download.
        assert output.model_dump()["artifact"] == {
            "type": "query_result",
            "query_ref": "q_abc",
            "slug": "slug",
            "formats": OFFERED_EXPORT_FORMATS,
        }
        assert "destination_table" not in event.to_sse()

    def test_tools_chunk_surfaces_non_query_result_artifact_on_serialization(self):
        """Only `query_result` handles are redacted; any other artifact is surfaced as-is."""
        artifact = {"type": "chart", "id": "c1"}
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"ok": true}',
                        tool_call_id="call_123",
                        name="some_tool",
                        status="success",
                        artifact=artifact,
                    )
                ]
            }
        }

        event = _process_chunk(chunk, "pt")
        output = event.data.tool_outputs[0]

        # A client-facing artifact passes through redaction untouched, in memory ...
        assert output.artifact == artifact
        # ... and on every serialization (SSE stream + persisted events).
        assert output.model_dump()["artifact"] == artifact
        assert "chart" in event.to_sse()

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

        event = _process_chunk(chunk, "pt")

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

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "tool_output"
        assert event.data.tool_outputs[0].status == "error"

    def test_tools_chunk_unexpected_format(self):
        """Test tools chunk with unexpected format returns empty tool_outputs."""
        chunk = {"tools": "unexpected string"}

        event = _process_chunk(chunk, "pt")

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

        event = _process_chunk(chunk, "pt")

        assert event is not None
        assert event.type == "model_call_limit"
        assert event.data.content == translate(MessageKey.ERROR_MODEL_CALL_LIMIT, "pt")

    def test_model_call_limit_passthrough_chunk_returns_none(self):
        """Test before_model passthrough chunk (None payload) returns None."""
        chunk = {"ModelCallLimitMiddleware.before_model": None}
        assert _process_chunk(chunk, "pt") is None

    def test_model_call_limit_passthrough_chunk_no_jump_returns_none(self):
        """Test before_model chunk without jump_to returns None."""
        chunk = {"ModelCallLimitMiddleware.before_model": {"messages": []}}
        assert _process_chunk(chunk, "pt") is None

    def test_unrecognized_chunk_returns_none(self):
        """Test unrecognized chunk returns None."""
        chunk = {"unknown_node": {"data": "something"}}
        event = _process_chunk(chunk, "pt")
        assert event is None

    def test_empty_chunk_returns_none(self):
        """Test empty chunk returns None."""
        chunk = {}
        event = _process_chunk(chunk, "pt")
        assert event is None


class TestRunAgent:
    """Tests for run_agent function."""

    async def _drain(self, queue: asyncio.Queue[StreamEvent]) -> list[StreamEvent]:
        events = []
        while True:
            event = await queue.get()
            events.append(event)
            if event.type == "complete":
                return events

    async def test_forwards_context_to_agent(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """The run context must reach `agent.astream` — that's how tools and the
        system-prompt middleware receive the thread's language/user/thread ids."""
        agent = MagicMock()
        seen = {}

        async def astream(*args, **kwargs):
            seen["context"] = kwargs.get("context")
            yield ("updates", {"model": {"messages": [AIMessage(content="ok")]}})

        agent.astream = astream
        context = AgentContext(thread_id=thread_id, user_id="u", language="es")
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=context,
            queue=queue,
        )

        assert seen["context"] is context

    async def test_localizes_error_message_to_context_language(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """A crash mid-run persists the error message in the thread's language."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover — make this an async generator

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(thread_id=thread_id, user_id="u", language="es"),
            queue=queue,
        )

        message = mock_database.create_message.call_args[0][0]
        assert message.status == MessageStatus.ERROR
        assert message.content == translate(MessageKey.ERROR_UNEXPECTED, "es")

    async def test_emits_events_and_persists_success(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Test happy path emits events and persists success message."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="Final answer")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["final_answer", "complete"]
        assert events[0].data.content == "Final answer"
        assert events[-1].data.run_id == config["run_id"]

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.SUCCESS
        assert message.content == "Final answer"

    async def test_handle_persist_failure_still_persists_message(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """A failed query-handle write must not lose the message (handles are best-effort)."""
        mock_database.create_query_handles = AsyncMock(
            side_effect=RuntimeError("handle boom")
        )

        destination = {"projectId": "p", "datasetId": "d", "tableId": "t"}
        structured = StructuredResponse(response="Answer")

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content='{"row_count": 1, "rows": [{"col1": "value1"}]}',
                                tool_call_id="1",
                                name="execute_bigquery_sql",
                                status="success",
                                artifact={
                                    "type": "query_result",
                                    "query_ref": "q_run",
                                    "slug": "slug",
                                    "destination_table": destination,
                                },
                            )
                        ]
                    }
                },
            )
            yield ("updates", {"model": {"structured_response": structured}})

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        complete = (await self._drain(queue))[-1]

        # The handle write was attempted and failed, but the message still persisted
        # and the run completes without a persistence error.
        mock_database.create_message.assert_called_once()
        mock_database.create_query_handles.assert_awaited_once()
        assert complete.type == "complete"
        assert complete.data.run_id == config["run_id"]
        assert complete.data.error_details is None

    async def test_structured_response_is_emitted_and_persisted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """A structured final answer is streamed and persisted on the message row,
        with each data source's display name resolved from its table UUID."""
        structured = StructuredResponse(
            response="Final answer",
            data_sources=[
                DataSource(dataset_id="ds1", table_id="tb1", name="model fallback")
            ],
            follow_up_prompts=["E em 2026?"],
        )

        async def fake_resolve(structured_response: dict[str, Any], language: str):
            for source in structured_response.get("data_sources") or []:
                source["name"] = "Conjunto DS1 - Tabela TB1"

        resolve_names = AsyncMock(side_effect=fake_resolve)

        monkeypatch.setattr(
            "app.api.streaming.agent_runner.resolve_data_source_names", resolve_names
        )

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "model": {
                        "messages": [AIMessage(content="")],
                        "structured_response": structured,
                    }
                },
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["final_answer", "complete"]
        assert events[0].data.content == "Final answer"
        assert events[-1].data.run_id == config["run_id"]

        assert events[0].data.structured_response is not None
        assert events[0].data.structured_response["response"] == "Final answer"
        resolve_names.assert_awaited_once()
        assert events[0].data.structured_response["data_sources"] == [
            {
                "dataset_id": "ds1",
                "table_id": "tb1",
                "name": "Conjunto DS1 - Tabela TB1",
            }
        ]
        assert events[0].data.structured_response["follow_up_prompts"] == ["E em 2026?"]

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.SUCCESS
        assert message.content == "Final answer"
        assert message.structured_response == events[0].data.structured_response

    async def test_executed_query_derives_download_and_persists_handle(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """An executed query streams its download descriptor and persists its handle.

        Lazy model: no file is exported at answer time. The affordance rides on the
        tool output that produced the query, and the handle is stored so the file can
        be materialized on the first download click.
        """
        destination = {"projectId": "p", "datasetId": "d", "tableId": "t"}
        structured = StructuredResponse(response="Answer")

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content='{"row_count": 1, "rows": [{"col1": "value1"}]}',
                                tool_call_id="1",
                                name="execute_bigquery_sql",
                                status="success",
                                artifact={
                                    "type": "query_result",
                                    "query_ref": "q_run",
                                    "slug": "slug",
                                    "destination_table": destination,
                                },
                            )
                        ]
                    }
                },
            )
            yield ("updates", {"model": {"structured_response": structured}})

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )
        events = await self._drain(queue)
        tool_output = next(e for e in events if e.type == "tool_output")

        # The affordance rides on the tool output, so the client can offer the download
        # inline with the query that produced it ...
        assert tool_output.data.tool_outputs[0].model_dump()["artifact"] == {
            "type": "query_result",
            "query_ref": "q_run",
            "slug": "slug",
            "formats": OFFERED_EXPORT_FORMATS,
        }
        # ... and the handle (slug + destination table) is stored, with no eager file export.
        mock_database.create_query_handles.assert_awaited_once()
        [handle] = mock_database.create_query_handles.call_args.args[0]
        assert handle.query_ref == "q_run"
        assert handle.slug == "slug"
        assert handle.destination_table == destination
        assert events[-1].data.error_details is None

    async def test_failed_run_still_persists_the_handles_it_streamed(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """A run that executes a query and then fails still persists its handle.

        The client offers a download for every streamed query, so persistence follows
        the handles, not the run's outcome — otherwise those buttons 404 on click.
        """
        destination = {"projectId": "p", "datasetId": "d", "tableId": "t"}

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content='{"row_count": 1, "rows": [{"col1": "value1"}]}',
                                tool_call_id="1",
                                name="execute_bigquery_sql",
                                status="success",
                                artifact={
                                    "type": "query_result",
                                    "query_ref": "q_run",
                                    "slug": "slug",
                                    "destination_table": destination,
                                },
                            )
                        ]
                    }
                },
            )
            raise RuntimeError("error")

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["tool_output", "error", "complete"]

        # No final answer, but the streamed query is still downloadable.
        mock_database.create_query_handles.assert_awaited_once()
        [handle] = mock_database.create_query_handles.call_args.args[0]
        assert handle.query_ref == "q_run"
        assert handle.destination_table == destination

    async def test_unexpected_exception_persists_error_row(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Test unexpected exceptions are handled properly."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            raise RuntimeError("error")
            yield  # make this a generator

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["error", "complete"]
        assert events[0].data.content == translate(MessageKey.ERROR_UNEXPECTED, "pt")
        assert events[0].data.error_details == {"reason": "agent_failed"}

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.ERROR
        assert message.content == translate(MessageKey.ERROR_UNEXPECTED, "pt")

    async def test_model_call_limit_persists_with_dedicated_status(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Test ModelCallLimit error is handled properly."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"ModelCallLimitMiddleware.before_model": {"jump_to": "end"}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["model_call_limit", "complete"]
        assert events[0].data.content == translate(
            MessageKey.ERROR_MODEL_CALL_LIMIT, "pt"
        )
        assert events[-1].data.run_id == config["run_id"]

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.MODEL_CALL_LIMIT
        assert message.content == translate(MessageKey.ERROR_MODEL_CALL_LIMIT, "pt")

    async def test_complete_still_emitted_when_db_write_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """If `database.create_message` raises, the consumer must still
        receive a `complete` event - otherwise it hangs on `queue.get()`.
        """
        db = MagicMock()
        db.create_message = AsyncMock(side_effect=RuntimeError("db down"))

        @asynccontextmanager
        async def mock_sessionmaker():
            yield None

        monkeypatch.setattr(
            "app.api.streaming.agent_runner.sessionmaker", mock_sessionmaker
        )

        monkeypatch.setattr(
            "app.api.streaming.agent_runner.AsyncDatabase", lambda session: db
        )

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="Final answer")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            agent=agent,
            config=config,
            thread_id=thread_id,
            user_message=mock_user_message,
            model_uri=MODEL_URI,
            context=AgentContext(
                thread_id="test-thread", user_id="test-user", language="pt"
            ),
            queue=queue,
        )

        events = await self._drain(queue)
        assert [e.type for e in events] == ["final_answer", "complete"]
        assert events[0].data.content == "Final answer"

        complete = events[-1]
        assert complete.type == "complete"
        assert complete.data.run_id is None
        assert complete.data.error_details == {"reason": "persistence_failed"}
        assert "db down" not in str(complete.data.error_details)

    async def test_consumer_cancel_does_not_cancel_producer(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Test the producer task runs to completion even if no one drains the queue."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="Final answer")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        task = asyncio.create_task(
            run_agent(
                agent=agent,
                config=config,
                thread_id=thread_id,
                user_message=mock_user_message,
                model_uri=MODEL_URI,
                context=AgentContext(
                    thread_id="test-thread", user_id="test-user", language="pt"
                ),
                queue=queue,
            )
        )

        # Simulate the consumer never attaching: just await the producer
        await asyncio.wait_for(task, timeout=2.0)

        # Producer persisted the message regardless of consumer presence
        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.SUCCESS
        assert message.content == "Final answer"

        # The complete event is sitting in the queue waiting
        events = await self._drain(queue)
        assert events[-1].type == "complete"

    async def test_cancellation_before_final_answer_persists_interrupted(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Cancelling the producer before any final_answer persists row with
        INTERRUPTED content + INTERRUPTED status, and re-raises CancelledError."""
        agent = MagicMock()
        started = asyncio.Event()

        async def astream(*args, **kwargs):
            started.set()
            # Block until cancelled — never yields a chunk
            await asyncio.sleep(60)
            yield  # make this a generator

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        task = asyncio.create_task(
            run_agent(
                agent=agent,
                config=config,
                thread_id=thread_id,
                user_message=mock_user_message,
                model_uri=MODEL_URI,
                context=AgentContext(
                    thread_id="test-thread", user_id="test-user", language="pt"
                ),
                queue=queue,
            )
        )

        # Wait until the producer reach the await inside agent.astream
        await started.wait()

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.INTERRUPTED
        assert message.content == translate(MessageKey.ERROR_INTERRUPTED, "pt")

    async def test_cancellation_after_final_answer_preserves_success(
        self,
        mock_database: MagicMock,
        mock_user_message: Message,
        config: ConfigDict,
        thread_id: str,
    ):
        """Cancelling the producer after a final_answer has been processed
        preserves the SUCCESS status — the CancelledError branch only sets
        INTERRUPTED when no status has been observed yet."""
        agent = MagicMock()
        processed = asyncio.Event()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="Final answer")]}},
            )
            # Reached when the producer asks for the next chunk, i.e., after it
            # has set status=SUCCESS for the final_answer above.
            processed.set()
            await asyncio.sleep(60)

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        task = asyncio.create_task(
            run_agent(
                agent=agent,
                config=config,
                thread_id=thread_id,
                user_message=mock_user_message,
                model_uri=MODEL_URI,
                context=AgentContext(
                    thread_id="test-thread", user_id="test-user", language="pt"
                ),
                queue=queue,
            )
        )

        # Wait until the final_answer event is processed
        await processed.wait()

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()

        mock_database.create_message.assert_called_once()
        message = mock_database.create_message.call_args[0][0]
        assert isinstance(message, MessageCreate)
        assert message.status == MessageStatus.SUCCESS
        assert message.content == "Final answer"
