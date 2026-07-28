import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.agent.schemas import StructuredResponse
from app.api.schemas import ConfigDict
from app.api.streaming.data_sources import resolve_data_source_names
from app.api.streaming.schemas import EventData, StreamEvent, ToolCall, ToolOutput
from app.api.streaming.security import sanitize_markdown_links
from app.db.database import AsyncDatabase, sessionmaker
from app.db.models import (
    Message,
    MessageCreate,
    MessageRole,
    MessageStatus,
    QueryHandle,
)
from app.exports import CollectedQueryHandle, collect_query_handles


class ErrorMessage:
    INTERRUPTED = (
        "A conexão com o servidor foi interrompida. Por favor, tente novamente."
    )

    MODEL_CALL_LIMIT_REACHED = (
        "Essa pergunta gerou um raciocínio muito longo e não consegui chegar a uma conclusão. "
        "Por favor, tente ser mais específico ou divida sua pergunta em partes menores."
    )

    UNEXPECTED = "Ocorreu um erro inesperado. Por favor, tente novamente. Se o problema persistir, avise-nos."


def _truncate_json(
    json_string: str, max_list_len: int = 10, max_str_len: int = 300
) -> str:
    """Iteratively truncates a serialized JSON object by shortening lists and strings
    and adding human-readable placeholders.

    Note:
        This function only processes JSON objects (dictionaries). If the serialized JSON
        represents any other type, the original JSON string will be returned unchanged.

    Args:
        json_string (str): The serialized JSON to process.
        max_list_len (int, optional): The max number of items to keep in a list. Defaults to 10.
        max_str_len (int, optional): The max length for any single string. Defaults to 300.

    Returns:
        str: The truncated, formatted, and serialized JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        return json_string

    if not isinstance(data, dict):
        return json_string

    stack = [data]

    while stack:
        current_node = stack.pop()

        if isinstance(current_node, dict):
            items_to_process = current_node.items()
        else:
            items_to_process = enumerate(current_node)

        for key_or_idx, item in items_to_process:
            if isinstance(item, str):
                if len(item) > max_str_len:
                    truncated_str = (
                        item[:max_str_len]
                        + f"... ({len(item) - max_str_len} more characters)"
                    )
                    current_node[key_or_idx] = truncated_str

            elif isinstance(item, list):
                if len(item) > max_list_len:
                    original_len = len(item)
                    del item[max_list_len:]
                    item.append(f"... ({original_len - max_list_len} more items)")
                stack.append(item)

            elif isinstance(item, dict):
                stack.append(item)

    return json.dumps(data, ensure_ascii=False, indent=2)


def _process_chunk(chunk: dict[str, Any]) -> StreamEvent | None:
    """Process a streaming chunk from a react agent workflow into a StreamEvent.

    Args:
        chunk (dict[str, Any]): A raw update chunk from the agent workflow.

    Returns:
        StreamEvent | None: Structured event or None if the chunk is ignored:
            - "tool_call" for agent messages with tool calls
            - "tool_output" for tool execution results
            - "final_answer" for agent messages without tool calls
            - "model_call_limit" when the model call limit is reached
            - None for ignored chunks
    """
    if "model" in chunk:
        update: dict[str, Any] = chunk["model"]

        # When `response_format` is set (see app.main:91), the model node sets `structured_response`
        # (a StructuredResponse) on the turn it produces the final answer. This is the final answer;
        # the accompanying structured-output tool call / ToolMessage in `update["messages"]` is
        # internal and must not be emitted as a tool_call.
        structured: StructuredResponse | None = update.get("structured_response")

        if structured is not None:
            response_text = sanitize_markdown_links(structured.response)
            structured_response = structured.model_dump()
            structured_response["response"] = response_text
            return StreamEvent(
                type="final_answer",
                data=EventData(
                    content=response_text,
                    structured_response=structured_response,
                ),
            )

        ai_messages: list[AIMessage] = update["messages"]

        # If no messages are returned, the model returned an empty response
        # with no tool calls. This also counts as a final (but empty) answer.
        if not ai_messages:
            return StreamEvent(type="final_answer", data=EventData(content=""))

        message = ai_messages[0]

        if message.tool_calls:
            event_type = "tool_call"
            tool_calls = [
                ToolCall(
                    id=tool_call["id"], name=tool_call["name"], args=tool_call["args"]
                )
                for tool_call in message.tool_calls
            ]
            content = message.text
        else:
            event_type = "final_answer"
            tool_calls = None
            content = sanitize_markdown_links(message.text)

        event_data = EventData(content=content, tool_calls=tool_calls)

        return StreamEvent(type=event_type, data=event_data)
    elif "tools" in chunk:
        updates = chunk["tools"]

        # single tool call
        if isinstance(updates, dict):
            tool_messages: list[ToolMessage] = updates["messages"]

        # multiple parallel tool calls
        elif isinstance(updates, list):
            tool_messages: list[ToolMessage] = [
                update["messages"][0] for update in updates if "messages" in update
            ]

        # defensive handling (langgraph should only output dicts and lists)
        else:
            tool_messages = []

        tool_outputs = [
            ToolOutput(
                status=message.status,
                tool_call_id=message.tool_call_id,
                tool_name=message.name,
                content=_truncate_json(message.content),
                artifact=message.artifact,  # internal artifacts are redacted on serialization (see ToolOutput)
            )
            for message in tool_messages
        ]

        return StreamEvent(
            type="tool_output",
            data=EventData(tool_outputs=tool_outputs),
        )
    elif "ModelCallLimitMiddleware.before_model" in chunk:
        # before_model runs on every model iteration; only the limit-exceeded
        # path sets jump_to="end", so check that rather than the key's presence.
        update = chunk["ModelCallLimitMiddleware.before_model"] or {}
        if update.get("jump_to") == "end":
            event_data = EventData(
                content=ErrorMessage.MODEL_CALL_LIMIT_REACHED,
                tool_calls=None,
            )
            return StreamEvent(type="model_call_limit", data=event_data)
    return None


async def _persist_query_handles(
    database: AsyncDatabase,
    *,
    message_id: str,
    collected_handles: list[CollectedQueryHandle],
):
    """Persist query handles for a message.

    Args:
        database (AsyncDatabase): The repository to persist through.
        message_id (str): The owning message/run the handles are scoped to.
        collected_handles: list[CollectedQueryHandle]: Collected query handles.
    """
    handles = [
        QueryHandle(
            query_ref=handle.query_ref,
            message_id=message_id,
            slug=handle.slug,
            destination_table=handle.destination_table,
        )
        for handle in collected_handles
    ]
    await database.create_query_handles(handles)


async def run_agent(
    agent: CompiledStateGraph,
    config: ConfigDict,
    thread_id: str,
    user_message: Message,
    model_uri: str,
    queue: asyncio.Queue[StreamEvent],
):
    """Run the agent to completion and push events onto the queue.

    Owns persistence: writes the assistant `messages` row in `finally` and
    emits a terminal `complete` event carrying either the persisted run_id
    (on success) or `error_details` (if persistence fails). Exactly one
    `complete` event is emitted per run.

    Args:
        agent (CompiledStateGraph): Agent compiled state graph.
        config (ConfigDict): Config for agent execution.
        thread_id (str): Thread unique identifier.
        user_message (Message): User message.
        model_uri (str): Model URI.
        queue (asyncio.Queue[StreamEvent]): Events queue.
    """
    events = []
    assistant_message = ""
    structured_response: dict[str, Any] | None = None
    collected_handles: list[CollectedQueryHandle] = []
    status: MessageStatus | None = None

    try:
        async for mode, chunk in agent.astream(  # pragma: no cover
            input={"messages": [{"role": "user", "content": user_message.content}]},
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                continue

            event = _process_chunk(chunk)

            if event is None:
                continue

            if event.type == "tool_output":
                # Collect every query handle from the execute_bigquery_sql
                # tool artifacts, for lazy on-click downloads.
                collect_query_handles(
                    (output.artifact for output in event.data.tool_outputs),
                    collected_handles,
                )
            elif event.type == "final_answer":
                # Resolve data source names to {dataset_name}—{table_name}
                if event.data.structured_response is not None:
                    await resolve_data_source_names(event.data.structured_response)
                structured_response = event.data.structured_response
                # Set the assistant message
                assistant_message = event.data.content
                status = MessageStatus.SUCCESS
            elif event.type == "model_call_limit":
                assistant_message = event.data.content
                status = MessageStatus.MODEL_CALL_LIMIT

            events.append(event.model_dump())
            await queue.put(event)
    except asyncio.CancelledError:
        if status is None:
            assistant_message = ErrorMessage.INTERRUPTED
            status = MessageStatus.INTERRUPTED
        raise
    except Exception:
        logger.exception(f"Unexpected error in run {config['run_id']}:")
        assistant_message = ErrorMessage.UNEXPECTED
        status = MessageStatus.ERROR
        event = StreamEvent(
            type="error",
            data=EventData(
                content=assistant_message,
                error_details={"reason": "agent_failed"},
            ),
        )
        events.append(event.model_dump())
        await queue.put(event)
    finally:
        message_create = MessageCreate(
            id=config["run_id"],
            thread_id=thread_id,
            user_message_id=user_message.id,
            model_uri=model_uri,
            role=MessageRole.ASSISTANT,
            content=assistant_message,
            events=events or None,
            structured_response=structured_response,
            status=status or MessageStatus.ERROR,
        )
        try:
            async with sessionmaker() as session:
                database = AsyncDatabase(session)
                message = await database.create_message(message_create)
                message_id = str(message.id)
                error_details = None
                # Query handles are a best-effort download convenience, persisted after
                # the message (its own commit) so a handle failure never loses the message.
                if collected_handles:
                    try:
                        await _persist_query_handles(
                            database,
                            message_id=message_id,
                            collected_handles=collected_handles,
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to persist query handles for run {config['run_id']}:"
                        )
        except Exception:
            logger.exception(
                f"Failed to persist assistant message for run {config['run_id']}:"
            )
            message_id = None
            error_details = {"reason": "persistence_failed"}
        await queue.put(
            StreamEvent(
                type="complete",
                data=EventData(run_id=message_id, error_details=error_details),
            )
        )
