import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.agent.context import AgentContext
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
from app.exports import collect_query_handles
from app.i18n import LanguageCode, MessageKey, translate


def _truncate_json(
    json_string: str, max_list_len: int = 10, max_str_len: int = 300
) -> str:
    """Shorten a serialized JSON object's long lists and strings, with placeholders.

    Non-dict JSON is returned unchanged.

    Args:
        json_string (str): The serialized JSON to process.
        max_list_len (int, optional): Max items to keep in a list. Defaults to 10.
        max_str_len (int, optional): Max length for any single string. Defaults to 300.

    Returns:
        str: The truncated, re-serialized JSON object.
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


def _process_chunk(chunk: dict[str, Any], language: LanguageCode) -> StreamEvent | None:
    """Turn a raw agent stream chunk into a StreamEvent.

    Args:
        chunk (dict[str, Any]): A raw update chunk from the agent workflow.
        language (LanguageCode): Language for localizing server-emitted content.

    Returns:
        StreamEvent | None: The tool_call, tool_output, final_answer or model_call_limit
            event, or None for an ignored chunk.
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
                content=translate(MessageKey.ERROR_MODEL_CALL_LIMIT, language),
                tool_calls=None,
            )
            return StreamEvent(type="model_call_limit", data=event_data)
    return None


async def _create_placeholder_message(
    *,
    run_id: str,
    thread_id: str,
    user_message: Message,
    model_uri: str,
) -> None:
    """Create the assistant row up front (id == run_id, STREAMING) so query handles
    can be persisted against a real FK during the run.

    Raises on failure so the caller aborts the run: persistence is deterministic,
    and `_finalize_message` only ever updates this row.

    Args:
        run_id (str): The run id, reused as the message id.
        thread_id (str): The thread the message belongs to.
        user_message (Message): The user message driving the run.
        model_uri (str): Model URI.
    """
    message_create = MessageCreate(
        id=run_id,
        thread_id=thread_id,
        user_message_id=user_message.id,
        model_uri=model_uri,
        role=MessageRole.ASSISTANT,
        content="",
        status=MessageStatus.STREAMING,
    )
    async with sessionmaker() as session:
        await AsyncDatabase(session).create_message(message_create)


async def _persist_query_handles(
    *,
    run_id: str,
    tool_outputs: list[ToolOutput],
) -> None:
    """Persist the query handles carried by the tool outputs, scoped to the run's message.

    Best-effort: a failure is logged and never interrupts the run. No dedup — a repeated
    query_ref would be a bug, so a duplicate-key insert surfaces it rather than hiding it.

    Args:
        run_id (str): The owning message/run the handles are scoped to.
        tool_outputs (list[ToolOutput]): The tool outputs to scan for query_result artifacts.
    """
    collected = collect_query_handles(output.artifact for output in tool_outputs)

    if not collected:
        return

    handles = [
        QueryHandle(
            query_ref=handle.query_ref,
            message_id=run_id,
            slug=handle.slug,
            destination_table=handle.destination_table,
        )
        for handle in collected
    ]

    try:
        async with sessionmaker() as session:
            await AsyncDatabase(session).create_query_handles(handles)
    except Exception:
        logger.exception(f"Failed to persist query handles for run {run_id}:")


async def _finalize_message(
    *,
    run_id: str,
    content: str,
    events: list[dict[str, Any]] | None,
    structured_response: dict[str, Any] | None,
    status: MessageStatus,
) -> tuple[str | None, dict[str, Any] | None]:
    """Write the terminal state onto the placeholder created at run start.

    The placeholder is guaranteed to exist (its creation gates the run), so this only updates.

    Args:
        run_id (str): The run/message id to finalize.
        content (str): The assistant message content.
        events (list[dict[str, Any]] | None): The streamed events to persist.
        structured_response (dict[str, Any] | None): The structured response, if any.
        status (MessageStatus): The terminal status to write.

    Returns:
        tuple[str | None, dict[str, Any] | None]: (message_id, None) on success, or
            (None, error_details) if the row could not be written.
    """
    try:
        async with sessionmaker() as session:
            message = await AsyncDatabase(session).update_message(
                run_id,
                content=content,
                events=events,
                structured_response=structured_response,
                status=status,
            )
        if message is None:
            logger.error(
                f"Placeholder message for run {run_id} vanished before finalize"
            )
            return None, {"reason": "persistence_failed"}
        return str(message.id), None
    except Exception:
        logger.exception(f"Failed to persist assistant message for run {run_id}:")
        return None, {"reason": "persistence_failed"}


async def run_agent(
    agent: CompiledStateGraph,
    config: ConfigDict,
    context: AgentContext,
    thread_id: str,
    user_message: Message,
    model_uri: str,
    queue: asyncio.Queue[StreamEvent],
):
    """Run the agent to completion, streaming events onto the queue and owning persistence.

    Creates the assistant row up front (STREAMING), persists query handles as tool outputs
    arrive, writes the terminal state in `finally`, and emits exactly one `complete` event.

    Args:
        agent (CompiledStateGraph): The compiled agent graph.
        config (ConfigDict): Config for agent execution.
        context (AgentContext): The run context.
        thread_id (str): Thread identifier.
        user_message (Message): The user message driving the run.
        model_uri (str): Model URI.
        queue (asyncio.Queue[StreamEvent]): Queue the events are pushed onto.
    """
    run_id = config["run_id"]
    events = []
    assistant_message = ""
    structured_response: dict[str, Any] | None = None
    status: MessageStatus | None = None

    # Create the assistant row up front so query handles persist eagerly against a real FK
    # during the run. It stays STREAMING until the finally block writes the terminal state.
    try:
        await _create_placeholder_message(
            run_id=run_id,
            thread_id=thread_id,
            user_message=user_message,
            model_uri=model_uri,
        )
    except Exception:
        logger.exception(f"Failed to create placeholder message for run {run_id}:")
        error_details = {"reason": "persistence_failed"}
        await queue.put(
            StreamEvent(
                type="error",
                data=EventData(
                    content=translate(MessageKey.ERROR_UNEXPECTED, context.language),
                    error_details=error_details,
                ),
            )
        )
        await queue.put(
            StreamEvent(
                type="complete",
                data=EventData(run_id=None, error_details=error_details),
            )
        )
        return

    try:
        async for mode, chunk in agent.astream(  # pragma: no cover
            input={"messages": [{"role": "user", "content": user_message.content}]},
            config=config,
            context=context,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                continue

            event = _process_chunk(chunk, context.language)

            if event is None:
                continue

            if event.type == "tool_output":
                # Persist each query handle as its tool output arrives, so a later
                # tool in the same run (or a later turn) can resolve it from the DB.
                await _persist_query_handles(
                    run_id=run_id,
                    tool_outputs=event.data.tool_outputs,
                )
            elif event.type == "final_answer":
                # Resolve data source names to a localized "{dataset_name} — {table_name}".
                if event.data.structured_response is not None:
                    await resolve_data_source_names(
                        event.data.structured_response, context.language
                    )
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
            assistant_message = translate(
                MessageKey.ERROR_INTERRUPTED, context.language
            )
            status = MessageStatus.INTERRUPTED
        raise
    except Exception:
        logger.exception(f"Unexpected error in run {config['run_id']}:")
        assistant_message = translate(MessageKey.ERROR_UNEXPECTED, context.language)
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
        message_id, error_details = await _finalize_message(
            run_id=run_id,
            content=assistant_message,
            events=events or None,
            structured_response=structured_response,
            status=status or MessageStatus.ERROR,
        )
        await queue.put(
            StreamEvent(
                type="complete",
                data=EventData(run_id=message_id, error_details=error_details),
            )
        )
