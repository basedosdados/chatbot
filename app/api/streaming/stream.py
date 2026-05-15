import asyncio
from typing import AsyncIterator

from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.api.schemas import ConfigDict
from app.api.streaming.agent_runner import (
    ErrorMessage,
    _process_chunk,
)
from app.api.streaming.schemas import EventData, StreamEvent
from app.db.database import AsyncDatabase
from app.db.models import Message, MessageCreate, MessageRole, MessageStatus


async def stream_response(
    database: AsyncDatabase,
    agent: CompiledStateGraph,
    user_message: Message,
    config: ConfigDict,
    thread_id: str,
    model_uri: str,
) -> AsyncIterator[str]:
    """Stream ReAct Agent's execution progress.

    Args:
        message (str): User's input message.
        config (ConfigDict): Configuration for the agent's execution.
        thread_id (str): Unique identifier for the conversation thread.

    Yields:
        Iterator[str]: JSON string containing the streaming status and the current step data.

    Raises:
        asyncio.CancelledError: Re-raised when the client disconnects mid-stream
            to honor cooperative cancellation — the surrounding ASGI cancel scope
            must observe it to unwind cleanly. The finally block still runs first,
            so the partial message is persisted.
    """
    events = []
    artifacts = []
    assistant_message = ""
    status = None

    try:
        async for mode, chunk in agent.astream(  # pragma: no cover
            input={"messages": [{"role": "user", "content": user_message.content}]},
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                continue

            event = _process_chunk(chunk)

            if event is not None:
                if event.type == "tool_output":
                    for output in event.data.tool_outputs:
                        if output.artifact:
                            artifacts.append(output.artifact)

                elif event.type == "final_answer":
                    assistant_message = event.data.content
                    status = MessageStatus.SUCCESS

                events.append(event.model_dump())
                yield event.to_sse()
    except asyncio.CancelledError:
        logger.info(f"Client disconnected mid-stream for run {config['run_id']}")
        assistant_message = assistant_message or ErrorMessage.UNEXPECTED
        status = status or MessageStatus.MODEL_CALL_LIMIT
        raise
    except Exception:
        logger.exception(f"Unexpected error responding message {config['run_id']}:")
        assistant_message = ErrorMessage.UNEXPECTED
        status = MessageStatus.ERROR
        event = StreamEvent(type="error", data=EventData(content=assistant_message))
        events.append(event.model_dump())
        try:
            yield event.to_sse()
        except asyncio.CancelledError:
            logger.info(f"Client disconnected mid-stream for run {config['run_id']}")
            raise
    finally:
        message_create = MessageCreate(
            id=config["run_id"],
            thread_id=thread_id,
            user_message_id=user_message.id,
            model_uri=model_uri,
            role=MessageRole.ASSISTANT,
            content=assistant_message,
            artifacts=artifacts or None,
            events=events or None,
            status=status,
        )
        message = await database.create_message(message_create)

    yield StreamEvent(type="complete", data=EventData(run_id=message.id)).to_sse()
