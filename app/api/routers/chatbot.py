import asyncio
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agent.context import AgentContext
from app.api.dependencies import Agent, AsyncDB, FeedbackSender, RunningRuns, UserID
from app.api.schemas import ConfigDict, UserMessage
from app.api.streaming import run_agent, stream_events
from app.api.streaming.schemas import StreamEvent
from app.db.database import AsyncDatabase
from app.db.models import (
    FeedbackCreate,
    FeedbackPayload,
    FeedbackPublic,
    Message,
    MessageCreate,
    MessagePublic,
    MessageRole,
    Thread,
    ThreadCreate,
    ThreadPayload,
)
from app.exports import (
    OFFERED_EXPORT_FORMATS,
    ExportFormat,
    ResultTableExpired,
    ResultTooLarge,
    materialize_export,
)
from app.i18n import MessageKey, translate
from app.settings import settings
from app.storage import generate_signed_url

router = APIRouter(prefix="/chatbot")


async def _authorize_thread(
    database: AsyncDatabase, thread_id: str, user_id: str
) -> Thread:
    """Fetch a thread and verify the caller owns it.

    Args:
        database (AsyncDatabase): The database repository.
        thread_id (str): The thread the caller is trying to reach.
        user_id (str): The authenticated caller.

    Returns:
        Thread: The thread, guaranteed to belong to `user_id`.

    Raises:
        HTTPException: 404 whether the thread is missing or owned by someone else.
    """
    thread = await database.get_thread(thread_id)

    if thread is None or str(thread.user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return thread


async def _authorize_message(
    database: AsyncDatabase, message_id: str, user_id: str
) -> tuple[Message, Thread]:
    """Fetch a message and its thread, verifying the caller owns the thread.

    Args:
        database (AsyncDatabase): The database repository.
        message_id (str): The message the caller is trying to reach.
        user_id (str): The authenticated caller.

    Returns:
        tuple[Message, Thread]: The message and its owning thread,
            guaranteed to belong to `user_id`.

    Raises:
        HTTPException: 404 whether the message is missing or the caller doesn't own its thread.
    """
    message = await database.get_message(message_id)
    thread = (
        await database.get_thread(message.thread_id) if message is not None else None
    )

    if message is None or thread is None or str(thread.user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found",
        )

    return message, thread


@router.get("/threads")
async def list_threads(
    database: AsyncDB, user_id: UserID, order_by: str | None = None
) -> list[Thread]:
    if order_by and order_by not in {"created_at", "-created_at"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'order_by' value. "
                "Valid values are 'created_at' and '-created_at'"
            ),
        )

    return await database.get_threads(user_id, order_by)


@router.post("/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(
    thread_payload: ThreadPayload,
    database: AsyncDB,
    user_id: UserID,
) -> Thread:
    thread_create = ThreadCreate(
        title=thread_payload.title,
        user_id=user_id,
        language=thread_payload.language,
    )

    return await database.create_thread(thread_create)


@router.delete("/threads/{thread_id}")
async def delete_thread_and_checkpoints(
    thread_id: str,
    database: AsyncDB,
    agent: Agent,
    user_id: UserID,
):
    await _authorize_thread(database, thread_id, user_id)

    await database.delete_thread(thread_id)

    if agent.checkpointer is not None:
        await agent.checkpointer.adelete_thread(thread_id)


@router.get("/threads/{thread_id}/messages", response_model=list[MessagePublic])
async def list_messages(
    thread_id: str, database: AsyncDB, user_id: UserID, order_by: str | None = None
):
    if order_by and order_by not in {"created_at", "-created_at"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'order_by' value. "
                "Valid values are 'created_at' and '-created_at'"
            ),
        )

    thread = await _authorize_thread(database, thread_id, user_id)

    return await database.get_messages(thread.id, order_by)


@router.post(
    "/threads/{thread_id}/messages",
    response_class=StreamingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    thread_id: str,
    user_message: UserMessage,
    database: AsyncDB,
    agent: Agent,
    running_runs: RunningRuns,
    user_id: UserID,
) -> StreamingResponse:
    thread = await _authorize_thread(database, thread_id, user_id)

    run_id = str(uuid.uuid4())

    # thread_id stays in `configurable` too because the
    # langgraph checkpointer keys persistence on it;
    config = ConfigDict(
        run_id=run_id,
        configurable={"thread_id": thread_id},
    )

    # application data rides on the context.
    context = AgentContext(
        thread_id=thread_id,
        user_id=user_id,
        language=thread.language,
    )

    message_create = MessageCreate(
        thread_id=thread_id,
        model_uri=settings.MODEL_URI,
        role=MessageRole.USER,
        content=user_message.content,
    )

    message = await database.create_message(message_create)

    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

    task = asyncio.create_task(
        run_agent(
            agent=agent,
            config=config,
            context=context,
            thread_id=thread_id,
            user_message=message,
            model_uri=settings.MODEL_URI,
            queue=queue,
        ),
        name=f"run_agent:{run_id}",
    )

    running_runs[run_id] = task

    def _cleanup(task: asyncio.Task):  # pragma: no cover
        del running_runs[run_id]
        if task.cancelled():
            logger.warning(f"run_agent task {run_id} was cancelled mid-run")
            return
        e = task.exception()
        if e is not None:
            logger.opt(exception=e).error(f"run_agent task {run_id} crashed mid-run:")

    task.add_done_callback(_cleanup)

    return StreamingResponse(
        stream_events(queue),
        status_code=status.HTTP_201_CREATED,
    )


def _sanitize_filename(slug: str, fallback: str) -> str:
    """Sanitize a query's slug into a safe base filename.

    Args:
        slug (str): The query's slug.
        fallback (str): Base name to use when the slug yields nothing filesystem-safe.

    Returns:
        str: A filesystem-safe base filename, without extension.
    """
    filename = re.sub(r"[^\w-]+", "_", slug).strip("_")
    return filename or fallback


@router.post("/messages/{message_id}/exports")
async def export_message_results(
    message_id: str,
    database: AsyncDB,
    user_id: UserID,
    query_ref: str,
    file_format: ExportFormat = Query("CSV", alias="format"),
):
    if file_format not in OFFERED_EXPORT_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported format '{file_format}'. "
                f"Available: {', '.join(OFFERED_EXPORT_FORMATS)}."
            ),
        )

    message, thread = await _authorize_message(database, message_id, user_id)

    query_handle = await database.get_query_handle(message.id, query_ref)

    if query_handle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No downloadable results for query_ref '{query_ref}'",
        )

    try:
        exported = await asyncio.to_thread(
            materialize_export,
            query_ref=query_handle.query_ref,
            destination_table=query_handle.destination_table,
            file_format=file_format,
            filename=_sanitize_filename(
                query_handle.slug,
                translate(MessageKey.DEFAULT_EXPORT_FILENAME, thread.language),
            ),
            message_id=str(message.id),
        )
    except ResultTableExpired as e:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=translate(MessageKey.RESULTS_EXPIRED, thread.language),
        ) from e
    except ResultTooLarge as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=translate(MessageKey.RESULTS_TOO_LARGE, thread.language),
        ) from e

    signed_url = generate_signed_url(
        bucket=exported.bucket,
        object_key=exported.object_key,
        download_filename=exported.filename,
    )

    return {"url": signed_url}


@router.put("/messages/{message_id}/feedback", response_model=FeedbackPublic)
async def upsert_feedback(
    message_id: str,
    feedback_payload: FeedbackPayload,
    background_tasks: BackgroundTasks,
    database: AsyncDB,
    feedback_sender: FeedbackSender,
    user_id: UserID,
):
    await _authorize_message(database, message_id, user_id)

    feedback_create = FeedbackCreate(
        **feedback_payload.model_dump(exclude_unset=True),
        message_id=message_id,
    )

    feedback, created = await database.upsert_feedback(feedback_create)

    async def send_feedback():
        # LangSmith's AsyncClient doesn't support the update_feedback method, so we use the sync Client instead.
        # Since it blocks the event loop, we run it in a separate thread to avoid blocking async execution.
        sync_status, synced_at = await asyncio.to_thread(
            feedback_sender.send_feedback, feedback, created
        )
        _ = await database.update_feedback_sync_status(
            feedback.id, sync_status, synced_at
        )

    background_tasks.add_task(send_feedback)

    return feedback
