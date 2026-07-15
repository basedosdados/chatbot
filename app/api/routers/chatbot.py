import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.dependencies import Agent, AsyncDB, FeedbackSender, RunningRuns, UserID
from app.api.schemas import ConfigDict, UserMessage
from app.api.streaming import run_agent, stream_events
from app.api.streaming.schemas import StreamEvent
from app.db.models import (
    FeedbackCreate,
    FeedbackPayload,
    FeedbackPublic,
    MessageCreate,
    MessagePublic,
    MessageRole,
    Thread,
    ThreadCreate,
    ThreadPayload,
)
from app.exports import (
    ExportFormat,
    ResultTableExpired,
    ResultTooLarge,
    derive_downloads,
    materialize_export,
)
from app.settings import settings
from app.storage import generate_signed_url

router = APIRouter(prefix="/chatbot")


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
    )

    return await database.create_thread(thread_create)


@router.delete("/threads/{thread_id}")
async def delete_thread_and_checkpoints(
    thread_id: str,
    database: AsyncDB,
    agent: Agent,
    user_id: UserID,
):
    thread = await database.delete_thread(thread_id)

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    if agent.checkpointer is not None:
        await agent.checkpointer.adelete_thread(thread_id)


@router.get("/threads/{thread_id}/messages")
async def list_messages(
    thread_id: str, database: AsyncDB, user_id: UserID, order_by: str | None = None
) -> list[MessagePublic]:
    if order_by and order_by not in {"created_at", "-created_at"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'order_by' value. "
                "Valid values are 'created_at' and '-created_at'"
            ),
        )

    thread = await database.get_thread(thread_id)

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

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
    run_id = str(uuid.uuid4())

    config = ConfigDict(
        run_id=run_id,
        configurable={"thread_id": thread_id, "user_id": user_id},
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


# User-facing details the frontend surfaces to the end user when a download fails.
RESULTS_EXPIRED_DETAIL = "Estes resultados não estão mais disponíveis para download."

RESULTS_TOO_LARGE_DETAIL = (
    "Estes resultados são grandes demais para baixar em um único arquivo."
)

# Base name for the downloaded file (materialization appends the extension).
DEFAULT_EXPORT_FILENAME = "resultados"


def _download_filename(query_ref: str, query_refs: list[str]) -> str:
    """Build the base name for a download (the extension is appended on materialization).

    Suffixed with the query's 1-based position only when the answer backs more than one
    query, so a lone result stays `resultados`.

    Args:
        query_ref (str): The `query_ref` being downloaded.
        query_refs (list[str]): The answer's downloadable `query_ref`s, in order.

    Returns:
        str: The base filename, without extension.
    """
    if len(query_refs) <= 1:
        return DEFAULT_EXPORT_FILENAME
    return f"{DEFAULT_EXPORT_FILENAME}_{query_refs.index(query_ref) + 1}"


@router.post("/messages/{message_id}/exports")
async def export_message_results(
    message_id: str,
    database: AsyncDB,
    user_id: UserID,
    query_ref: str,
    file_format: ExportFormat = Query("CSV", alias="format"),
):
    message = await database.get_message(message_id)

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found",
        )

    thread = await database.get_thread(message.thread_id)

    # Identical 404 whether the message is missing or the caller doesn't own its
    # thread, so a non-owner can't tell the two apart (IDOR protection).
    if thread is None or str(thread.user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found",
        )

    query_refs = [
        item["query_ref"] for item in derive_downloads(message.structured_response)
    ]

    if query_ref not in query_refs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No downloadable results for query_ref '{query_ref}'",
        )

    query_handle = await database.get_query_handle(message.id, query_ref)

    if query_handle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No downloadable results for query_ref '{query_ref}'",
        )

    try:
        exported = await asyncio.to_thread(
            materialize_export,
            query_ref=query_ref,
            destination_table=query_handle.destination_table,
            file_format=file_format,
            filename=_download_filename(query_ref, query_refs),
            message_id=str(message.id),
        )
    except ResultTableExpired as e:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=RESULTS_EXPIRED_DETAIL,
        ) from e
    except ResultTooLarge as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=RESULTS_TOO_LARGE_DETAIL,
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
