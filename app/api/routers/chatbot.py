import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import Agent, AsyncDB, FeedbackSender, UserID
from app.api.schemas import ConfigDict, UserMessage
from app.api.streaming import stream_response
from app.db.models import (FeedbackCreate, FeedbackPayload, FeedbackPublic,
                           Message, MessageCreate, MessageRole, Thread,
                           ThreadCreate, ThreadPayload)
from app.settings import settings

router = APIRouter(prefix="/chatbot")


@router.get("/threads/")
async def list_threads(
    database: AsyncDB, user_id: UserID, order_by: str | None = None
) -> list[Thread]:
    if order_by and order_by not in {"created_at", "-created_at"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'order_by' value. "
                "Valid values are 'created_at' and '-created_at'"
            )
        )

    return await database.get_threads(user_id, order_by)


@router.post("/threads/", status_code=status.HTTP_201_CREATED)
async def create_thread(
    thread_payload: ThreadPayload, database: AsyncDB, user_id: UserID,
) -> Thread:
    thread_create = ThreadCreate(
        title=thread_payload.title,
        user_id=user_id,
    )

    return await database.create_thread(thread_create)


@router.delete("/threads/{thread_id}/")
async def delete_thread_and_checkpoints(
    thread_id: str, database: AsyncDB, agent: Agent, user_id: UserID,
):
    thread = await database.delete_thread(thread_id)

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    await agent.aclear_thread(thread_id)


@router.get("/threads/{thread_id}/messages/")
async def list_messages(
    thread_id: str, database: AsyncDB, user_id: UserID, order_by: str | None = None
) -> list[Message]:
    if order_by and order_by not in {"created_at", "-created_at"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid 'order_by' value. "
                "Valid values are 'created_at' and '-created_at'"
            )
        )

    thread = await database.get_thread(thread_id)

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return await database.get_messages(thread.id, order_by)


@router.post("/threads/{thread_id}/messages/")
async def send_message(
    thread_id: str,
    user_message: UserMessage,
    agent: Agent,
    database: AsyncDB,
    user_id: UserID,
) -> Message:
    run_id = str(uuid.uuid4())

    config = ConfigDict(
        run_id=run_id,
        recursion_limit=32,
        configurable={"thread_id": thread_id},
    )

    message_create = MessageCreate(
        thread_id=thread_id,
        model_uri=settings.MODEL_URI,
        role=MessageRole.USER,
        content=user_message.content,
    )

    message = await database.create_message(message_create)

    return StreamingResponse(
        stream_response(
            database=database,
            agent=agent,
            user_message=message,
            config=config,
            thread_id=thread_id,
            model_uri=settings.MODEL_URI,
        ),
        status_code=status.HTTP_201_CREATED
    )


@router.put("/messages/{message_id}/feedbacks/", response_model=FeedbackPublic)
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
        sync_status, synced_at = await asyncio.to_thread(feedback_sender.send_feedback, feedback, created)
        _ = await database.update_feedback_sync_status(feedback.id, sync_status, synced_at)

    background_tasks.add_task(send_feedback)

    return feedback
