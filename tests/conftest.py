from collections.abc import AsyncGenerator, Awaitable, Callable

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from testcontainers.postgres import PostgresContainer

from app.db.database import AsyncDatabase
from app.db.models import (
    Feedback,
    FeedbackCreate,
    FeedbackRating,
    Message,
    MessageCreate,
    MessageRole,
    MessageStatus,
    Thread,
    ThreadCreate,
)

type ThreadFactory = Callable[[str], Awaitable[Thread]]
type MessagesFactory = Callable[[], Awaitable[tuple[Message, Message]]]


# =============================================================
# Database Fixtures
# =============================================================
@pytest.fixture(scope="session")
def postgres_container():
    """Create a test PostgreSQL container."""
    with PostgresContainer(image="postgres:13-alpine", driver="psycopg") as postgres:
        yield postgres


@pytest_asyncio.fixture(loop_scope="session")
async def async_engine(
    postgres_container: PostgresContainer,
) -> AsyncGenerator[AsyncEngine, None]:
    """Create an async engine connected to the test PostgreSQL container."""
    postgres_url = postgres_container.get_connection_url()
    engine = create_async_engine(postgres_url, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def database(async_engine: AsyncEngine):
    """Create a AsyncDatabase instance connected to the test PostgreSQL container."""
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    sessionmaker = async_sessionmaker(async_engine, expire_on_commit=False)

    async with sessionmaker() as session:
        yield AsyncDatabase(session)

    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


# =============================================================
# User Fixtures
# =============================================================
@pytest_asyncio.fixture
async def user_id() -> int:
    """Mock user ID for testing."""
    return 1


# =============================================================
# Thread Fixtures
# =============================================================
@pytest.fixture
def thread_create(user_id: int) -> ThreadCreate:
    """Mock ThreadCreate instance for testing."""
    return ThreadCreate(title="Mock Thread", user_id=user_id)


@pytest_asyncio.fixture
async def thread(database: AsyncDatabase, thread_create: ThreadCreate) -> Thread:
    """Mock Thread instance for testing."""
    return await database.create_thread(thread_create)


@pytest_asyncio.fixture
async def thread_factory(database: AsyncDatabase, user_id: int) -> ThreadFactory:
    """Factory to create multiple threads in a single test."""

    async def factory(title: str) -> Thread:
        thread_create = ThreadCreate(title=title, user_id=user_id)
        thread = await database.create_thread(thread_create)
        return thread

    return factory


# =============================================================
# Message Fixtures
# =============================================================
@pytest.fixture
def user_message_create(thread: Thread) -> MessageCreate:
    """Mock MessageCreate instance for testing (user)."""
    return MessageCreate(
        thread_id=thread.id,
        model_uri="mock-model",
        role=MessageRole.USER,
        content="Mock user message",
        status=MessageStatus.SUCCESS,
    )


@pytest_asyncio.fixture
async def user_message(
    database: AsyncDatabase, user_message_create: MessageCreate
) -> Message:
    """Mock Message instance for testing (user)."""
    return await database.create_message(user_message_create)


@pytest.fixture
def assistant_message_create(user_message: Message) -> MessageCreate:
    """Mock MessageCreate instance for testing (assistant)."""
    return MessageCreate(
        thread_id=user_message.thread_id,
        user_message_id=user_message.id,
        model_uri="mock_model",
        role=MessageRole.ASSISTANT,
        content="Mock assistant message",
        artifacts=[{"mock_artifact": "artifact"}],
        events=[{"mock_event": "event"}],
        status=MessageStatus.SUCCESS,
    )


@pytest_asyncio.fixture
async def assistant_message(
    database: AsyncDatabase, assistant_message_create: MessageCreate
) -> Message:
    """Mock Message instance for testing (assistant)."""
    return await database.create_message(assistant_message_create)


@pytest_asyncio.fixture
async def messages_factory(database: AsyncDatabase, thread: Thread) -> MessagesFactory:
    """Factory to create a user/assistant message pair in a single test."""

    async def factory() -> tuple[Message, Message]:
        user_message_create = MessageCreate(
            thread_id=thread.id,
            model_uri="mock-model",
            role=MessageRole.USER,
            content="Mock user message",
            status=MessageStatus.SUCCESS,
        )

        user_message = await database.create_message(user_message_create)

        assistant_message_create = MessageCreate(
            thread_id=user_message.thread_id,
            user_message_id=user_message.id,
            model_uri="mock_model",
            role=MessageRole.ASSISTANT,
            content="Mock assistant message",
            artifacts=[{"mock_artifact": "artifact"}],
            events=[{"mock_event": "event"}],
            status=MessageStatus.SUCCESS,
        )

        assistant_message = await database.create_message(assistant_message_create)

        return user_message, assistant_message

    return factory


# =============================================================
# Feedback Fixtures
# =============================================================
@pytest.fixture
def feedback_create(assistant_message: Message) -> FeedbackCreate:
    return FeedbackCreate(
        message_id=assistant_message.id,
        rating=FeedbackRating.POSITIVE,
        comments="Mock comments",
    )


@pytest_asyncio.fixture
async def feedback(
    database: AsyncDatabase, feedback_create: FeedbackCreate
) -> Feedback:
    feedback, _ = await database.upsert_feedback(feedback_create)
    return feedback
