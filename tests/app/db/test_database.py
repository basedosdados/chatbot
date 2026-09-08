import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.database import AsyncDatabase, init_database
from app.db.models import (
    Feedback,
    FeedbackCreate,
    FeedbackRating,
    FeedbackSyncStatus,
    Message,
    MessageCreate,
    MessageRole,
    MessageStatus,
    QueryHandle,
    Thread,
    ThreadCreate,
)
from tests.conftest import MessagesFactory, ThreadFactory


async def test_init_database(async_engine: AsyncEngine):
    await init_database(async_engine)

    async with async_engine.connect() as conn:
        result = await conn.execute(text("SELECT tablename FROM pg_tables"))
        tables = {row[0] for row in result.all()}

    assert "thread" in tables
    assert "message" in tables
    assert "feedback" in tables
    assert "query_handles" in tables
    assert "artifact" not in tables


class TestAsyncDatabaseThread:
    """Tests for Thread CRUD operations."""

    async def test_create_thread_success(
        self, database: AsyncDatabase, thread_create: ThreadCreate
    ):
        """Test successful thread creation."""
        thread = await database.create_thread(thread_create)

        assert thread.title == thread_create.title
        assert thread.user_id == thread_create.user_id
        assert thread.deleted is False
        assert isinstance(thread.id, uuid.UUID)
        assert isinstance(thread.created_at, datetime)

    async def test_get_thread_found(self, database: AsyncDatabase, thread: Thread):
        """Test getting an existing thread."""
        thread_from_db = await database.get_thread(thread.id)

        assert thread_from_db is not None
        assert thread_from_db.id == thread.id
        assert thread_from_db.title == thread.title

    async def test_get_thread_not_found(self, database: AsyncDatabase):
        """Test getting a non-existent thread returns None."""
        thread = await database.get_thread(uuid.uuid4())

        assert thread is None

    async def test_get_threads(
        self, database: AsyncDatabase, thread_factory: ThreadFactory
    ):
        """Test getting all threads for a user."""
        thread_1 = await thread_factory("Mock Thread 1")
        thread_2 = await thread_factory("Mock Thread 2")  # noqa: F841

        threads = await database.get_threads(user_id=thread_1.user_id)

        assert isinstance(threads, list)
        assert len(threads) == 2

    async def test_get_threads_ordered_asc(
        self, database: AsyncDatabase, thread_factory: ThreadFactory
    ):
        """Test threads ordering."""
        thread_1 = await thread_factory("Mock Thread 1")
        thread_2 = await thread_factory("Mock Thread 2")  # noqa: F841

        threads = await database.get_threads(
            user_id=thread_1.user_id, order_by="created_at"
        )

        assert isinstance(threads, list)
        assert len(threads) == 2
        assert threads[0].created_at <= threads[1].created_at

    async def test_get_threads_ordered_desc(
        self, database: AsyncDatabase, thread_factory: ThreadFactory
    ):
        """Test threads ordering."""
        thread_1 = await thread_factory("Mock Thread 1")
        thread_2 = await thread_factory("Mock Thread 2")  # noqa: F841

        threads = await database.get_threads(
            user_id=thread_1.user_id, order_by="-created_at"
        )

        assert isinstance(threads, list)
        assert len(threads) == 2
        assert threads[0].created_at >= threads[1].created_at

    async def test_delete_thread_success(self, database: AsyncDatabase, thread: Thread):
        """Test (soft)deleting an existing thread."""
        thread_deleted = await database.delete_thread(thread.id)

        assert thread_deleted is not None
        assert thread_deleted.deleted is True

    async def test_delete_thread_not_found(self, database: AsyncDatabase):
        """Test deleting non-existent thread returns None."""
        thread_deleted = await database.delete_thread(uuid.uuid4())

        assert thread_deleted is None


class TestAsyncDatabaseMessage:
    """Tests for Message CRUD operations."""

    async def test_create_user_message(
        self, database: AsyncDatabase, user_message_create: MessageCreate
    ):
        """Test successful user message creation."""
        message = await database.create_message(user_message_create)

        assert message.id == user_message_create.id
        assert message.thread_id == user_message_create.thread_id
        assert (
            message.user_message_id is None
            and user_message_create.user_message_id is None
        )
        assert message.model_uri == user_message_create.model_uri
        assert message.role == user_message_create.role
        assert message.content == user_message_create.content
        assert message.events == user_message_create.events
        assert message.status == user_message_create.status
        assert isinstance(message.created_at, datetime)

    async def test_create_assistant_message(
        self, database: AsyncDatabase, assistant_message_create: MessageCreate
    ):
        """Test successful assistant message creation."""
        message = await database.create_message(assistant_message_create)

        assert message.id == assistant_message_create.id
        assert message.thread_id == assistant_message_create.thread_id
        assert message.user_message_id is not None
        assert message.user_message_id == assistant_message_create.user_message_id
        assert message.model_uri == assistant_message_create.model_uri
        assert message.role == assistant_message_create.role
        assert message.content == assistant_message_create.content
        assert message.events == assistant_message_create.events
        assert message.status == assistant_message_create.status
        assert isinstance(message.created_at, datetime)

    async def test_get_message_found(
        self, database: AsyncDatabase, user_message: Message
    ):
        """Test getting an existing message."""
        message = await database.get_message(user_message.id)

        assert message is not None
        assert message.id == user_message.id
        assert message.thread_id == user_message.thread_id

    async def test_get_message_not_found(self, database: AsyncDatabase):
        """Test getting a non-existent message returns None."""
        message = await database.get_message(uuid.uuid4())

        assert message is None

    async def test_get_messages(
        self, database: AsyncDatabase, messages_factory: MessagesFactory
    ):
        """Test getting all messages for a thread."""
        user_message, assistant_message = await messages_factory()

        messages = await database.get_messages(user_message.thread_id)

        assert isinstance(messages, list)
        assert len(messages) == 2

    async def test_get_messages_ordered_asc(
        self, database: AsyncDatabase, messages_factory: MessagesFactory
    ):
        """Test getting all messages for a thread."""
        user_message, assistant_message = await messages_factory()

        messages = await database.get_messages(
            user_message.thread_id, order_by="created_at"
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0].created_at <= messages[1].created_at

    async def test_get_messages_ordered_desc(
        self, database: AsyncDatabase, messages_factory: MessagesFactory
    ):
        """Test getting all messages for a thread."""
        user_message, assistant_message = await messages_factory()

        messages = await database.get_messages(
            user_message.thread_id, order_by="-created_at"
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0].created_at >= messages[1].created_at

    async def test_get_messages_not_found(
        self, database: AsyncDatabase, thread: Thread
    ):
        """Test getting all messages for a thread without messages."""
        messages = await database.get_messages(thread.id)
        assert isinstance(messages, list)
        assert len(messages) == 0

    async def test_get_messages_excludes_streaming_placeholders(
        self, database: AsyncDatabase, messages_factory: MessagesFactory
    ):
        """In-flight STREAMING placeholders are hidden from the thread listing."""
        user_message, _ = await messages_factory()

        placeholder = MessageCreate(
            thread_id=user_message.thread_id,
            user_message_id=user_message.id,
            model_uri="mock-model",
            role=MessageRole.ASSISTANT,
            content="",
            status=MessageStatus.STREAMING,
        )
        await database.create_message(placeholder)

        messages = await database.get_messages(user_message.thread_id)

        # Only the two completed messages surface; the placeholder is excluded.
        assert len(messages) == 2
        assert all(message.status != MessageStatus.STREAMING for message in messages)

    async def test_update_message_writes_terminal_fields(
        self, database: AsyncDatabase, user_message: Message
    ):
        """update_message overwrites the terminal fields of an existing row."""
        placeholder = MessageCreate(
            id=uuid.uuid4(),
            thread_id=user_message.thread_id,
            user_message_id=user_message.id,
            model_uri="mock-model",
            role=MessageRole.ASSISTANT,
            content="",
            status=MessageStatus.STREAMING,
        )
        created = await database.create_message(placeholder)

        updated = await database.update_message(
            created.id,
            content="Final answer",
            events=[{"type": "final_answer"}],
            structured_response={"response": "Final answer"},
            status=MessageStatus.SUCCESS,
        )

        assert updated is not None
        assert updated.id == created.id
        assert updated.content == "Final answer"
        assert updated.status == MessageStatus.SUCCESS
        assert updated.events == [{"type": "final_answer"}]
        assert updated.structured_response == {"response": "Final answer"}

    async def test_update_message_not_found_returns_none(self, database: AsyncDatabase):
        """Updating a non-existent message returns None (finalize then falls back)."""
        result = await database.update_message(
            uuid.uuid4(),
            content="x",
            events=None,
            structured_response=None,
            status=MessageStatus.SUCCESS,
        )

        assert result is None


class TestAsyncDatabaseQueryHandle:
    """Tests for QueryHandle operations."""

    DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}

    def _handle(
        self, message: Message, query_ref: str, destination: dict
    ) -> QueryHandle:
        return QueryHandle(
            query_ref=query_ref,
            message_id=message.id,
            slug="resultado",
            destination_table=destination,
        )

    async def test_create_query_handles_persists(
        self, database: AsyncDatabase, assistant_message: Message
    ):
        """create_query_handles bulk-inserts every handle."""
        await database.create_query_handles(
            [
                self._handle(assistant_message, "qr_a", self.DESTINATION),
                self._handle(assistant_message, "qr_b", self.DESTINATION),
            ]
        )

        first = await database.get_query_handle_from_message(
            "qr_a", assistant_message.id
        )
        second = await database.get_query_handle_from_message(
            "qr_b", assistant_message.id
        )

        assert first is not None and first.destination_table == self.DESTINATION
        assert second is not None and second.message_id == assistant_message.id

    async def test_create_query_handles_empty_is_noop(self, database: AsyncDatabase):
        """An empty list persists nothing and does not error."""
        await database.create_query_handles([])

    async def test_get_query_handle_from_message_found(
        self, database: AsyncDatabase, query_handle: QueryHandle
    ):
        """An existing handle is returned by its (query_ref, message_id)."""
        found = await database.get_query_handle_from_message(
            query_handle.query_ref, query_handle.message_id
        )

        assert found is not None
        assert found.query_ref == query_handle.query_ref

    async def test_get_query_handle_from_message_not_found(
        self, database: AsyncDatabase
    ):
        """A missing (query_ref, message_id) returns None."""
        assert (
            await database.get_query_handle_from_message("qr_missing", uuid.uuid4())
            is None
        )

    async def test_get_query_handle_from_thread_found(
        self,
        database: AsyncDatabase,
        query_handle: QueryHandle,
        assistant_message: Message,
    ):
        """A handle resolves by query_ref alone when scoped to its own thread."""
        found = await database.get_query_handle_from_thread(
            query_handle.query_ref, assistant_message.thread_id
        )

        assert found is not None
        assert found.query_ref == query_handle.query_ref

    async def test_get_query_handle_from_thread_wrong_thread_returns_none(
        self, database: AsyncDatabase, query_handle: QueryHandle
    ):
        """A handle is invisible to a thread it does not belong to (authorization)."""
        found = await database.get_query_handle_from_thread(
            query_handle.query_ref, uuid.uuid4()
        )

        assert found is None

    async def test_get_query_handles_by_thread(
        self,
        database: AsyncDatabase,
        query_handle: QueryHandle,
        assistant_message: Message,
    ):
        """Every handle produced in a thread is listed."""
        handles = await database.get_query_handles_by_thread(
            assistant_message.thread_id
        )

        assert [handle.query_ref for handle in handles] == [query_handle.query_ref]

    async def test_get_query_handles_by_thread_empty(
        self, database: AsyncDatabase, thread: Thread
    ):
        """A thread with no query results returns an empty list."""
        assert await database.get_query_handles_by_thread(thread.id) == []


class TestAsyncDatabaseFeedback:
    """Tests for Feedback CRUD operations."""

    async def test_upsert_feedback_create(
        self, database: AsyncDatabase, feedback_create: FeedbackCreate
    ):
        """Test creating new feedback."""
        feedback, created = await database.upsert_feedback(feedback_create)

        assert feedback.message_id == feedback_create.message_id
        assert feedback.rating == feedback_create.rating
        assert feedback.comments == feedback_create.comments
        assert created is True
        assert isinstance(feedback.id, uuid.UUID)
        assert isinstance(feedback.created_at, datetime)

    async def test_upsert_feedback_update(
        self, database: AsyncDatabase, feedback: Feedback
    ):
        """Test updating existing feedback."""
        feedback_create = FeedbackCreate(
            message_id=feedback.message_id,
            rating=FeedbackRating.NEGATIVE,
            comments="Changed my mind",
        )

        updated_feedback, created = await database.upsert_feedback(feedback_create)

        assert updated_feedback.id == feedback.id
        assert updated_feedback.message_id == feedback.message_id
        assert updated_feedback.rating == feedback_create.rating
        assert updated_feedback.comments == feedback_create.comments
        assert updated_feedback.sync_status == FeedbackSyncStatus.PENDING
        assert created is False
        assert isinstance(updated_feedback.updated_at, datetime)

    async def test_update_feedback_sync_status_success(
        self, database: AsyncDatabase, feedback: Feedback
    ):
        """Test updating feedback sync status."""
        synced_at = datetime.now(timezone.utc)

        feedback_synced = await database.update_feedback_sync_status(
            feedback_id=feedback.id,
            sync_status=FeedbackSyncStatus.SUCCESS,
            synced_at=synced_at,
        )

        assert feedback_synced.id == feedback.id
        assert feedback_synced.sync_status == FeedbackSyncStatus.SUCCESS
        assert feedback_synced.synced_at == synced_at

    async def test_update_feedback_sync_status_not_found(self, database: AsyncDatabase):
        """Test updating non-existent feedback returns None."""
        feedback_synced = await database.update_feedback_sync_status(
            feedback_id=uuid.uuid4(),
            sync_status=FeedbackSyncStatus.SUCCESS,
            synced_at=datetime.now(timezone.utc),
        )

        assert feedback_synced is None
