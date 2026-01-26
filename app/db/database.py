from datetime import datetime
from typing import TypeVar
from uuid import UUID

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel, select

from app.db.models import (
    Feedback,
    FeedbackCreate,
    FeedbackSyncStatus,
    Message,
    MessageCreate,
    Thread,
    ThreadCreate,
)
from app.settings import settings

T = TypeVar("T", bound=SQLModel)

engine = create_async_engine(settings.SQLALCHEMY_DB_URL)

sessionmaker = async_sessionmaker(engine, expire_on_commit=False)


async def init_database(engine: AsyncEngine):
    """Initialize the database by creating the schema and all tables.

    Args:
        engine: An AsyncEngine instance.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA_CHATBOT}")
        )
        await conn.run_sync(SQLModel.metadata.create_all)


class AsyncDatabase:
    """SQL database repository with async operations."""

    def __init__(self, session: AsyncSession):
        """Initialize the database repository with an async session.

        Args:
            session: An AsyncSession instance for database operations.
        """
        self.session = session
        self.logger = logger.bind(classname=self.__class__.__name__)

    @staticmethod
    def _apply_order_by(query, model: type[T], order_by: str | None):
        """Apply ordering to a query based on an order_by string.

        Args:
            query: The SQLAlchemy query to apply ordering to.
            model: The SQLModel class to get the field from.
            order_by: Field name with optional '-' prefix for descending order.

        Returns:
            The query with ordering applied.
        """
        if order_by:
            desc = order_by.startswith("-")
            field_name = order_by.lstrip("-")
            field = getattr(model, field_name)
            ordering = field.desc() if desc else field.asc()
            return query.order_by(ordering)
        return query

    # ===================================== Thread =====================================
    async def create_thread(self, thread_create: ThreadCreate) -> Thread:
        """Create a thread in the threads table.

        Args:
            thread_create (ThreadCreate): A ThreadCreate object.

        Returns:
            Thread: The created Thread object.
        """
        thread = Thread.model_validate(thread_create)

        self.session.add(thread)
        await self.session.commit()
        await self.session.refresh(thread)

        return thread

    async def get_thread(self, thread_id: str | UUID) -> Thread | None:
        """Get a thread from the threads table.

        Args:
            thread_id (str | UUID): The thread unique identifier.

        Returns:
            Thread | None: A Thread object if the thread was found. None otherwise.
        """
        query = select(Thread).where(Thread.id == thread_id, Thread.deleted.is_(False))
        results = await self.session.execute(query)
        thread = results.scalars().one_or_none()

        if thread is None:
            self.logger.warning(f"Thread {thread_id} not found")

        return thread

    async def get_threads(
        self, user_id: int, order_by: str | None = None
    ) -> list[Thread]:
        """Get all threads that belongs to a user.

        Args:
            user_id (int): The user unique identifier.
            order_by (str | None, optional): A field by which results should be ordered. Defaults to None.

        Returns:
            list[Thread]: A list of Thread objects.
        """
        query = select(Thread).where(
            Thread.user_id == user_id, Thread.deleted.is_(False)
        )
        query = self._apply_order_by(query, Thread, order_by)
        results = await self.session.execute(query)
        threads = results.scalars().all()

        return threads

    async def delete_thread(self, thread_id: str | UUID) -> Thread | None:
        """Logically delete a thread by setting its deletion flag.

        Args:
            thread_id (str | UUID): A Thread ID.

        Returns:
            Thread | None: The updated Thread object with the deletion flag set or None if not found.
        """
        query = select(Thread).where(Thread.id == thread_id, Thread.deleted.is_(False))
        results = await self.session.execute(query)
        thread = results.scalars().one_or_none()

        if thread is not None:
            thread.deleted = True
            self.session.add(thread)
            await self.session.commit()
            await self.session.refresh(thread)
        else:
            self.logger.warning(f"Thread {thread_id} not found")

        return thread

    # ================================== Message Pair ==================================
    async def create_message(self, message_create: MessageCreate) -> Message:
        """Create a message in the messages table.

        Args:
            message_create (MessageCreate): A MessageCreate object.

        Returns:
            Message: The created Message object.
        """
        message_pair = Message.model_validate(message_create)

        self.session.add(message_pair)
        await self.session.commit()
        await self.session.refresh(message_pair)

        return message_pair

    async def get_messages(
        self, thread_id: str | UUID, order_by: str | None = None
    ) -> list[Message]:
        """Get all message pairs that belongs to a thread.

        Args:
            thread_id (str | UUID): The thread unique identifier.
            order_by (str | None, optional): A field by which results should be ordered. Defaults to None.

        Returns:
            list[MessagePair]: A list of MessagePair objects.
        """
        query = select(Message).where(Message.thread_id == thread_id)
        query = self._apply_order_by(query, Message, order_by)
        results = await self.session.execute(query)
        messages = results.scalars().all()

        return messages

    # ==================================== Feedback ====================================
    async def upsert_feedback(
        self, feedback_create: FeedbackCreate
    ) -> tuple[Feedback, bool]:
        """Upsert a feedback into the feedback table, i.e., if there is already a feedback associated
        with the message pair ID, it is updated. Otherwise, it is created.

        Args:
            feedback_create (FeedbackCreate): A FeedbackCreate object
                containing the data to create or update the feedback.

        Returns:
            tuple[Feedback, bool]: The created or updated Feedback object,
                and a boolean indicating if it was created (True) or updated (False).
        """
        message_id = feedback_create.message_id

        query = select(Feedback).where(Feedback.message_id == message_id)
        results = await self.session.execute(query)
        db_feedback = results.scalars().one_or_none()

        if db_feedback is not None:
            feedback_data = feedback_create.model_dump(
                exclude_unset=True, exclude={"message_id"}
            )

            db_feedback.sqlmodel_update(feedback_data)
            db_feedback.updated_at = datetime.now()
            db_feedback.sync_status = FeedbackSyncStatus.PENDING
            self.session.add(db_feedback)
            await self.session.commit()
            await self.session.refresh(db_feedback)
            created = False
        else:
            db_feedback = Feedback.model_validate(feedback_create)
            self.session.add(db_feedback)
            await self.session.commit()
            await self.session.refresh(db_feedback)
            created = True

        return db_feedback, created

    async def update_feedback_sync_status(
        self,
        feedback_id: str | UUID,
        sync_status: FeedbackSyncStatus,
        synced_at: datetime,
    ) -> Feedback | None:
        """Update the sync_status and synced_at attributes of an existing feedback.

        Args:
            feedback_id (str | UUID): The feedback ID.
            sync_status (FeedbackSyncStatus): The synchronization status.
            synced_at (datetime): The synchronization datetime.

        Returns:
            Feedback | None: The updated Feedback object if it was found. None otherwise.
        """
        db_feedback = await self.session.get(Feedback, feedback_id)

        if db_feedback is not None:
            db_feedback.sync_status = sync_status
            db_feedback.synced_at = synced_at
            self.session.add(db_feedback)
            await self.session.commit()
            await self.session.refresh(db_feedback)
        else:
            self.logger.warning(f"Feedback {feedback_id} not found")

        return db_feedback
