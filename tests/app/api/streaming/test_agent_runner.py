import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.api.schemas import ConfigDict
from app.api.streaming.agent_runner import ErrorMessage, run_agent
from app.api.streaming.schemas import StreamEvent
from app.db.models import Message, MessageRole, MessageStatus


@pytest.fixture
def thread_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def model_uri() -> str:
    return "mock-model"


@pytest.fixture
def config(thread_id: str) -> ConfigDict:
    return {"run_id": uuid.uuid4(), "configurable": {"thread_id": thread_id}}


@pytest.fixture
def user_message(thread_id: str, model_uri: str) -> Message:
    return Message(
        thread_id=thread_id,
        model_uri=model_uri,
        role=MessageRole.USER,
        content="Hello",
        status=MessageStatus.SUCCESS,
    )


@pytest.fixture
def database(config: ConfigDict, user_message: Message):
    db = MagicMock()
    db.create_message = AsyncMock(
        return_value=Message(
            id=config["run_id"],
            thread_id=user_message.thread_id,
            user_message_id=user_message.id,
            model_uri=user_message.model_uri,
            role=MessageRole.ASSISTANT,
            content="ok",
            status=MessageStatus.SUCCESS,
        )
    )
    return db


async def _drain(queue: asyncio.Queue) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    while True:
        event = await queue.get()
        events.append(event)
        if event.type == "complete":
            return events


class TestRunAgentHappyPath:
    async def test_emits_events_and_persists_success(
        self, database, user_message, config, thread_id, model_uri
    ):
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="Here is your answer.")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            database=database,
            agent=agent,
            user_message=user_message,
            config=config,
            thread_id=thread_id,
            model_uri=model_uri,
            queue=queue,
        )

        events = await _drain(queue)
        assert [e.type for e in events] == ["final_answer", "complete"]
        assert events[-1].data.run_id == config["run_id"]

        database.create_message.assert_called_once()
        persisted = database.create_message.call_args[0][0]
        assert persisted.status == MessageStatus.SUCCESS
        assert persisted.content == "Here is your answer."


class TestRunAgentErrorPaths:
    async def test_unexpected_exception_persists_error_row(
        self, database, user_message, config, thread_id, model_uri
    ):
        agent = MagicMock()

        async def astream(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # make this a generator

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            database=database,
            agent=agent,
            user_message=user_message,
            config=config,
            thread_id=thread_id,
            model_uri=model_uri,
            queue=queue,
        )

        events = await _drain(queue)
        assert [e.type for e in events] == ["error", "complete"]
        assert events[0].data.content == ErrorMessage.UNEXPECTED

        persisted = database.create_message.call_args[0][0]
        assert persisted.status == MessageStatus.ERROR
        assert persisted.content == ErrorMessage.UNEXPECTED

    async def test_model_call_limit_persists_with_dedicated_status(
        self, database, user_message, config, thread_id, model_uri
    ):
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "ModelCallLimitMiddleware.before_model": {"jump_to": "end"},
                },
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            database=database,
            agent=agent,
            user_message=user_message,
            config=config,
            thread_id=thread_id,
            model_uri=model_uri,
            queue=queue,
        )

        events = await _drain(queue)
        assert events[0].type == "final_answer"
        assert events[0].data.content == ErrorMessage.MODEL_CALL_LIMIT_REACHED

        persisted = database.create_message.call_args[0][0]
        assert persisted.status == MessageStatus.MODEL_CALL_LIMIT


class TestRunAgentSurvivesConsumerCancellation:
    async def test_consumer_cancel_does_not_cancel_producer(
        self, database, user_message, config, thread_id, model_uri
    ):
        """The producer task must run to completion even if no one drains the queue."""
        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="answer")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        task = asyncio.create_task(
            run_agent(
                database=database,
                agent=agent,
                user_message=user_message,
                config=config,
                thread_id=thread_id,
                model_uri=model_uri,
                queue=queue,
            )
        )

        # Simulate the consumer never attaching: just await the producer.
        await asyncio.wait_for(task, timeout=2.0)

        # Producer wrote the row regardless of consumer presence.
        database.create_message.assert_called_once()

        # And the complete event is sitting in the queue waiting.
        events = await _drain(queue)
        assert events[-1].type == "complete"


class TestRunAgentPersistenceFailure:
    async def test_complete_still_emitted_when_db_write_fails(
        self, user_message, config, thread_id, model_uri
    ):
        """If `database.create_message` raises, the consumer must still receive
        a `complete` event — otherwise it hangs on `queue.get()`."""
        database = MagicMock()
        database.create_message = AsyncMock(side_effect=RuntimeError("db down"))

        agent = MagicMock()

        async def astream(*args, **kwargs):
            yield (
                "updates",
                {"model": {"messages": [AIMessage(content="answer")]}},
            )

        agent.astream = astream
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        await run_agent(
            database=database,
            agent=agent,
            user_message=user_message,
            config=config,
            thread_id=thread_id,
            model_uri=model_uri,
            queue=queue,
        )

        events = await _drain(queue)
        complete = events[-1]
        assert complete.type == "complete"
        # No run_id — the message was never persisted, so there is nothing
        # for the client to correlate against.
        assert complete.data.run_id is None
        # error_details signals the failure mode to the client.
        assert complete.data.error_details is not None
        assert complete.data.error_details["reason"] == "persistence_failed"
