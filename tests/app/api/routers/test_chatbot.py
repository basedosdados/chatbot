import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import jwt
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.dependencies import get_database, get_feedback_sender
from app.api.streaming import StreamEvent
from app.db.database import AsyncDatabase
from app.db.models import (Account, Feedback, FeedbackPublic, FeedbackRating,
                           FeedbackSyncStatus, Message, Thread)
from app.main import app
from app.settings import settings
from tests.conftest import MessagesFactory, ThreadFactory


class MockLangSmithFeedbackSender:
    def send_feedback(self, feedback: Feedback, created: bool):
        return FeedbackSyncStatus.SUCCESS, datetime.now()


class MockReActAgent:
    def invoke(self, input, config):
        return {"messages": [AIMessage("Mock response")]}

    async def ainvoke(self, input, config):
        return {"messages": [AIMessage("Mock response")]}

    def stream(self, input, config, stream_mode):
        chunk = {"agent": {"messages": [AIMessage("Mock response")]}}
        yield "updates", chunk
        yield "values", chunk

    async def astream(self, input, config, stream_mode):
        chunk = {"agent": {"messages": [AIMessage("Mock response")]}}
        yield "updates", chunk
        yield "values", chunk

    def clear_thread(self, thread_id):
        return

    async def aclear_thread(self, thread_id):
        return


@pytest.fixture
def access_token(user: Account) -> str:
    """Generate a valid JWT access token for testing."""
    payload = {"user_id": user.id}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def client(database: AsyncDatabase):
    @asynccontextmanager
    async def mock_lifespan(app: FastAPI):
        app.state.agent = MockReActAgent()
        yield

    def get_database_override():
        return database

    def get_feedback_sender_override():
        return MockLangSmithFeedbackSender()

    app.dependency_overrides[get_database] = get_database_override
    app.dependency_overrides[get_feedback_sender] = get_feedback_sender_override
    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


class TestListThreadsEndpoint:
    """Tests for GET /api/v1/chatbot/threads/"""

    def test_list_threads_success(self, client: TestClient, access_token: str, thread: Thread):
        """Test successful thread listing."""
        response = client.get(
            url="/api/v1/chatbot/threads/",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        threads = [Thread.model_validate(thread) for thread in response.json()]

        assert isinstance(threads, list)
        assert len(threads) == 1

    def test_list_threads_empty(self, client: TestClient, access_token: str):
        """Test empty thread list."""
        response = client.get(
            url="/api/v1/chatbot/threads/",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    async def test_list_threads_with_valid_order_by_asc(self, client: TestClient, access_token: str, thread_factory: ThreadFactory):
        """Test thread listing with valid order_by parameter (ascending)."""
        await thread_factory("Thread 1")
        await thread_factory("Thread 2")

        response = client.get(
            url="/api/v1/chatbot/threads/?order_by=created_at",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        threads = [Thread.model_validate(thread) for thread in response.json()]

        assert len(threads) == 2
        assert threads[0].created_at <= threads[1].created_at

    async def test_list_threads_with_valid_order_by_desc(self, client: TestClient, access_token: str, thread_factory: ThreadFactory):
        """Test thread listing with valid order_by parameter (descending)."""
        await thread_factory("Thread 1")
        await thread_factory("Thread 2")

        response = client.get(
            url="/api/v1/chatbot/threads/?order_by=-created_at",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        threads = [Thread.model_validate(thread) for thread in response.json()]

        assert len(threads) == 2
        assert threads[0].created_at >= threads[1].created_at

    def test_list_threads_with_invalid_order_by(self, client: TestClient, access_token: str):
        """Test thread listing with invalid order_by returns 400."""
        response = client.get(
            url="/api/v1/chatbot/threads/?order_by=id",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_list_threads_unauthorized(self, client: TestClient):
        """Test thread listing unauthorized returns 401."""
        response = client.get(url="/api/v1/chatbot/threads/")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateThreadEndpoint:
    """Tests for POST /api/v1/chatbot/threads/"""

    def test_create_thread_success(self, client: TestClient, access_token: str):
        """Test successful thread creation."""
        response = client.post(
            url="/api/v1/chatbot/threads/",
            json={"title": "Mock Title"},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_201_CREATED

        thread = Thread.model_validate(response.json())

        assert thread.title == "Mock Title"

    def test_create_thread_missing_title(self, client: TestClient, access_token: str):
        """Test thread creation without title field."""
        response = client.post(
            url="/api/v1/chatbot/threads/",
            json={},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_create_thread_unauthorized(self, client: TestClient):
        """Test thread creation unauthorized returns 401."""
        response = client.post(
            url="/api/v1/chatbot/threads/",
            json={"title": "Mock Title"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestDeleteThreadEndpoint:
    """Tests for DELETE /api/v1/chatbot/threads/{thread_id}/"""

    def test_delete_thread_success(self, client: TestClient, access_token: str, thread: Thread):
        """Test successful thread deletion."""
        response = client.delete(
            url=f"/api/v1/chatbot/threads/{thread.id}/",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == status.HTTP_200_OK

    def test_delete_thread_not_found(self, client: TestClient, access_token: str):
        """Test deleting non-existent thread returns 404."""
        response = client.delete(
            url=f"/api/v1/chatbot/threads/{uuid.uuid4()}/",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_thread_unauthorized(self, client: TestClient):
        """Test thread deletion unauthorized returns 401."""
        response = client.delete(url=f"/api/v1/chatbot/threads/{uuid.uuid4()}/")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestListMessagesEndpoint:
    """Tests for GET /api/v1/chatbot/threads/{thread_id}/messages/"""

    async def test_list_messages_success(self, client: TestClient, access_token: str, messages_factory: MessagesFactory):
        """Test successful message listing."""
        user_message, assistant_message = await messages_factory()

        response = client.get(
            url=f"/api/v1/chatbot/threads/{user_message.thread_id}/messages/",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        messages = [Message.model_validate(message) for message in response.json()]

        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_list_messages_empty(self, client: TestClient, access_token: str, thread: Thread):
        """Test empty message list."""
        response = client.get(
            url=f"/api/v1/chatbot/threads/{thread.id}/messages/",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_list_messages_thread_not_found(self, client: TestClient, access_token: str):
        """Test listing messages for non-existent thread returns 404."""
        response = client.get(
            url=f"/api/v1/chatbot/threads/{uuid.uuid4()}/messages/",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_list_messages_with_valid_order_by_asc(self, client: TestClient, access_token: str, messages_factory: MessagesFactory):
        """Test messages with valid order_by parameter (ascending)."""
        user_message, _ = await messages_factory()

        response = client.get(
            url=f"/api/v1/chatbot/threads/{user_message.thread_id}/messages/?order_by=created_at",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        messages = [Message.model_validate(message) for message in response.json()]

        assert len(messages) == 2
        assert messages[0].created_at <= messages[1].created_at

    async def test_list_messages_with_valid_order_by_desc(self, client: TestClient, access_token: str, messages_factory: MessagesFactory):
        """Test messages with valid order_by parameter (descending)."""
        user_message, _ = await messages_factory()

        response = client.get(
            url=f"/api/v1/chatbot/threads/{user_message.thread_id}/messages/?order_by=-created_at",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        messages = [Message.model_validate(message) for message in response.json()]

        assert len(messages) == 2
        assert messages[0].created_at >= messages[1].created_at

    def test_list_messages_with_invalid_order_by(self, client: TestClient, access_token: str, thread: Thread):
        """Test messages with invalid order_by returns 400."""
        response = client.get(
            url=f"/api/v1/chatbot/threads/{thread.id}/messages/?order_by=id",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_list_messages_unauthorized(self, client: TestClient, thread: Thread):
        """Test message listing unauthorized returns 401."""
        response = client.get(url=f"/api/v1/chatbot/threads/{thread.id}/messages/")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestSendMessageEndpoint:
    """Tests for POST /api/v1/chatbot/threads/{thread_id}/messages/"""

    def test_send_message_success(self, client: TestClient, access_token: str, thread: Thread):
        """Test successful message sending (streaming response)."""
        response = client.post(
            url=f"/api/v1/chatbot/threads/{thread.id}/messages/",
            json={"content": "Hello, chatbot!"},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_201_CREATED

        events: list[StreamEvent] = []

        for line in response.iter_lines():
            if line:
                event = StreamEvent.model_validate_json(line)
                events.append(event)

        assert len(events) >= 1
        assert events[-1].type == "complete"

    def test_send_message_missing_content(self, client: TestClient, access_token: str, thread: Thread):
        """Test sending message without content field."""
        response = client.post(
            url=f"/api/v1/chatbot/threads/{thread.id}/messages/",
            json={},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_send_message_unauthorized(self, client: TestClient, thread: Thread):
        """Test sending message unauthorized returns 401."""
        response = client.post(
            url=f"/api/v1/chatbot/threads/{thread.id}/messages/",
            json={"content": "Hello, chatbot!"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestUpsertFeedbackEndpoint:
    """Tests for PUT /api/v1/chatbot/messages/{message_id}/feedbacks/"""

    def test_create_feedback_success(self, client: TestClient, access_token: str, assistant_message: Message):
        """Test successful feedback creation."""
        rating = FeedbackRating.POSITIVE
        comments = "Great response!"

        response = client.put(
            url=f"/api/v1/chatbot/messages/{assistant_message.id}/feedbacks/",
            json={"rating": rating.value, "comments": comments},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        feedback = FeedbackPublic.model_validate(response.json())

        assert feedback.message_id == assistant_message.id
        assert feedback.rating == rating
        assert feedback.comments == comments
        assert feedback.updated_at is None

    def test_create_feedback_without_comments(self, client: TestClient, access_token: str, assistant_message: Message):
        """Test feedback creation without comments."""
        rating = FeedbackRating.NEGATIVE

        response = client.put(
            url=f"/api/v1/chatbot/messages/{assistant_message.id}/feedbacks/",
            json={"rating": rating.value},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        feedback = FeedbackPublic.model_validate(response.json())

        assert feedback.message_id == assistant_message.id
        assert feedback.rating == rating
        assert feedback.comments is None
        assert feedback.updated_at is None

    def test_update_feedback_success(self, client: TestClient, access_token: str, feedback: Feedback):
        """Test successful feedback update."""
        original_comments = feedback.comments
        new_comments = "Updated comments"

        response = client.put(
            url=f"/api/v1/chatbot/messages/{feedback.message_id}/feedbacks/",
            json={"rating": feedback.rating, "comments": new_comments},
            headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

        updated_feedback = FeedbackPublic.model_validate(response.json())

        assert updated_feedback.id == feedback.id
        assert updated_feedback.rating == feedback.rating
        assert updated_feedback.comments != original_comments
        assert updated_feedback.comments == new_comments
        assert updated_feedback.created_at == feedback.created_at
        assert updated_feedback.updated_at is not None

    def test_upsert_feedback_missing_rating(self, client: TestClient, access_token: str, assistant_message: Message):
        """Test feedback upsert missing rating field."""
        response = client.put(
            url=f"/api/v1/chatbot/messages/{assistant_message.id}/feedbacks/",
            json={"comments": "Great response!"},
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_upsert_feedback_unauthorized(self, client: TestClient, assistant_message: Message):
        """Test feedback upsert unauthorized returns 401."""
        response = client.put(
            url=f"/api/v1/chatbot/messages/{assistant_message.id}/feedbacks/",
            json={"rating": FeedbackRating.POSITIVE.value}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
