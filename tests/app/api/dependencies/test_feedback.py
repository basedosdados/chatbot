import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from app.api.dependencies.feedback import LangSmithFeedbackSender, get_feedback_sender
from app.db.models import Feedback, FeedbackRating, FeedbackSyncStatus


class TestLangSmithFeedbackSender:
    """Tests for LangSmithFeedbackSender class."""

    @pytest.fixture
    def mock_feedback(self) -> Feedback:
        """Create a mock Feedback instance for testing."""
        return Feedback(
            message_id=uuid.uuid4(),
            rating=FeedbackRating.POSITIVE,
            comments="Mock comment",
        )

    @pytest.fixture
    def sender(self, mocker: MockerFixture) -> LangSmithFeedbackSender:
        """Create a LangSmithFeedbackSender with mocked client."""
        mock_langsmith_client = mocker.patch("app.api.dependencies.feedback.Client")
        mock_langsmith_client.return_value = MagicMock()
        return LangSmithFeedbackSender()

    def test_create_feedback_success(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test successful feedback creation on LangSmith."""
        result = sender._create_langsmith_feedback(mock_feedback)

        assert result is True

        sender.client.create_feedback.assert_called_once_with(
            run_id=mock_feedback.message_id,
            key="helpfulness",
            feedback_id=mock_feedback.id,
            score=mock_feedback.rating,
            comment=mock_feedback.comments,
        )

    def test_create_feedback_failure(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test failed feedback creation on LangSmith."""
        sender.client.create_feedback.side_effect = Exception("API Error")

        result = sender._create_langsmith_feedback(mock_feedback)

        assert result is False
        sender.client.create_feedback.assert_called_once()

    def test_update_feedback_success(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test successful feedback update on LangSmith."""
        result = sender._update_langsmith_feedback(mock_feedback)

        assert result is True

        sender.client.update_feedback.assert_called_once_with(
            feedback_id=mock_feedback.id,
            score=mock_feedback.rating,
            comment=mock_feedback.comments,
        )

    def test_update_feedback_failure(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test failed feedback update on LangSmith."""
        sender.client.update_feedback.side_effect = Exception("API Error")

        result = sender._update_langsmith_feedback(mock_feedback)

        assert result is False
        sender.client.update_feedback.assert_called_once()

    def test_send_feedback_create_success(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test send_feedback for new feedback (created=True) with success."""
        sync_status, synced_at = sender.send_feedback(mock_feedback, created=True)

        assert sync_status == FeedbackSyncStatus.SUCCESS
        assert isinstance(synced_at, datetime)
        sender.client.create_feedback.assert_called_once()

    def test_send_feedback_create_failure(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test send_feedback for new feedback (created=True) with failure."""
        sender.client.create_feedback.side_effect = Exception("API Error")

        sync_status, synced_at = sender.send_feedback(mock_feedback, created=True)

        assert sync_status == FeedbackSyncStatus.FAILED
        assert isinstance(synced_at, datetime)
        sender.client.create_feedback.assert_called_once()

    def test_send_feedback_update_success(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test send_feedback for existing feedback (created=False) with success."""
        sync_status, synced_at = sender.send_feedback(mock_feedback, created=False)

        assert sync_status == FeedbackSyncStatus.SUCCESS
        assert isinstance(synced_at, datetime)
        sender.client.update_feedback.assert_called_once()

    def test_send_feedback_update_failure(
        self, sender: LangSmithFeedbackSender, mock_feedback: Feedback
    ):
        """Test send_feedback for existing feedback (created=False) with failure."""
        sender.client.update_feedback.side_effect = Exception("API Error")

        sync_status, synced_at = sender.send_feedback(mock_feedback, created=False)

        assert sync_status == FeedbackSyncStatus.FAILED
        assert isinstance(synced_at, datetime)
        sender.client.update_feedback.assert_called_once()


class TestGetFeedbackSender:
    """Tests for get_feedback_sender dependency."""

    def test_returns_langsmith_feedback_sender(self):
        """Test that get_feedback_sender returns a LangSmithFeedbackSender instance."""
        sender = get_feedback_sender()
        assert isinstance(sender, LangSmithFeedbackSender)
