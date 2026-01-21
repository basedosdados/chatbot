from datetime import datetime
from typing import Annotated

from fastapi import Depends
from langsmith import Client
from loguru import logger

from app.db.models import Feedback, FeedbackSyncStatus


class LangSmithFeedbackSender:
    """A wrapper to send feedbacks to LangSmith."""

    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.client = Client(
            api_url=api_url,
            api_key=api_key,
        )

        self.logger = logger.bind(classname=self.__class__.__name__)

    def _create_langsmith_feedback(self, feedback: Feedback) -> bool:
        """Create feedback on LangSmith.

        Args:
            feedback (Feedback): The feedback instance to create.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            _ = self.client.create_feedback(
                run_id=feedback.message_id,
                key="helpfulness",
                feedback_id=feedback.id,
                score=feedback.rating,
                comment=feedback.comments,
            )
            self.logger.info(
                f"Successfully created feedback {feedback.id} "
                f"for run {feedback.message_id} on LangSmith"
            )
            return True
        except Exception:
            self.logger.exception(
                f"Failed to create feedback {feedback.id} "
                f"for run {feedback.message_id} on LangSmith:"
            )
            return False

    def _update_langsmith_feedback(self, feedback: Feedback) -> bool:
        """Update existing feedback on LangSmith.

        Args:
            feedback (Feedback): The feedback instance to update.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.client.update_feedback(
                feedback_id=feedback.id,
                score=feedback.rating,
                comment=feedback.comments,
            )
            self.logger.info(
                f"Successfully updated feedback {feedback.id} "
                f"for run {feedback.message_id} on LangSmith"
            )
            return True
        except Exception:
            self.logger.exception(
                f"Failed to update feedback {feedback.id} "
                f"for run {feedback.message_id} on LangSmith:"
            )
            return False

    def send_feedback(
        self, feedback: Feedback, created: bool
    ) -> tuple[FeedbackSyncStatus, datetime]:
        """Create or update a feedback on LangSmith.

        Args:
            feedback (Feedback): The feedback instance to send.
            created (bool): True if this is a new feedback, False if it's an update.
        """
        if created:
            success = self._create_langsmith_feedback(feedback)
        else:
            success = self._update_langsmith_feedback(feedback)

        sync_status = (
            FeedbackSyncStatus.SUCCESS if success else FeedbackSyncStatus.FAILED
        )
        synced_at = datetime.now()

        return sync_status, synced_at


def get_feedback_sender() -> LangSmithFeedbackSender:
    return LangSmithFeedbackSender()


FeedbackSender = Annotated[LangSmithFeedbackSender, Depends(get_feedback_sender)]
