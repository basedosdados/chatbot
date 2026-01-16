from unittest.mock import MagicMock

from app.api.dependencies.agent import get_agent


class TestGetAgent:
    """Tests for get_agent dependency."""

    def test_returns_agent_from_app_state(self):
        """Test that get_agent returns the agent from request.app.state.agent"""
        mock_agent = MagicMock()
        mock_request = MagicMock()

        mock_request.app.state.agent = mock_agent

        result = get_agent(mock_request)

        assert result is mock_agent
