import json

from google.api_core.exceptions import BadRequest

from app.agent.tools.exceptions import handle_tool_errors


class TestHandleToolErrors:
    """Tests for handle_tool_errors decorator."""

    def test_decorator_passes_through_success(self):
        """Test decorator returns function result on success."""

        @handle_tool_errors
        def successful_function():
            return '{"key": "value"}'

        output = successful_function()
        assert json.loads(output) == {"key": "value"}

    def test_decorator_catches_exception(self):
        """Test decorator catches exceptions and returns ToolError JSON."""

        @handle_tool_errors
        def failing_function():
            raise ValueError("something went wrong")

        output = json.loads(failing_function())

        assert output["status"] == "error"
        assert output["message"] == "something went wrong"

    def test_decorator_catches_google_api_error(self):
        """Test decorator catches GoogleAPICallError."""

        @handle_tool_errors
        def failing_function():
            raise BadRequest(
                message="Some bad request",
                errors=[{"reason": "testReason", "message": "Test message"}],
            )

        output = json.loads(failing_function())

        assert output["status"] == "error"
        assert output["message"] == "400 Some bad request"
