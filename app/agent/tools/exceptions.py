import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from pydantic import BaseModel


class ToolError(BaseModel):
    """Error response format for agents."""

    status: Literal["error"] = "error"
    message: str


def handle_tool_errors(
    func: Callable[..., Any] | None = None,
    *,
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> Callable[..., Any]:
    """Catch exceptions raised by a tool and return them as structured errors.

    Support both bare and parameterized usage:
        - @handle_tool_errors
        - @handle_tool_errors(response_format="content")
        - @handle_tool_errors(response_format="content_and_artifact")
    """

    def format_error(e: Exception) -> str | tuple[str, None]:
        tool_error = ToolError(message=str(e)).model_dump_json(indent=2)
        if response_format == "content_and_artifact":
            return tool_error, None
        return tool_error

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def async_wrapper(*args, **kwargs) -> Any:
                try:
                    return await f(*args, **kwargs)
                except Exception as e:
                    return format_error(e)

            return async_wrapper

        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                return format_error(e)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
