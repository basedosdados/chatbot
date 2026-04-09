import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from pydantic import BaseModel


class ToolError(BaseModel):
    "Error response format for agents."

    status: Literal["error"] = "error"
    message: str


def handle_tool_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that catches exceptions raised by a tool and returns them as structured errors.

    Args:
        func (Callable[..., Any]): Function to wrap.

    Returns:
        Callable[..., Any]: Wrapped function.
    """

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return ToolError(message=str(e)).model_dump_json(indent=2)

        return async_wrapper

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return ToolError(message=str(e)).model_dump_json(indent=2)

    return wrapper
