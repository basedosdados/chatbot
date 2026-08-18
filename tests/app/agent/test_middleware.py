from datetime import date

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.middleware import system_prompt_middleware
from app.agent.prompts import SYSTEM_PROMPT


def _run_middleware(system_prompt: str, messages: list) -> ModelRequest:
    """Run system_prompt_middleware over a request and return the request the model
    would have been called with (the middleware overrides its `system_message`)."""
    request = ModelRequest(
        model=None,
        messages=messages,
        system_message=SystemMessage(content=system_prompt),
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=None,
        model_settings={},
    )

    captured: dict = {}

    def handler(seen: ModelRequest) -> AIMessage:
        captured["request"] = seen
        return AIMessage(content="ok")

    system_prompt_middleware.wrap_model_call(request, handler)

    return captured["request"]


class TestSystemPromptMiddleware:
    def test_real_system_prompt_renders_without_stray_placeholders(self):
        """Render the actual SYSTEM_PROMPT: the date must land and no brace may survive."""
        request = _run_middleware(SYSTEM_PROMPT, [HumanMessage(content="oi")])
        content = request.system_message.content

        assert date.today().isoformat() in content
        assert "{" not in content and "}" not in content
