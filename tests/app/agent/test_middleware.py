from datetime import date

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.agent.context import AgentContext
from app.agent.middleware import system_prompt_middleware
from app.i18n import LanguageCode, language_directive

# A template mirroring the real prompt's placeholders (see app.agent.prompts.SYSTEM_PROMPT).
PROMPT_TEMPLATE = "Today is {current_date}.\n{language_directive}"


class _SpyModel(GenericFakeChatModel):
    """Fake model that records the system message it was asked to answer with."""

    system_seen: str = ""

    def _generate(self, messages: list[BaseMessage], *args, **kwargs):
        type(self).system_seen = messages[0].content
        return super()._generate(messages, *args, **kwargs)


def _system_prompt_for(language: LanguageCode) -> str:
    model = _SpyModel(messages=iter([AIMessage(content="ok")]))
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=PROMPT_TEMPLATE,
        middleware=[system_prompt_middleware],
        context_schema=AgentContext,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "oi"}]},
        context=AgentContext(thread_id="t", user_id="u", language=language),
    )
    return _SpyModel.system_seen


class TestSystemPromptMiddleware:
    @pytest.mark.parametrize("language", ["pt", "en", "es"])
    def test_fills_language_directive_placeholder(self, language: str):
        system = _system_prompt_for(language)

        assert language_directive(language) in system
        assert "{language_directive}" not in system

    def test_fills_current_date_placeholder(self):
        system = _system_prompt_for("pt")

        assert date.today().isoformat() in system
        assert "{current_date}" not in system

    def test_user_message_is_left_untouched(self):
        """The directive rides on the system prompt, never the user's message."""
        model = _SpyModel(messages=iter([AIMessage(content="ok")]))
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=PROMPT_TEMPLATE,
            middleware=[system_prompt_middleware],
            context_schema=AgentContext,
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "oi"}]},
            context=AgentContext(thread_id="t", user_id="u", language="en"),
        )

        assert result["messages"][0].content == "oi"
