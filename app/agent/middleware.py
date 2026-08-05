from datetime import date

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from app.agent.context import AgentContext
from app.i18n import language_directive


@dynamic_prompt
def system_prompt_middleware(request: ModelRequest) -> str:
    """Render the system prompt template, filling `{current_date}` and `{language_directive}`."""
    context: AgentContext = request.runtime.context

    return request.system_message.content.format(
        current_date=date.today().isoformat(),
        language_directive=language_directive(context.language),
    )
