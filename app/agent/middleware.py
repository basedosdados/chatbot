from datetime import date

from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dynamic_prompt
def system_prompt_middleware(request: ModelRequest) -> str:
    """Render the system prompt template, filling `{current_date}`."""
    return request.system_message.content.format(current_date=date.today().isoformat())
