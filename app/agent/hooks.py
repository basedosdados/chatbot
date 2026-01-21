from langchain.messages import RemoveMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agent.types import StateT
from app.settings import settings


def trim_messages_before_agent(state: StateT) -> dict[str, BaseMessage]:
    messages = state["messages"]

    # For the first message, skip trimming. If it's too long, let it fail.
    if len(messages) == 1:
        return {"messages": []}

    # For subsequent turns, trim chat history to fit within token limits.
    remaining_messages = trim_messages(
        messages,
        token_counter=count_tokens_approximately,  # The accurate counter is too slow.
        max_tokens=settings.MAX_TOKENS,
        strategy="last",
        start_on="human",
        end_on="human",
        include_system=True,
        allow_partial=False,
    )

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *remaining_messages]}
