from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.agent import runtime_config
from app.agent.context import AgentContext
from app.agent.middleware import system_prompt_middleware
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.schemas import StructuredResponse
from app.agent.tools import BDToolkit
from app.settings import settings


def build_agent(
    *,
    checkpointer: BaseCheckpointSaver,
    reasoning_effort: str | None = None,
) -> CompiledStateGraph:
    """Build the agent graph. Production and the eval harness both call this.

    Args:
        checkpointer: Where the graph persists thread state.
        reasoning_effort: The model's reasoning effort. The eval harness
            overrides it per run. None means `settings.REASONING_EFFORT`.

    Returns:
        The compiled agent graph.
    """
    model = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_URI,
        reasoning={
            "effort": reasoning_effort or settings.REASONING_EFFORT,
            "summary": runtime_config.REASONING_SUMMARY,
        },
    )

    summ_middleware = SummarizationMiddleware(
        model=model,
        trigger=runtime_config.SUMMARIZATION_TRIGGER,
        keep=runtime_config.SUMMARIZATION_KEEP,
        trim_tokens_to_summarize=runtime_config.SUMMARIZATION_TRIM_TOKENS,
    )

    limit_middleware = ModelCallLimitMiddleware(
        run_limit=runtime_config.MODEL_CALL_RUN_LIMIT,
        exit_behavior=runtime_config.MODEL_CALL_EXIT_BEHAVIOR,
    )

    return create_agent(
        model=model,
        tools=BDToolkit.get_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            system_prompt_middleware,
            summ_middleware,
            limit_middleware,
        ],
        response_format=StructuredResponse,
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
