from typing import Annotated

from fastapi import Depends, Request

from app.agent import ReActAgent


def get_agent(request: Request) -> ReActAgent:
    return request.app.state.agent


Agent = Annotated[ReActAgent, Depends(get_agent)]
