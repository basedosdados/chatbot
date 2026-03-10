from typing import Annotated

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph


def get_agent(request: Request) -> CompiledStateGraph:
    return request.app.state.agent


Agent = Annotated[CompiledStateGraph, Depends(get_agent)]
