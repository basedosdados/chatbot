import asyncio
import uuid
from typing import Annotated

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph


def get_agent(request: Request) -> CompiledStateGraph:
    return request.app.state.agent


def get_running_runs(request: Request) -> dict[uuid.UUID, asyncio.Task]:
    return request.app.state.running_runs


Agent = Annotated[CompiledStateGraph, Depends(get_agent)]
RunningRuns = Annotated[dict[uuid.UUID, asyncio.Task], Depends(get_running_runs)]
