import asyncio
from typing import AsyncIterator

from app.api.streaming.schemas import StreamEvent


async def sse_forwarder(
    queue: asyncio.Queue[StreamEvent],
) -> AsyncIterator[str]:
    """Forward StreamEvents from the queue as SSE strings until `complete`.

    The producer (`run_agent`) is responsible for ensuring exactly one
    `complete` event is emitted per run. This generator does no accumulation
    and no persistence: cancelling it on client disconnect is safe and has
    no side effects on the in-flight run.
    """
    while True:
        event = await queue.get()
        yield event.to_sse()
        if event.type == "complete":
            return
