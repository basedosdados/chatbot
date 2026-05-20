import asyncio
import uuid

from app.api.streaming.schemas import EventData, StreamEvent
from app.api.streaming.stream import stream_events


class TestStreamEvents:
    async def test_stream_until_complete_then_exits(self):
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        run_id = str(uuid.uuid4())
        await queue.put(StreamEvent(type="final_answer", data=EventData(content="ok")))
        await queue.put(StreamEvent(type="complete", data=EventData(run_id=run_id)))

        events = []
        async for sse in stream_events(queue):
            events.append(sse)

        assert len(events) == 2
        assert '"type":"final_answer"' in events[0]
        assert '"type":"complete"' in events[1]
        assert run_id in events[1]

    async def test_stream_error_event_before_complete(self):
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        await queue.put(StreamEvent(type="error", data=EventData(content="bad")))
        await queue.put(
            StreamEvent(type="complete", data=EventData(run_id=str(uuid.uuid4())))
        )

        events = []
        async for sse in stream_events(queue):
            events.append(sse)

        assert '"type":"error"' in events[0]
        assert '"type":"complete"' in events[1]
