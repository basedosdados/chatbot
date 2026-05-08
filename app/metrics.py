import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "handler", "status"],
)

request_latency = Histogram(
    "http_request_duration_seconds",
    "Total request duration (TTLB)",
    ["method", "handler"],
    buckets=[
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
    ],
)

request_ttfb = Histogram(
    "http_request_ttfb_seconds",
    "Time to first byte (TTFB)",
    ["method", "handler"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


class MetricsMiddleware:
    """ASGI middleware for Prometheus metrics collection with TTFB support.

    Attributes:
        app (ASGIApp): A startlette ASGI app.
        excluded_paths (set[str]): A set of HTTP paths to be excluded.

    Reference: https://starlette.dev/middleware/#pure-asgi-middleware
    """

    def __init__(self, app: ASGIApp, excluded_paths: set[str] | None = None) -> None:
        self.app = app
        self.excluded_paths = excluded_paths or set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["path"] in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()

        # Default 500: an unhandled exception escapes past us before
        # http.response.start is sent, so _send never overwrites this.
        # Starlette's ServerErrorMiddleware (outside us) then returns 500 on the
        # wire, so the recorded status matches what the client received.
        status_code = 500

        # None is a sentinel for "no http.response.start was ever sent" (e.g. the
        # app crashed before responding, or the client disconnected first). TTFB
        # is undefined in that case, so we skip observing the histogram below.
        ttfb = None

        # Track whether the client hung up and whether the inner app finished
        # writing the response body. A client disconnect that happens before the
        # final body chunk is the streaming-abort case; we label those as "499"
        # instead of "2xx" so they aren't confused with successful responses.
        client_disconnected = False
        response_complete = False

        async def _receive() -> Message:
            nonlocal client_disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                client_disconnected = True
            return message

        async def _send(message: Message) -> None:
            nonlocal status_code, ttfb, response_complete
            if message["type"] == "http.response.start":
                status_code = message["status"]
                ttfb = time.perf_counter() - start_time
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_complete = True
            elif message["type"] == "http.response.pathsend":
                response_complete = True
            await send(message)

        try:
            await self.app(scope, _receive, _send)
        finally:
            latency = time.perf_counter() - start_time

            if client_disconnected and not response_complete:
                # Client closed the connection before the server finished sending the response.
                # Intentionally log as 499 to distinguish this edge case from other 4xx errors.
                status = "499"
            else:
                status = f"{status_code // 100}xx"

            method = scope["method"]

            # The router mutates scope["route"] when it matches a path. This
            # finally block runs after the inner app returns (or raises), so
            # routing has already happened and the key is populated. Falls
            # back to "__unmatched__" for 404s where no route matched.
            route: Route | None = scope.get("route")
            handler = route.path if route else "__unmatched__"

            request_count.labels(method=method, handler=handler, status=status).inc()
            request_latency.labels(method=method, handler=handler).observe(latency)

            if ttfb is not None:
                request_ttfb.labels(method=method, handler=handler).observe(ttfb)


async def metrics_endpoint(request: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
