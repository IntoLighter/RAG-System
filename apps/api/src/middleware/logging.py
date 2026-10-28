import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils import elapsed_ms

REQUEST_ID_HEADER = "X-Request-ID"


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with a correlation id, status code, and execution time."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind a request id, time the request, and log the outcome."""
        structlog.contextvars.clear_contextvars()
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger = structlog.get_logger()
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                elapsed_ms=elapsed_ms(started_at),
            )
            raise

        duration_ms = elapsed_ms(started_at)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=duration_ms,
        )
        return response
