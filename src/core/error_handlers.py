from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = structlog.get_logger(__name__)


def _err(
    status: int,
    code: str,
    msg: str,
    detail: str | None = None,
    retryable: bool = False,
) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": msg, "retryable": retryable}}
    if detail:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def install_exception_handlers(app: FastAPI) -> None:
    from .errors import ShielvaException

    @app.exception_handler(ShielvaException)
    async def _shielva(req, exc: ShielvaException) -> JSONResponse:
        logger.warning(
            "shielva_exception",
            error_code=exc.error_code,
            message=exc.message,
            path=req.url.path,
        )
        return _err(exc.status_code, exc.error_code, exc.message, exc.detail, exc.retryable)

    @app.exception_handler(StarletteHTTPException)
    async def _http(req, exc: StarletteHTTPException) -> JSONResponse:
        logger.info("http_exception", status_code=exc.status_code, path=req.url.path)
        return _err(exc.status_code, "HTTP_ERROR", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(req, exc: RequestValidationError) -> JSONResponse:
        logger.info("validation_error", path=req.url.path)
        return _err(422, "VALIDATION_ERROR", "Request validation failed", str(exc.errors()))

    @app.exception_handler(Exception)
    async def _generic(req, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", exc_type=type(exc).__name__, path=req.url.path)
        return _err(500, "INTERNAL_ERROR", "An internal error occurred.")
