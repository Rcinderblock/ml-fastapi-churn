"""Единый формат ошибок HTTP и внутренних операций сервиса."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


def validation_details(errors: list[dict]) -> list[dict[str, Any]]:
    # Не включаем исходный запрос и объекты исключений в JSON ответа.
    return [
        {"field": list(error["loc"]), "message": error["msg"], "type": error["type"]}
        for error in errors[:20]
    ]


def install_error_handlers(app: FastAPI):
    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        logger.warning("%s %s: %s", request.method, request.url.path, exc.code)
        return JSONResponse(
            status_code=exc.status,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        logger.warning("Неверный запрос: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "Проверьте поля и типы значений запроса",
                "details": validation_details(exc.errors()),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        logger.warning("Ошибка HTTP %d: %s", exc.status_code, request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": "HTTP_ERROR", "message": str(exc.detail), "details": None},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Внутренняя ошибка: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "Не удалось выполнить операцию",
                "details": None,
            },
        )
