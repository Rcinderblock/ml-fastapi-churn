"""Запуск: uvicorn main:app --reload."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from core.config import Settings
from core.errors import install_error_handlers
from core.service import ChurnService
from schemas.churn import ErrorResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
        app.state.service = ChurnService(settings)
        app.state.service.startup()
        yield

    app = FastAPI(
        title="ML Churn Service",
        version="1.0.0",
        description="Предсказание оттока клиентов по девяти табличным признакам.",
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    install_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
