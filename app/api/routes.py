"""HTTP adapters call the same service as Gradio."""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.schemas import ApproveRequest, ChatRequest, ChatResponse, HealthResponse
from app.service import AgentService, ServiceError
from core.config import DATA_PATH, TOOLS_JSON_PATH
from core.memory import get_memory_manager

logger = logging.getLogger(__name__)


def create_app(
    *,
    service: AgentService | None = None,
    memory_factory=get_memory_manager,
    mount_ui: bool = True,
    initialize_memory: bool = True,
) -> FastAPI:
    service = service or AgentService()

    def initialize():
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY를 .env에 설정해주세요.")
        memory = memory_factory()
        if Path(TOOLS_JSON_PATH).is_file():
            logger.info("Loaded %s catalog tools", memory.load_tools_from_json(TOOLS_JSON_PATH))
        else:
            logger.warning("Catalog is missing: %s", TOOLS_JSON_PATH)
        logger.info("Loaded %s PDF chunks", memory.load_pdfs_from_directory(DATA_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if initialize_memory:
            await run_in_threadpool(initialize)
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="AI 101",
        description="LangGraph 기반 AI 도구 추천 에이전트",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.agent_service = service

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError):
        if error.status_code >= 500:
            logger.error("Agent request failed", exc_info=error)
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    # Sync endpoints run in FastAPI's worker thread pool, keeping the event loop free
    # while LangGraph, embedding, and model clients perform blocking work.
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        memory = memory_factory()
        return HealthResponse(
            status="healthy",
            tools_count=memory.get_tools_count(),
            profiles_count=memory.get_profiles_count(),
        )

    @app.post("/chat/start", response_model=ChatResponse)
    def start_chat(request: ChatRequest):
        return ChatResponse(
            **asdict(service.start(request.query, request.user_id, thread_id=request.thread_id))
        )

    @app.post("/chat/approve", response_model=ChatResponse)
    def approve_plan(request: ApproveRequest):
        return ChatResponse(
            **asdict(
                service.resume(request.thread_id, request.user_id, request.action, request.feedback)
            )
        )

    if mount_ui:
        import gradio as gr

        from app.ui.gradio_app import create_gradio_ui

        app = gr.mount_gradio_app(app, create_gradio_ui(service, memory_factory), path="/")
    return app
