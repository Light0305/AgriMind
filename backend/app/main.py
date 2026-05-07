"""FastAPI application entry point for AgriMind."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.websocket import diagnose_ws
from app.config import settings
from app.model.inference import VLMInference

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.

    The VLM model itself is **lazy-loaded** inside ``VLMInference``, so
    startup is fast — the heavy weight load happens on the first actual
    inference request.
    """
    logger.info("AgriMind starting up — VLM will lazy-load on first request")
    app.state.vlm = VLMInference.from_config(settings)
    yield
    logger.info("AgriMind shutting down")
    app.state.vlm = None


app = FastAPI(
    title="AgriMind API",
    description="作物智能会诊系统后端",
    version="0.1.0",
    lifespan=lifespan,
)

# -- CORS (Vite dev server on :5173, plus common dev ports) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- REST routes -----------------------------------------------------------
app.include_router(router)

# -- WebSocket route -------------------------------------------------------
app.add_api_websocket_route("/ws/diagnose/{session_id}", diagnose_ws)
