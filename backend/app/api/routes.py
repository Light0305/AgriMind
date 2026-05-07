"""REST API routes for diagnosis sessions."""

from __future__ import annotations

import base64
import logging
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas import DiagnosisContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory session store — sufficient for a competition demo.
sessions: dict[str, dict[str, Any]] = {}


@router.post("/diagnose")
async def create_diagnosis(
    request: Request,
    files: list[UploadFile] = File(...),
    context: str | None = Form(default=None),
):
    """Create a new diagnosis session.

    Accepts one or more image files plus an optional text description.
    Returns a ``session_id`` that the client uses to open a WebSocket
    for the streaming DDP debate.
    """
    if not files:
        raise HTTPException(status_code=400, detail="至少需要上传一张图片")

    # Encode every uploaded image as base64
    images_b64: list[str] = []
    for f in files:
        raw = await f.read()
        images_b64.append(base64.b64encode(raw).decode())

    session_id = uuid.uuid4().hex
    diag_ctx = DiagnosisContext(images=images_b64, user_context=context)

    sessions[session_id] = {
        "context": diag_ctx,
        "status": "created",
        "result": None,
    }

    logger.info("Session %s created with %d image(s)", session_id, len(images_b64))
    return {"session_id": session_id, "status": "created"}


@router.get("/diagnose/{session_id}")
async def get_diagnosis(session_id: str):
    """Return session status and, if complete, the debate result."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    payload: dict[str, Any] = {"session_id": session_id, "status": session["status"]}
    if session["result"] is not None:
        payload["result"] = session["result"].model_dump()
    return payload


@router.post("/diagnose/{session_id}/upload")
async def add_image(session_id: str, file: UploadFile = File(...)):
    """Append an additional image to an existing session.

    Useful for the AVD agent's follow-up questions that request extra
    photos of specific plant parts.
    """
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    raw = await file.read()
    b64 = base64.b64encode(raw).decode()
    session["context"].images.append(b64)

    return {
        "session_id": session_id,
        "image_count": len(session["context"].images),
    }
