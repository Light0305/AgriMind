"""WebSocket endpoint for streaming AVD + DDP diagnosis results."""

from __future__ import annotations

import asyncio
import base64
import io
import logging

from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from starlette.websockets import WebSocketState

from app.agents.ddp import DDPOrchestrator
from app.api.routes import sessions
from app.avd.engine import AVDEngine
from app.avd.session import AVDSession
from app.schemas import AVDAssessment, AVDStatus, AgentMessage

logger = logging.getLogger(__name__)

# Sentinel that tells the sender coroutine the debate is done.
_DONE = object()

# In-memory store for active AVD sessions (keyed by session_id).
avd_sessions: dict[str, AVDSession] = {}


async def diagnose_ws(websocket: WebSocket, session_id: str):
    """Stream an AVD → DDP diagnosis over WebSocket.

    Protocol
    --------
    1. Client connects to ``/ws/diagnose/{session_id}``.
    2. Server runs AVD assessment on uploaded image(s).
       - If **questioning**: sends ``{"type": "avd_question", ...}`` and
         waits for the client to send ``{"action": "continue"}`` after
         uploading a new image via the REST endpoint.
       - If **sufficient**: sends ``{"type": "avd_sufficient", ...}`` and
         proceeds to DDP.
       - If **forced**: same as sufficient (with a warning).
    3. DDP debate is streamed as before:
       ``{"type": "status", "data": {"status": "debating"}}``
       ``{"type": "agent_message", "data": <AgentMessage>}`` × 4
       ``{"type": "result",  "data": <DebateResult>}``
    4. Connection is closed.

    Errors are sent as ``{"type": "error", "data": {"message": "..."}}``.
    """
    await websocket.accept()

    try:
        # ── Validate session ────────────────────────────────────────
        session = sessions.get(session_id)
        if session is None:
            await websocket.send_json(
                {"type": "error", "data": {"message": "Session not found"}}
            )
            await websocket.close(code=4004)
            return

        # ── Read optional API config from first message ─────────────
        try:
            first_msg = await asyncio.wait_for(websocket.receive_json(), timeout=2)
        except asyncio.TimeoutError:
            first_msg = {}
        except WebSocketDisconnect:
            return

        api_key = ""
        if isinstance(first_msg, dict) and first_msg.get("type") == "config":
            api_key = str(first_msg.get("api_key", "")).strip()

        # ── Use API or local VLM ────────────────────────────────────
        if api_key:
            from app.config import Settings
            from app.model.inference import VLMInference

            cfg = Settings(api_key=api_key)
            vlm = VLMInference.from_config(cfg)
            logger.info("Using API mode (key=***%s)", api_key[-4:])
        else:
            vlm = websocket.app.state.vlm  # type: ignore[attr-defined]
            if vlm is None:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Model not loaded yet"}}
                )
                await websocket.close(code=4503)
                return
            logger.info("Using local model")

        # ── Structured Interview (replaces photo-based AVD) ────────
        diag_ctx = session["context"]
        collected_context: list[str] = []
        if diag_ctx.user_context:
            collected_context.append(diag_ctx.user_context)

        # Structured questions: location → time → weather → symptoms
        questions = [
            ("location", "请问作物种植在哪个地区？（例如：陕西杨凌）"),
            ("season", "目前是什么季节/月份？"),
            ("weather", "近期天气情况如何？（例如：近期多雨、高温干旱等）"),
            ("symptoms", "请描述您观察到的具体症状。"),
        ]

        for qid, question in questions:
            await websocket.send_json({
                "type": "avd_question",
                "data": {
                    "question": question,
                    "reason": "",
                    "target_part": qid,
                    "step": questions.index((qid, question)) + 1,
                    "total": len(questions),
                },
            })
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=120)
            except asyncio.TimeoutError:
                msg = {"action": "skip"}
            except WebSocketDisconnect:
                return

            action = msg.get("action") if isinstance(msg, dict) else None
            if action == "continue":
                answer = str(msg.get("answer", "")).strip()
                if answer:
                    collected_context.append(f"{qid}: {answer}")
            elif action == "skip":
                continue  # skip this question, proceed to next

        # Build enriched context from interview answers
        diag_ctx.user_context = "；".join(collected_context)

        # ── AVD done — proceed to DDP debate (v2 with retrievers) ────
        # Lazy-init retrievers so the debate benefits from RAG & case memory
        try:
            from app.rag.retriever import KnowledgeRetriever
            knowledge_retriever = KnowledgeRetriever()
            knowledge_retriever.collection  # force lazy init
        except Exception:
            logger.warning("KnowledgeRetriever unavailable, proceeding without RAG")
            knowledge_retriever = None

        try:
            from app.retrieval.similar_cases import SimilarCaseRetriever
            case_retriever = SimilarCaseRetriever()
            case_retriever.collection  # force lazy init
        except Exception:
            logger.warning("SimilarCaseRetriever unavailable, proceeding without cases")
            case_retriever = None

        orchestrator = DDPOrchestrator(
            vlm,
            case_retriever=case_retriever,
            knowledge_retriever=knowledge_retriever,
        )

        queue: asyncio.Queue = asyncio.Queue()

        def on_message(msg: AgentMessage) -> None:
            queue.put_nowait(msg)

        async def _sender():
            """Drain the queue and push agent messages over WS."""
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(
                        {"type": "agent_message", "data": item.model_dump()}
                    )

        session["status"] = "debating"
        await websocket.send_json(
            {"type": "status", "data": {"status": "debating"}}
        )

        # Use the AVD session's context (which has all collected images).
        context = avd_session.to_diagnosis_context()

        sender_task = asyncio.create_task(_sender())
        result = await orchestrator.run_debate(context, on_message=on_message)
        queue.put_nowait(_DONE)
        await sender_task

        # ── Store result and send to client ─────────────────────────
        session["result"] = result
        session["status"] = "completed"

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {"type": "result", "data": result.model_dump()}
            )
            await websocket.close()

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    except Exception as exc:
        logger.exception("Error in diagnose_ws for session %s", session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {"type": "error", "data": {"message": str(exc)}}
            )
            await websocket.close(code=4500)
    finally:
        # Clean up AVD session.
        avd_sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_avd_session(session_id: str, diag_ctx) -> AVDSession:
    """Create an :class:`AVDSession` from a :class:`DiagnosisContext`."""
    avd_session = AVDSession(
        session_id=session_id,
        user_context=diag_ctx.user_context,
    )
    for i, b64 in enumerate(diag_ctx.images):
        img = _decode_b64_image(b64)
        desc = diag_ctx.image_descriptions[i] if i < len(diag_ctx.image_descriptions) else ""
        avd_session.add_image(img, desc)
    return avd_session


def _sync_avd_session(avd_session: AVDSession, diag_ctx) -> None:
    """Sync new images added via REST into the AVD session."""
    existing = len(avd_session.images)
    for i in range(existing, len(diag_ctx.images)):
        img = _decode_b64_image(diag_ctx.images[i])
        desc = diag_ctx.image_descriptions[i] if i < len(diag_ctx.image_descriptions) else ""
        avd_session.add_image(img, desc)


def _decode_b64_image(b64: str) -> Image.Image:
    """Decode a single base64 string into a PIL Image."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


async def _send_avd_assessment(websocket: WebSocket, assessment: AVDAssessment) -> None:
    """Send the appropriate AVD WebSocket message for an assessment."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return

    if assessment.status == AVDStatus.QUESTIONING and assessment.question:
        await websocket.send_json({
            "type": "avd_question",
            "data": {
                "question": assessment.question.question,
                "reason": assessment.question.reason,
                "target_part": assessment.question.target_part,
                "confidence": assessment.confidence,
                "summary": assessment.summary,
            },
        })
    else:
        # SUFFICIENT or FORCED
        await websocket.send_json({
            "type": "avd_sufficient",
            "data": {
                "summary": assessment.summary,
                "confidence": assessment.confidence,
                "forced": assessment.status == AVDStatus.FORCED,
            },
        })
