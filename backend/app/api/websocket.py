"""WebSocket endpoint for streaming structured interview + AVD + DDP diagnosis."""

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

_DONE = object()
avd_sessions: dict[str, AVDSession] = {}


async def diagnose_ws(websocket: WebSocket, session_id: str):
    """Stream structured interview → AVD → DDP diagnosis over WebSocket."""
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

        # ── Check for API key in query params ───────────────────────
        api_key = websocket.query_params.get("api_key", "")

        # ── Select VLM backend ──────────────────────────────────────
        if api_key:
            from app.config import Settings
            from app.model.inference import VLMInference
            cfg = Settings(api_key=api_key)
            vlm = VLMInference.from_config(cfg)
        else:
            vlm = websocket.app.state.vlm
            if vlm is None:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Model not loaded"}}
                )
                await websocket.close(code=4503)
                return

        # ── Phase 1: Structured Interview ───────────────────────────
        diag_ctx = session["context"]
        collected_context: list[str] = []
        if diag_ctx.user_context:
            collected_context.append(diag_ctx.user_context)

        questions = [
            "请问作物种植在哪个地区？（例如：陕西杨凌）",
            "目前是什么季节/月份？",
            "近期天气情况如何？（例如：近期多雨、高温干旱等）",
            "请描述您观察到的具体症状。",
        ]

        for idx, question in enumerate(questions):
            if websocket.client_state != WebSocketState.CONNECTED:
                return
            await websocket.send_json({
                "type": "avd_question",
                "data": {
                    "question": question,
                    "reason": "",
                    "target_part": "",
                    "step": idx + 1,
                    "total": len(questions),
                },
            })
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=120
                )
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                return

            action = msg.get("action") if isinstance(msg, dict) else None
            if action == "continue":
                answer = str(msg.get("answer", "")).strip()
                if answer:
                    collected_context.append(answer)

        diag_ctx.user_context = "；".join(collected_context)

        # ── Phase 2: AVD image assessment ───────────────────────────
        if websocket.client_state != WebSocketState.CONNECTED:
            return

        await websocket.send_json({
            "type": "avd_sufficient",
            "data": {
                "summary": "问诊完成，正在分析图片...",
                "confidence": 0.0,
                "forced": False,
            },
        })

        avd_session = _build_avd_session(session_id, diag_ctx)
        avd_sessions[session_id] = avd_session
        avd_engine = AVDEngine(vlm)

        assessment = await avd_engine.assess(avd_session)
        await _send_avd_assessment(websocket, assessment)

        while assessment.status == AVDStatus.QUESTIONING:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                return

            action = msg.get("action") if isinstance(msg, dict) else None
            if action == "continue":
                _sync_avd_session(avd_session, diag_ctx)
                assessment = await avd_engine.assess(avd_session)
                await _send_avd_assessment(websocket, assessment)
            elif action == "skip":
                assessment = AVDAssessment(
                    status=AVDStatus.FORCED,
                    confidence=assessment.confidence,
                    question=None,
                    summary="用户跳过补充拍照，直接进入诊断。",
                )
                await _send_avd_assessment(websocket, assessment)
                break

        # ── Phase 3: DDP Debate ─────────────────────────────────────
        if websocket.client_state != WebSocketState.CONNECTED:
            return

        knowledge_retriever = None
        case_retriever = None
        try:
            from app.rag.retriever import KnowledgeRetriever
            knowledge_retriever = KnowledgeRetriever()
        except Exception:
            pass
        try:
            from app.retrieval.similar_cases import SimilarCaseRetriever
            case_retriever = SimilarCaseRetriever()
        except Exception:
            pass

        orchestrator = DDPOrchestrator(
            vlm,
            case_retriever=case_retriever,
            knowledge_retriever=knowledge_retriever,
        )

        queue: asyncio.Queue = asyncio.Queue()

        def on_message(msg: AgentMessage) -> None:
            queue.put_nowait(msg)

        async def _sender():
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

        context = avd_session.to_diagnosis_context()

        sender_task = asyncio.create_task(_sender())
        result = await orchestrator.run_debate(context, on_message=on_message)
        queue.put_nowait(_DONE)
        await sender_task

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
        avd_sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_avd_session(session_id: str, diag_ctx) -> AVDSession:
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
    existing = len(avd_session.images)
    for i in range(existing, len(diag_ctx.images)):
        img = _decode_b64_image(diag_ctx.images[i])
        desc = diag_ctx.image_descriptions[i] if i < len(diag_ctx.image_descriptions) else ""
        avd_session.add_image(img, desc)


def _decode_b64_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


async def _send_avd_assessment(websocket: WebSocket, assessment: AVDAssessment) -> None:
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
        await websocket.send_json({
            "type": "avd_sufficient",
            "data": {
                "summary": assessment.summary,
                "confidence": assessment.confidence,
                "forced": assessment.status == AVDStatus.FORCED,
            },
        })
