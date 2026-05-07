"""WebSocket endpoint for streaming DDP debate results."""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.agents.ddp import DDPOrchestrator
from app.api.routes import sessions
from app.schemas import AgentMessage

logger = logging.getLogger(__name__)

# Sentinel that tells the sender coroutine the debate is done.
_DONE = object()


async def diagnose_ws(websocket: WebSocket, session_id: str):
    """Stream a DDP debate over WebSocket.

    Protocol
    --------
    1. Client connects to ``/ws/diagnose/{session_id}``.
    2. Server sends ``{"type": "status", "data": {"status": "debating"}}``.
    3. Each agent turn is pushed as
       ``{"type": "agent_message", "data": <AgentMessage>}``.
    4. The final ruling is sent as
       ``{"type": "result",  "data": <DebateResult>}``.
    5. The server closes the connection.

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

        # ── Ensure model is loaded (lazy init via app.state) ────────
        vlm = websocket.app.state.vlm  # type: ignore[attr-defined]
        if vlm is None:
            await websocket.send_json(
                {"type": "error", "data": {"message": "Model not loaded yet"}}
            )
            await websocket.close(code=4503)
            return

        # ── Build orchestrator ──────────────────────────────────────
        orchestrator = DDPOrchestrator(vlm)

        # ── Queue-based streaming ───────────────────────────────────
        # DDPOrchestrator.on_message is a *sync* callback invoked
        # between awaited agent turns.  We bridge sync → async via
        # an asyncio.Queue: the sync callback puts, a sender task gets.
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

        # ── Run debate ──────────────────────────────────────────────
        session["status"] = "debating"
        await websocket.send_json(
            {"type": "status", "data": {"status": "debating"}}
        )

        context = session["context"]

        # Run the debate and the sender concurrently.
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
