"""Tests for the FastAPI REST + WebSocket API layer."""

from __future__ import annotations

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api.routes import sessions
from app.main import app
from app.schemas import (
    AgentMessage,
    AgentRole,
    Confidence,
    DebateResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_png_bytes() -> bytes:
    """Return raw bytes of a 4×4 red PNG."""
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny_b64() -> str:
    return base64.b64encode(_make_tiny_png_bytes()).decode()


DUMMY_RESULT = DebateResult(
    final_diagnosis="小麦白粉病",
    confidence=Confidence.HIGH,
    supporting_evidence=["白色粉状菌层"],
    rejected_diagnoses=[],
    uncertainty_notes=[],
    debate_transcript=[
        AgentMessage(role=AgentRole.PROPOSER, content="初诊", round=1),
        AgentMessage(role=AgentRole.CHALLENGER, content="质疑", round=1),
        AgentMessage(role=AgentRole.PROPOSER, content="回应", round=2),
        AgentMessage(role=AgentRole.ARBITER, content="裁定", round=2),
    ],
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def _clear_sessions():
    """Ensure session store is empty before each test."""
    sessions.clear()
    yield
    sessions.clear()


@pytest_asyncio.fixture()
async def client():
    """httpx async client wired to the FastAPI app."""
    # Patch app.state.vlm so routes don't require a real model
    app.state.vlm = MagicMock()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.state.vlm = None


# ---------------------------------------------------------------------------
# POST /api/diagnose
# ---------------------------------------------------------------------------


class TestCreateDiagnosis:
    @pytest.mark.asyncio
    async def test_create_session(self, client: AsyncClient):
        png = _make_tiny_png_bytes()
        resp = await client.post(
            "/api/diagnose",
            files=[("files", ("leaf.png", png, "image/png"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "created"
        # Session should exist in the store
        assert data["session_id"] in sessions

    @pytest.mark.asyncio
    async def test_create_with_context(self, client: AsyncClient):
        png = _make_tiny_png_bytes()
        resp = await client.post(
            "/api/diagnose",
            files=[("files", ("leaf.png", png, "image/png"))],
            data={"context": "叶片发黄"},
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        assert sessions[sid]["context"].user_context == "叶片发黄"

    @pytest.mark.asyncio
    async def test_create_multiple_images(self, client: AsyncClient):
        png = _make_tiny_png_bytes()
        resp = await client.post(
            "/api/diagnose",
            files=[
                ("files", ("leaf1.png", png, "image/png")),
                ("files", ("leaf2.png", png, "image/png")),
            ],
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        assert len(sessions[sid]["context"].images) == 2


# ---------------------------------------------------------------------------
# GET /api/diagnose/{session_id}
# ---------------------------------------------------------------------------


class TestGetDiagnosis:
    @pytest.mark.asyncio
    async def test_get_existing_session(self, client: AsyncClient):
        # Create first
        png = _make_tiny_png_bytes()
        create_resp = await client.post(
            "/api/diagnose",
            files=[("files", ("leaf.png", png, "image/png"))],
        )
        sid = create_resp.json()["session_id"]

        resp = await client.get(f"/api/diagnose/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, client: AsyncClient):
        resp = await client.get("/api/diagnose/nonexistent_id_12345")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_completed_session_includes_result(self, client: AsyncClient):
        # Manually inject a completed session
        sid = "test_completed"
        sessions[sid] = {
            "context": None,
            "status": "completed",
            "result": DUMMY_RESULT,
        }

        resp = await client.get(f"/api/diagnose/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["result"]["final_diagnosis"] == "小麦白粉病"


# ---------------------------------------------------------------------------
# POST /api/diagnose/{session_id}/upload
# ---------------------------------------------------------------------------


class TestAddImage:
    @pytest.mark.asyncio
    async def test_add_image_to_session(self, client: AsyncClient):
        # Create session
        png = _make_tiny_png_bytes()
        create_resp = await client.post(
            "/api/diagnose",
            files=[("files", ("leaf.png", png, "image/png"))],
        )
        sid = create_resp.json()["session_id"]
        assert len(sessions[sid]["context"].images) == 1

        # Add another image
        resp = await client.post(
            f"/api/diagnose/{sid}/upload",
            files=[("file", ("leaf2.png", png, "image/png"))],
        )
        assert resp.status_code == 200
        assert resp.json()["image_count"] == 2
        assert len(sessions[sid]["context"].images) == 2

    @pytest.mark.asyncio
    async def test_add_image_404(self, client: AsyncClient):
        png = _make_tiny_png_bytes()
        resp = await client.post(
            "/api/diagnose/nonexistent/upload",
            files=[("file", ("leaf.png", png, "image/png"))],
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket /ws/diagnose/{session_id}
# ---------------------------------------------------------------------------


class TestDiagnoseWebSocket:
    @pytest.mark.asyncio
    async def test_ws_session_not_found(self, client: AsyncClient):
        """Connecting with a bad session_id should receive an error frame."""
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/diagnose/bad_session_id") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "not found" in msg["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_ws_debate_stream(self):
        """Full AVD+DDP flow should stream AVD assessment, agent messages, then result."""
        from starlette.testclient import TestClient

        from app.schemas import AVDAssessment, AVDStatus, DiagnosisContext

        # Inject a session
        sid = "ws_test_session"
        sessions[sid] = {
            "context": DiagnosisContext(images=[_make_tiny_b64()]),
            "status": "created",
            "result": None,
        }

        # Mock the VLM
        mock_vlm = MagicMock()
        app.state.vlm = mock_vlm

        async def fake_run_debate(context, on_message=None):
            msgs = DUMMY_RESULT.debate_transcript
            for m in msgs:
                if on_message:
                    on_message(m)
                await asyncio.sleep(0)
            return DUMMY_RESULT

        with patch(
            "app.api.websocket.AVDEngine"
        ) as MockAVD, patch(
            "app.api.websocket.DDPOrchestrator"
        ) as MockOrch:
            avd_instance = MockAVD.return_value
            avd_instance.assess = AsyncMock(return_value=AVDAssessment(
                status=AVDStatus.SUFFICIENT,
                confidence=0.9,
                question=None,
                summary="信息充分，可以进行诊断。",
            ))

            orch_instance = MockOrch.return_value
            orch_instance.run_debate = AsyncMock(side_effect=fake_run_debate)

            with TestClient(app) as tc:
                with tc.websocket_connect(f"/ws/diagnose/{sid}") as ws:
                    # Send config message (required by new protocol)
                    ws.send_json({"type": "config", "api_key": ""})

                    collected = []
                    while True:
                        try:
                            msg = ws.receive_json()
                            collected.append(msg)
                            # Answer interview questions automatically
                            if msg.get("type") == "avd_question":
                                ws.send_json({"action": "skip"})
                        except Exception:
                            break

        types = [m["type"] for m in collected]
        # 4 interview questions + 1 avd_sufficient + 1 status + 4 agent_message + 1 result
        assert types.count("avd_question") == 4
        assert "avd_sufficient" in types
        assert "status" in types
        assert types.count("agent_message") == 4
        assert "result" in types

        result_msg = next(m for m in collected if m["type"] == "result")
        assert result_msg["data"]["final_diagnosis"] == "小麦白粉病"
