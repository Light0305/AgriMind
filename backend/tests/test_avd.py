"""Tests for the AVD (Active Visual Diagnosis) module."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.avd.engine import AVDEngine
from app.avd.session import AVDSession
from app.schemas import (
    AVDAssessment,
    AVDQuestion,
    AVDStatus,
    AgentMessage,
    AgentRole,
    Confidence,
    DebateResult,
    DiagnosisContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_image(color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a 4×4 PIL Image."""
    return Image.new("RGB", (4, 4), color=color)


def _make_tiny_b64_image() -> str:
    """Create a 4×4 red PNG and return its base64 encoding."""
    img = _make_tiny_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


CANNED_VLM_SUFFICIENT = json.dumps({
    "sufficient": True,
    "confidence": 0.9,
    "current_assessment": "叶面白色粉状物明显，初步判断为白粉病。",
    "if_insufficient": {
        "question": "请拍叶片背面",
        "reason": "需要确认背面是否有菌丝",
        "target_part": "叶片背面",
    },
})

CANNED_VLM_INSUFFICIENT = json.dumps({
    "sufficient": False,
    "confidence": 0.4,
    "current_assessment": "图片较模糊，无法确认病斑类型。",
    "if_insufficient": {
        "question": "请拍摄叶片病斑的近距离特写",
        "reason": "需要观察病斑边缘形态和颜色",
        "target_part": "病斑特写",
    },
})

CANNED_VLM_BORDERLINE = json.dumps({
    "sufficient": False,
    "confidence": 0.75,
    "current_assessment": "置信度刚好在阈值上。",
    "if_insufficient": {
        "question": "额外照片",
        "reason": "边界测试",
        "target_part": "任意",
    },
})

DUMMY_DEBATE_RESULT = DebateResult(
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


# ===========================================================================
# AVDSession tests
# ===========================================================================


class TestAVDSession:
    """Tests for AVDSession state management."""

    def test_initial_state(self):
        session = AVDSession(session_id="test-1")
        assert session.current_round == 0
        assert session.can_ask_more is True
        assert session.images == []
        assert session.descriptions == []
        assert session.questions_asked == []

    def test_add_image_increments_round(self):
        session = AVDSession(session_id="test-2")
        img = _make_tiny_image()
        session.add_image(img, "叶片正面")

        assert session.current_round == 1
        assert len(session.images) == 1
        assert session.descriptions == ["叶片正面"]

    def test_add_image_default_description(self):
        session = AVDSession(session_id="test-3")
        session.add_image(_make_tiny_image())
        assert session.descriptions == [""]

    def test_can_ask_more_respects_max_rounds(self):
        session = AVDSession(session_id="test-4", max_rounds=2)
        assert session.can_ask_more is True

        session.add_image(_make_tiny_image())
        assert session.can_ask_more is True  # round 1 < max 2

        session.add_image(_make_tiny_image())
        assert session.can_ask_more is False  # round 2 == max 2

    def test_can_ask_more_at_max(self):
        """Exactly at max_rounds → can_ask_more is False."""
        session = AVDSession(session_id="test-5", max_rounds=1)
        session.add_image(_make_tiny_image())
        assert session.can_ask_more is False

    def test_current_round_tracks_images(self):
        session = AVDSession(session_id="test-6")
        for i in range(5):
            session.add_image(_make_tiny_image())
            assert session.current_round == i + 1


class TestAVDSessionConversion:
    """Tests for AVDSession.to_diagnosis_context()."""

    def test_converts_images_to_base64(self):
        session = AVDSession(session_id="conv-1")
        session.add_image(_make_tiny_image((255, 0, 0)))
        session.add_image(_make_tiny_image((0, 255, 0)))

        ctx = session.to_diagnosis_context()
        assert isinstance(ctx, DiagnosisContext)
        assert len(ctx.images) == 2

        # Each image should be valid base64 that decodes to a PNG.
        for b64 in ctx.images:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            assert img.size == (4, 4)

    def test_preserves_descriptions(self):
        session = AVDSession(session_id="conv-2")
        session.add_image(_make_tiny_image(), "叶片正面")
        session.add_image(_make_tiny_image(), "叶片背面")

        ctx = session.to_diagnosis_context()
        assert ctx.image_descriptions == ["叶片正面", "叶片背面"]

    def test_preserves_user_context(self):
        session = AVDSession(session_id="conv-3", user_context="最近降雨较多")
        session.add_image(_make_tiny_image())

        ctx = session.to_diagnosis_context()
        assert ctx.user_context == "最近降雨较多"

    def test_empty_session(self):
        session = AVDSession(session_id="conv-4")
        ctx = session.to_diagnosis_context()
        assert ctx.images == []
        assert ctx.image_descriptions == []
        assert ctx.user_context is None


# ===========================================================================
# AVDEngine._parse_assessment tests
# ===========================================================================


class TestParseAssessment:
    """Tests for AVDEngine._parse_assessment() parsing logic."""

    @pytest.fixture()
    def engine(self):
        vlm = MagicMock()
        return AVDEngine(vlm)

    def test_parse_sufficient(self, engine: AVDEngine):
        result = engine._parse_assessment(CANNED_VLM_SUFFICIENT)
        assert result.status == AVDStatus.SUFFICIENT
        assert result.confidence == pytest.approx(0.9)
        assert result.question is None
        assert "白粉病" in result.summary

    def test_parse_insufficient(self, engine: AVDEngine):
        result = engine._parse_assessment(CANNED_VLM_INSUFFICIENT)
        assert result.status == AVDStatus.QUESTIONING
        assert result.confidence == pytest.approx(0.4)
        assert result.question is not None
        assert "特写" in result.question.question
        assert result.question.target_part == "病斑特写"

    def test_parse_borderline_confidence_at_threshold(self, engine: AVDEngine):
        """confidence == threshold → SUFFICIENT (>= check)."""
        result = engine._parse_assessment(CANNED_VLM_BORDERLINE, threshold=0.75)
        assert result.status == AVDStatus.SUFFICIENT

    def test_parse_borderline_confidence_below_threshold(self, engine: AVDEngine):
        """confidence < threshold → QUESTIONING."""
        result = engine._parse_assessment(CANNED_VLM_BORDERLINE, threshold=0.76)
        assert result.status == AVDStatus.QUESTIONING

    def test_parse_malformed_json(self, engine: AVDEngine):
        """Malformed JSON → defaults to SUFFICIENT."""
        result = engine._parse_assessment("{this is not valid json!}")
        assert result.status == AVDStatus.SUFFICIENT
        assert result.confidence == pytest.approx(0.5)

    def test_parse_no_json_at_all(self, engine: AVDEngine):
        """No JSON in response → defaults to SUFFICIENT."""
        result = engine._parse_assessment("这是纯文本，没有JSON。")
        assert result.status == AVDStatus.SUFFICIENT

    def test_parse_empty_response(self, engine: AVDEngine):
        result = engine._parse_assessment("")
        assert result.status == AVDStatus.SUFFICIENT

    def test_parse_json_in_markdown_fence(self, engine: AVDEngine):
        """JSON wrapped in markdown code fence."""
        wrapped = f"```json\n{CANNED_VLM_INSUFFICIENT}\n```"
        result = engine._parse_assessment(wrapped)
        assert result.status == AVDStatus.QUESTIONING

    def test_parse_json_with_surrounding_prose(self, engine: AVDEngine):
        """JSON embedded in explanatory text."""
        with_prose = f"根据分析结果：\n{CANNED_VLM_SUFFICIENT}\n以上是我的评估。"
        result = engine._parse_assessment(with_prose)
        assert result.status == AVDStatus.SUFFICIENT

    def test_parse_missing_if_insufficient_block(self, engine: AVDEngine):
        """sufficient=False but no if_insufficient block → uses defaults."""
        data = json.dumps({
            "sufficient": False,
            "confidence": 0.3,
            "current_assessment": "不确定",
        })
        result = engine._parse_assessment(data)
        assert result.status == AVDStatus.QUESTIONING
        assert result.question is not None
        assert result.question.question  # has a default

    def test_parse_sufficient_override_by_flag(self, engine: AVDEngine):
        """Even with low confidence, sufficient=True → SUFFICIENT."""
        data = json.dumps({
            "sufficient": True,
            "confidence": 0.1,
            "current_assessment": "症状明显",
        })
        result = engine._parse_assessment(data)
        assert result.status == AVDStatus.SUFFICIENT


# ===========================================================================
# AVDEngine.assess() tests (with mocked VLM)
# ===========================================================================


class TestAVDEngineAssess:
    """Tests for AVDEngine.assess() with mocked VLM."""

    @pytest.fixture()
    def mock_vlm(self):
        vlm = MagicMock()
        vlm.generate = AsyncMock()
        return vlm

    @pytest.fixture()
    def engine(self, mock_vlm):
        return AVDEngine(mock_vlm)

    @pytest.mark.asyncio
    async def test_assess_sufficient(self, engine: AVDEngine, mock_vlm):
        mock_vlm.generate.return_value = CANNED_VLM_SUFFICIENT
        session = AVDSession(session_id="assess-1")
        session.add_image(_make_tiny_image())

        result = await engine.assess(session)
        assert result.status == AVDStatus.SUFFICIENT
        assert mock_vlm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_assess_questioning(self, engine: AVDEngine, mock_vlm):
        mock_vlm.generate.return_value = CANNED_VLM_INSUFFICIENT
        session = AVDSession(session_id="assess-2")
        session.add_image(_make_tiny_image())

        result = await engine.assess(session)
        assert result.status == AVDStatus.QUESTIONING
        assert result.question is not None

    @pytest.mark.asyncio
    async def test_assess_forced_when_max_rounds(self, engine: AVDEngine, mock_vlm):
        """When max rounds reached, VLM should NOT be called — direct FORCED."""
        session = AVDSession(session_id="assess-3", max_rounds=1)
        session.add_image(_make_tiny_image())

        result = await engine.assess(session)
        assert result.status == AVDStatus.FORCED
        mock_vlm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_assess_tracks_questions_asked(self, engine: AVDEngine, mock_vlm):
        """Questions from QUESTIONING assessments are recorded in session."""
        mock_vlm.generate.return_value = CANNED_VLM_INSUFFICIENT
        session = AVDSession(session_id="assess-4", max_rounds=3)
        session.add_image(_make_tiny_image())

        await engine.assess(session)
        assert len(session.questions_asked) == 1
        assert "特写" in session.questions_asked[0].question

    @pytest.mark.asyncio
    async def test_assess_does_not_track_sufficient(self, engine: AVDEngine, mock_vlm):
        """Sufficient assessments don't add to questions_asked."""
        mock_vlm.generate.return_value = CANNED_VLM_SUFFICIENT
        session = AVDSession(session_id="assess-5")
        session.add_image(_make_tiny_image())

        await engine.assess(session)
        assert len(session.questions_asked) == 0

    @pytest.mark.asyncio
    async def test_assess_includes_user_context_in_prompt(
        self, engine: AVDEngine, mock_vlm
    ):
        mock_vlm.generate.return_value = CANNED_VLM_SUFFICIENT
        session = AVDSession(session_id="assess-6", user_context="近期持续降雨")
        session.add_image(_make_tiny_image(), "叶片正面")

        await engine.assess(session)

        # Verify the VLM was called and the user content includes context.
        call_args = mock_vlm.generate.call_args
        user_content = call_args.kwargs.get("user_content") or call_args[0][1]
        text_items = [c["text"] for c in user_content if c.get("type") == "text"]
        combined = " ".join(text_items)
        assert "降雨" in combined
        assert "叶片正面" in combined


# ===========================================================================
# AVDEngine.run_session() — full AVD → DDP flow
# ===========================================================================


class TestAVDRunSession:
    """Test the full AVD → DDP pipeline."""

    @pytest.fixture()
    def mock_vlm(self):
        vlm = MagicMock()
        vlm.generate = AsyncMock()
        return vlm

    @pytest.fixture()
    def mock_ddp(self):
        ddp = MagicMock()
        ddp.run_debate = AsyncMock(return_value=DUMMY_DEBATE_RESULT)
        return ddp

    @pytest.mark.asyncio
    async def test_sufficient_runs_ddp(self, mock_vlm, mock_ddp):
        """When AVD says sufficient, DDP debate should run."""
        mock_vlm.generate.return_value = CANNED_VLM_SUFFICIENT

        engine = AVDEngine(mock_vlm)
        session = AVDSession(session_id="flow-1")
        session.add_image(_make_tiny_image())

        assessments: list[AVDAssessment] = []
        result = await engine.run_session(
            session, mock_ddp, on_assessment=assessments.append
        )

        assert result is not None
        assert result.final_diagnosis == "小麦白粉病"
        assert len(assessments) == 1
        assert assessments[0].status == AVDStatus.SUFFICIENT
        mock_ddp.run_debate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forced_runs_ddp(self, mock_vlm, mock_ddp):
        """When max rounds reached (FORCED), DDP should still run."""
        engine = AVDEngine(mock_vlm)
        session = AVDSession(session_id="flow-2", max_rounds=1)
        session.add_image(_make_tiny_image())

        result = await engine.run_session(session, mock_ddp)

        assert result is not None
        assert result.final_diagnosis == "小麦白粉病"
        mock_ddp.run_debate.assert_awaited_once()
        mock_vlm.generate.assert_not_awaited()  # No VLM call for FORCED

    @pytest.mark.asyncio
    async def test_questioning_returns_none(self, mock_vlm, mock_ddp):
        """When AVD needs more images, run_session returns None."""
        mock_vlm.generate.return_value = CANNED_VLM_INSUFFICIENT

        engine = AVDEngine(mock_vlm)
        session = AVDSession(session_id="flow-3")
        session.add_image(_make_tiny_image())

        result = await engine.run_session(session, mock_ddp)

        assert result is None
        mock_ddp.run_debate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_message_callback_forwarded(self, mock_vlm, mock_ddp):
        """on_message callback should be forwarded to DDP."""
        mock_vlm.generate.return_value = CANNED_VLM_SUFFICIENT

        engine = AVDEngine(mock_vlm)
        session = AVDSession(session_id="flow-4")
        session.add_image(_make_tiny_image())

        messages: list[AgentMessage] = []
        await engine.run_session(
            session, mock_ddp, on_message=messages.append
        )

        # Verify on_message was passed to ddp.run_debate.
        call_kwargs = mock_ddp.run_debate.call_args.kwargs
        assert "on_message" in call_kwargs
        assert call_kwargs["on_message"] == messages.append

    @pytest.mark.asyncio
    async def test_multi_round_flow(self, mock_vlm, mock_ddp):
        """Simulate a multi-round AVD conversation."""
        engine = AVDEngine(mock_vlm)
        session = AVDSession(session_id="flow-5", max_rounds=3)
        session.add_image(_make_tiny_image())

        # Round 1: insufficient
        mock_vlm.generate.return_value = CANNED_VLM_INSUFFICIENT
        result = await engine.run_session(session, mock_ddp)
        assert result is None

        # User adds another image
        session.add_image(_make_tiny_image((0, 255, 0)))

        # Round 2: still insufficient
        mock_vlm.generate.return_value = CANNED_VLM_INSUFFICIENT
        result = await engine.run_session(session, mock_ddp)
        assert result is None

        # User adds a third image
        session.add_image(_make_tiny_image((0, 0, 255)))

        # Round 3: max rounds → FORCED, DDP runs
        result = await engine.run_session(session, mock_ddp)
        assert result is not None
        assert result.final_diagnosis == "小麦白粉病"


# ===========================================================================
# Schema tests for AVD types
# ===========================================================================


class TestAVDSchemas:
    """Tests for the AVD Pydantic models."""

    def test_avd_status_values(self):
        assert AVDStatus.QUESTIONING == "questioning"
        assert AVDStatus.SUFFICIENT == "sufficient"
        assert AVDStatus.FORCED == "forced"

    def test_avd_question_creation(self):
        q = AVDQuestion(
            question="请拍叶片背面",
            reason="需要确认孢子堆颜色",
            target_part="叶片背面",
        )
        assert q.question == "请拍叶片背面"
        assert q.reason == "需要确认孢子堆颜色"
        assert q.target_part == "叶片背面"

    def test_avd_assessment_sufficient(self):
        a = AVDAssessment(
            status=AVDStatus.SUFFICIENT,
            confidence=0.9,
            summary="信息充分",
        )
        assert a.question is None

    def test_avd_assessment_questioning(self):
        q = AVDQuestion(
            question="请拍特写",
            reason="需要细节",
            target_part="病斑",
        )
        a = AVDAssessment(
            status=AVDStatus.QUESTIONING,
            confidence=0.3,
            question=q,
            summary="信息不足",
        )
        assert a.question is q

    def test_avd_assessment_json_round_trip(self):
        q = AVDQuestion(
            question="请拍背面",
            reason="需要确认",
            target_part="叶片背面",
        )
        a = AVDAssessment(
            status=AVDStatus.QUESTIONING,
            confidence=0.4,
            question=q,
            summary="初步判断不确定",
        )
        restored = AVDAssessment.model_validate_json(a.model_dump_json())
        assert restored == a
