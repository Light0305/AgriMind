"""Tests for the DDP Orchestrator — uses mocked VLM inference."""

from __future__ import annotations

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from PIL import Image

from app.agents.ddp import DDPOrchestrator
from app.schemas import (
    AgentMessage,
    AgentRole,
    Confidence,
    DiagnosisContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tiny_b64_image() -> str:
    """Create a 4×4 red PNG and return its base64 encoding."""
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


CANNED_PROPOSER_R1 = (
    "观察症状：叶面出现白色粉状物覆盖，病斑逐渐扩展。\n"
    "初步诊断：小麦白粉病\n"
    "支持证据：\n"
    "1. 白色粉状菌层明显\n"
    "2. 病斑沿叶脉扩展\n"
    "置信度：高"
)

CANNED_CHALLENGER_R1 = (
    "审查意见：初诊基本合理，但需排除以下可能：\n"
    "替代诊断：小麦霜霉病\n"
    "理由：霜霉病也会在叶面形成白色霉层，但通常出现在叶背面。"
)

CANNED_PROPOSER_R2 = (
    "回应质疑：感谢审查。经对比分析：\n"
    "1. 白粉病菌层在叶面正面，霜霉则在背面，本样本为正面覆盖。\n"
    "2. 白粉病菌丝呈粉状，霜霉呈绒毛状。\n"
    "维持原诊断：小麦白粉病。"
)

CANNED_ARBITER = (
    "✅最终诊断：小麦白粉病\n"
    "支持证据：\n"
    "1. 叶面正面白色粉状菌层\n"
    "2. 病斑沿叶脉扩展特征符合\n"
    "3. 初诊与回应论证充分\n"
    "置信度：高\n"
    "❌排除诊断：\n"
    "小麦霜霉病：病原在叶背面，与本样本不符\n"
    "⚠️不确定因素：\n"
    "建议取样做显微镜检查确认病原菌种类"
)


@pytest.fixture()
def mock_vlm() -> MagicMock:
    """Create a mock VLMInference that returns canned responses in order."""
    vlm = MagicMock()
    vlm.generate = AsyncMock(
        side_effect=[
            CANNED_PROPOSER_R1,
            CANNED_CHALLENGER_R1,
            CANNED_PROPOSER_R2,
            CANNED_ARBITER,
        ]
    )
    return vlm


@pytest.fixture()
def context() -> DiagnosisContext:
    return DiagnosisContext(images=[_make_tiny_b64_image()])


@pytest.fixture()
def orchestrator(mock_vlm: MagicMock) -> DDPOrchestrator:
    return DDPOrchestrator(mock_vlm)


# ---------------------------------------------------------------------------
# Tests — full debate flow
# ---------------------------------------------------------------------------

class TestDDPOrchestrator:
    @pytest.mark.asyncio
    async def test_debate_runs_four_steps(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext, mock_vlm: MagicMock
    ):
        """VLM.generate must be called exactly 4 times."""
        result = await orchestrator.run_debate(context)
        assert mock_vlm.generate.await_count == 4

    @pytest.mark.asyncio
    async def test_on_message_called_four_times(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        """The on_message callback should fire once per agent turn."""
        messages: list[AgentMessage] = []
        await orchestrator.run_debate(context, on_message=messages.append)
        assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_message_roles_and_rounds(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        messages: list[AgentMessage] = []
        await orchestrator.run_debate(context, on_message=messages.append)

        assert messages[0].role == AgentRole.PROPOSER
        assert messages[0].round == 1
        assert messages[1].role == AgentRole.CHALLENGER
        assert messages[1].round == 1
        assert messages[2].role == AgentRole.PROPOSER
        assert messages[2].round == 2
        assert messages[3].role == AgentRole.ARBITER
        assert messages[3].round == 2

    @pytest.mark.asyncio
    async def test_result_diagnosis(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        result = await orchestrator.run_debate(context)
        assert "白粉病" in result.final_diagnosis
        assert result.confidence is Confidence.HIGH

    @pytest.mark.asyncio
    async def test_result_evidence(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        result = await orchestrator.run_debate(context)
        assert len(result.supporting_evidence) >= 1

    @pytest.mark.asyncio
    async def test_result_rejected(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        result = await orchestrator.run_debate(context)
        assert len(result.rejected_diagnoses) >= 1
        assert "霜霉" in result.rejected_diagnoses[0].name

    @pytest.mark.asyncio
    async def test_result_uncertainty(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        result = await orchestrator.run_debate(context)
        assert len(result.uncertainty_notes) >= 1

    @pytest.mark.asyncio
    async def test_transcript_in_result(
        self, orchestrator: DDPOrchestrator, context: DiagnosisContext
    ):
        result = await orchestrator.run_debate(context)
        assert len(result.debate_transcript) == 4


# ---------------------------------------------------------------------------
# Tests — _parse_result (unit)
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_extracts_diagnosis(self):
        result = DDPOrchestrator._parse_result(CANNED_ARBITER, [])
        assert "白粉病" in result.final_diagnosis

    def test_extracts_high_confidence(self):
        result = DDPOrchestrator._parse_result(CANNED_ARBITER, [])
        assert result.confidence is Confidence.HIGH

    def test_extracts_evidence(self):
        result = DDPOrchestrator._parse_result(CANNED_ARBITER, [])
        assert len(result.supporting_evidence) >= 2

    def test_extracts_rejected(self):
        result = DDPOrchestrator._parse_result(CANNED_ARBITER, [])
        names = [r.name for r in result.rejected_diagnoses]
        assert any("霜霉" in n for n in names)

    def test_extracts_uncertainty(self):
        result = DDPOrchestrator._parse_result(CANNED_ARBITER, [])
        assert len(result.uncertainty_notes) >= 1

    def test_fallback_on_unstructured_text(self):
        """Even with messy input, _parse_result should not crash."""
        result = DDPOrchestrator._parse_result("这是一段没有标记的文字", [])
        assert result.final_diagnosis  # falls back to default
        assert result.confidence is Confidence.MEDIUM

    def test_low_confidence(self):
        text = "✅最终诊断：疑似病害\n置信度：低"
        result = DDPOrchestrator._parse_result(text, [])
        assert result.confidence is Confidence.LOW


# ---------------------------------------------------------------------------
# Tests — _decode_images
# ---------------------------------------------------------------------------

class TestDecodeImages:
    def test_decode_basic(self):
        b64 = _make_tiny_b64_image()
        images = DDPOrchestrator._decode_images([b64])
        assert len(images) == 1
        assert images[0].size == (4, 4)
        assert images[0].mode == "RGB"

    def test_decode_data_uri(self):
        """Should handle data:image/png;base64,... prefix."""
        b64 = _make_tiny_b64_image()
        data_uri = f"data:image/png;base64,{b64}"
        images = DDPOrchestrator._decode_images([data_uri])
        assert len(images) == 1
        assert images[0].size == (4, 4)

    def test_decode_multiple(self):
        b64 = _make_tiny_b64_image()
        images = DDPOrchestrator._decode_images([b64, b64, b64])
        assert len(images) == 3
