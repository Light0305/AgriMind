"""Tests for Pydantic schemas — serialization / deserialization round-trips."""

import json

import pytest

from app.schemas import (
    AgentMessage,
    AgentRole,
    Confidence,
    DebateResult,
    DiagnosisContext,
    GroundingBox,
    RejectedDiagnosis,
)


# ---------------------------------------------------------------------------
# AgentRole / Confidence enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_confidence_values(self):
        assert Confidence.HIGH == "high"
        assert Confidence.MEDIUM == "medium"
        assert Confidence.LOW == "low"

    def test_agent_role_values(self):
        assert AgentRole.PROPOSER == "proposer"
        assert AgentRole.CHALLENGER == "challenger"
        assert AgentRole.ARBITER == "arbiter"

    def test_confidence_from_value(self):
        assert Confidence("high") is Confidence.HIGH

    def test_agent_role_from_value(self):
        assert AgentRole("arbiter") is AgentRole.ARBITER


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------

class TestAgentMessage:
    def test_create(self):
        msg = AgentMessage(role=AgentRole.PROPOSER, content="hello", round=1)
        assert msg.role == AgentRole.PROPOSER
        assert msg.content == "hello"
        assert msg.round == 1

    def test_json_round_trip(self):
        msg = AgentMessage(role=AgentRole.CHALLENGER, content="challenge!", round=2)
        data = json.loads(msg.model_dump_json())
        restored = AgentMessage.model_validate(data)
        assert restored == msg

    def test_role_string_coercion(self):
        """String values should be coerced to enum members."""
        msg = AgentMessage(role="proposer", content="x", round=1)
        assert msg.role is AgentRole.PROPOSER


# ---------------------------------------------------------------------------
# RejectedDiagnosis / GroundingBox
# ---------------------------------------------------------------------------

class TestSmallModels:
    def test_rejected_diagnosis(self):
        rd = RejectedDiagnosis(name="锈病", reason="无锈孢子堆")
        assert rd.name == "锈病"
        assert rd.reason == "无锈孢子堆"

    def test_grounding_box(self):
        gb = GroundingBox(x=10.0, y=20.0, width=50.0, height=30.0, label="病斑")
        assert gb.label == "病斑"
        assert gb.width == 50.0


# ---------------------------------------------------------------------------
# DiagnosisContext
# ---------------------------------------------------------------------------

class TestDiagnosisContext:
    def test_minimal(self):
        ctx = DiagnosisContext(images=["aGVsbG8="])
        assert ctx.images == ["aGVsbG8="]
        assert ctx.image_descriptions == []
        assert ctx.user_context is None
        assert ctx.conversation_history == []

    def test_full(self):
        ctx = DiagnosisContext(
            images=["aGVsbG8=", "d29ybGQ="],
            image_descriptions=["叶片正面", "叶片背面"],
            user_context="最近持续降雨",
            conversation_history=[{"role": "user", "content": "帮我看看"}],
        )
        assert len(ctx.images) == 2
        assert ctx.user_context == "最近持续降雨"

    def test_json_round_trip(self):
        ctx = DiagnosisContext(
            images=["aGVsbG8="],
            user_context="test",
        )
        restored = DiagnosisContext.model_validate_json(ctx.model_dump_json())
        assert restored == ctx


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------

class TestDebateResult:
    @pytest.fixture()
    def sample_result(self):
        return DebateResult(
            final_diagnosis="小麦白粉病",
            confidence=Confidence.HIGH,
            supporting_evidence=["叶面白色粉状物", "病斑扩展迅速"],
            rejected_diagnoses=[
                RejectedDiagnosis(name="锈病", reason="无锈孢子堆"),
            ],
            uncertainty_notes=["需检测是否有混合感染"],
            debate_transcript=[
                AgentMessage(role=AgentRole.PROPOSER, content="初诊…", round=1),
                AgentMessage(role=AgentRole.CHALLENGER, content="质疑…", round=1),
                AgentMessage(role=AgentRole.PROPOSER, content="回应…", round=2),
                AgentMessage(role=AgentRole.ARBITER, content="裁定…", round=2),
            ],
            grounding_boxes=[
                GroundingBox(x=0, y=0, width=100, height=100, label="病斑"),
            ],
        )

    def test_field_access(self, sample_result: DebateResult):
        assert sample_result.final_diagnosis == "小麦白粉病"
        assert sample_result.confidence is Confidence.HIGH
        assert len(sample_result.supporting_evidence) == 2
        assert len(sample_result.rejected_diagnoses) == 1
        assert len(sample_result.debate_transcript) == 4

    def test_json_round_trip(self, sample_result: DebateResult):
        raw = sample_result.model_dump_json()
        restored = DebateResult.model_validate_json(raw)
        assert restored == sample_result

    def test_defaults(self):
        """Minimal construction — optional lists default to empty."""
        result = DebateResult(
            final_diagnosis="测试",
            confidence=Confidence.LOW,
            supporting_evidence=[],
            rejected_diagnoses=[],
            debate_transcript=[],
        )
        assert result.uncertainty_notes == []
        assert result.grounding_boxes == []
