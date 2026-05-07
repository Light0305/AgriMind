"""Pydantic data models for the DDP debate protocol."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentRole(str, Enum):
    PROPOSER = "proposer"
    CHALLENGER = "challenger"
    ARBITER = "arbiter"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    """A single message produced by a DDP agent during debate."""

    role: AgentRole
    content: str
    round: int  # 1 or 2


class RejectedDiagnosis(BaseModel):
    """A diagnosis explicitly ruled out by the arbiter."""

    name: str
    reason: str


class GroundingBox(BaseModel):
    """Bounding box highlight on the original image."""

    x: float
    y: float
    width: float
    height: float
    label: str


# ---------------------------------------------------------------------------
# DDP I/O
# ---------------------------------------------------------------------------

class DiagnosisContext(BaseModel):
    """Input payload for the DDP engine."""

    images: list[str]  # base64-encoded images
    image_descriptions: list[str] = Field(default_factory=list)
    user_context: str | None = None
    conversation_history: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AVD (Active Visual Diagnosis) types
# ---------------------------------------------------------------------------

class AVDStatus(str, Enum):
    QUESTIONING = "questioning"  # AVD is asking for more photos
    SUFFICIENT = "sufficient"    # Enough info, ready for DDP
    FORCED = "forced"           # Max rounds reached, proceed anyway


class AVDQuestion(BaseModel):
    """A follow-up question from the AVD engine."""

    question: str       # The question text ("请拍叶片背面...")
    reason: str         # Why this info is needed ("需要确认孢子堆颜色")
    target_part: str    # What to photograph ("叶片背面")


class AVDAssessment(BaseModel):
    """Information sufficiency assessment."""

    status: AVDStatus
    confidence: float                    # 0-1, how confident we are with current info
    question: AVDQuestion | None = None  # Only if status == QUESTIONING
    summary: str                         # Current understanding summary


# ---------------------------------------------------------------------------
# DDP output
# ---------------------------------------------------------------------------

class DebateResult(BaseModel):
    """Structured output produced by the DDP engine."""

    final_diagnosis: str
    confidence: Confidence
    supporting_evidence: list[str]
    rejected_diagnoses: list[RejectedDiagnosis]
    uncertainty_notes: list[str] = Field(default_factory=list)
    debate_transcript: list[AgentMessage]
    grounding_boxes: list[GroundingBox] = Field(default_factory=list)
