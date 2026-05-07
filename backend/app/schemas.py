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


class DebateResult(BaseModel):
    """Structured output produced by the DDP engine."""

    final_diagnosis: str
    confidence: Confidence
    supporting_evidence: list[str]
    rejected_diagnoses: list[RejectedDiagnosis]
    uncertainty_notes: list[str] = Field(default_factory=list)
    debate_transcript: list[AgentMessage]
    grounding_boxes: list[GroundingBox] = Field(default_factory=list)
