"""Challenger agent — diagnostic reviewer / skeptic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas import AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference

SYSTEM_PROMPT = (
    "你是一位严谨的植保审核专家（质疑专家）。审查初诊专家的诊断，找出漏洞，"
    "提出至少一个替代诊断及理由。如果初诊正确，说明验证了哪些方面。用中文回答。"
)


class ChallengerAgent(BaseAgent):
    """Second-opinion reviewer that challenges the initial diagnosis."""

    def __init__(self, vlm: VLMInference) -> None:
        super().__init__(vlm, SYSTEM_PROMPT, AgentRole.CHALLENGER)
