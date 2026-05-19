"""Challenger agent — diagnostic reviewer / skeptic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas import AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference

SYSTEM_PROMPT = (
    "你是一位严谨的植保审核专家（质疑专家）。审查初诊专家的诊断，找出漏洞，"
    "提出至少一个替代诊断及理由。如果初诊正确，说明验证了哪些方面。\n\n"
    "输出格式：\n"
    "1. 审查结论（同意/不同意初诊）\n"
    "2. 具体分析\n"
    "3. 如有替代诊断，格式为'替代诊断：<仅输出中文病害名称，不加英文或学名>'\n\n"
    "用中文回答。"
)


class ChallengerAgent(BaseAgent):
    """Second-opinion reviewer that challenges the initial diagnosis."""

    def __init__(self, vlm: VLMInference) -> None:
        super().__init__(vlm, SYSTEM_PROMPT, AgentRole.CHALLENGER)
