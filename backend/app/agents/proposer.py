"""Proposer agent — initial diagnosis expert."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas import AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference

SYSTEM_PROMPT = (
    "你是一位资深植物病理学家（初诊专家）。根据用户提供的作物图片，给出最可能的诊断。"
    "输出：(1)观察到的症状特征 (2)初步诊断 (3)支持证据2-3条 (4)置信度（高/中/低）。"
    "用中文回答。"
)


class ProposerAgent(BaseAgent):
    """First-pass diagnostic expert."""

    def __init__(self, vlm: VLMInference) -> None:
        super().__init__(vlm, SYSTEM_PROMPT, AgentRole.PROPOSER)
