"""Arbiter agent — final decision maker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas import AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference

SYSTEM_PROMPT = (
    "你是诊断委员会主席（仲裁专家）。综合双方意见做最终裁定。"
    "使用结构化格式输出：✅最终诊断+支持证据, ❌排除诊断+排除原因, ⚠️不确定因素。"
    "用中文回答。"
)


class ArbiterAgent(BaseAgent):
    """Committee chair that synthesises the debate into a final ruling."""

    def __init__(self, vlm: VLMInference) -> None:
        super().__init__(vlm, SYSTEM_PROMPT, AgentRole.ARBITER)
