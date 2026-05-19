"""Arbiter agent — final decision maker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.schemas import AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference

SYSTEM_PROMPT = (
    "你是诊断委员会主席（仲裁专家）。综合双方意见做最终裁定。\n\n"
    "请严格按以下格式输出：\n"
    "✅ 最终诊断：<仅输出中文病害名称，不要加英文、学名或括号>\n"
    "支持证据：列出2-3条关键证据\n"
    "❌ 排除诊断：列出被排除的病害及排除原因\n"
    "⚠️ 不确定因素：如有不确定之处请说明\n"
    "置信度：高/中/低\n\n"
    "注意：最终诊断名称务必简洁，例如'番茄晚疫病'而非'番茄晚疫病（Phytophthora infestans）'。"
    "用中文回答。"
)


class ArbiterAgent(BaseAgent):
    """Committee chair that synthesises the debate into a final ruling."""

    def __init__(self, vlm: VLMInference) -> None:
        super().__init__(vlm, SYSTEM_PROMPT, AgentRole.ARBITER)
