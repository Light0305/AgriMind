"""Base class for all DDP agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from app.schemas import AgentMessage, AgentRole

if TYPE_CHECKING:
    from app.model.inference import VLMInference


class BaseAgent:
    """Abstract base for Proposer / Challenger / Arbiter.

    Each concrete agent supplies its own *system_prompt* and *role*.
    """

    def __init__(
        self,
        vlm: VLMInference,
        system_prompt: str,
        role: AgentRole,
    ) -> None:
        self.vlm = vlm
        self.system_prompt = system_prompt
        self.role = role

    async def run(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        round_num: int = 1,
    ) -> AgentMessage:
        """Execute a single inference turn and return a structured message.

        Args:
            images: PIL images of the crop to diagnose.
            prompt: The textual instruction / context for this turn.
            round_num: DDP round number (1 or 2).
        """
        user_content: list[dict] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})

        response = await self.vlm.generate(
            system_prompt=self.system_prompt,
            user_content=user_content,
            images=images,
        )
        return AgentMessage(role=self.role, content=response, round=round_num)
