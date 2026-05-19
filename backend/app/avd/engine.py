"""AVD Engine — decides when current images are sufficient for diagnosis."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable

from app.avd.session import AVDSession
from app.schemas import AVDAssessment, AVDQuestion, AVDStatus, DebateResult

if TYPE_CHECKING:
    from app.agents.ddp import DDPOrchestrator
    from app.model.inference import VLMInference
    from app.schemas import AgentMessage

logger = logging.getLogger(__name__)


class AVDEngine:
    """Active Visual Diagnosis engine — decides when to ask for more photos.

    Sits between the user and the DDP debate engine.  For each session it
    evaluates whether the collected images are sufficient for a reliable
    diagnosis.  If not, it formulates a targeted follow-up question telling
    the user exactly *what* to photograph and *why*.
    """

    ASSESSMENT_SYSTEM_PROMPT = """\
你是一位经验丰富的植物病理学家，正在进行远程问诊。
用户已通过文字描述了种植地区、季节、天气和观察到的症状。
请评估当前收到的图片是否足以做出可靠诊断。

输出JSON（不要markdown代码块）：
{
    "sufficient": true,
    "confidence": 0.85,
    "current_assessment": "叶片正面症状清晰，可见典型褐色同心轮纹，足以判断为番茄早疫病"
}

如果信息不足，将sufficient设为false，并在if_insufficient中说明需要补充什么：
{
    "sufficient": false,
    "confidence": 0.5,
    "current_assessment": "叶片正面有病斑但看不清细节",
    "if_insufficient": {
        "question": "具体追问文字（请根据实际缺失内容生成，不要套用模板）",
        "reason": "追问的科学依据",
        "target_part": "需要拍摄的部位"
    }
}

关键规则：
1. 如果图片症状清晰典型 → sufficient=true，不需要追问
2. 如果图片模糊或关键特征不可见 → 追问具体缺失的部位
3. 不要总是建议拍植株整体——除非确实需要看整体
4. 追问必须针对当前图片中看不到的信息"""

    def __init__(self, vlm: VLMInference) -> None:
        self.vlm = vlm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assess(self, session: AVDSession) -> AVDAssessment:
        """Assess whether current images are sufficient for diagnosis.

        Decision logic
        ~~~~~~~~~~~~~~
        1. If ``session.can_ask_more`` is ``False`` → **FORCED** (proceed
           with whatever we have).
        2. Call VLM to assess sufficiency.
        3. Parse the JSON response.
        4. If the VLM says *sufficient* **or** confidence ≥ threshold →
           **SUFFICIENT**.
        5. Otherwise → **QUESTIONING** with a follow-up question.
        """
        # Gate: max rounds reached — force DDP to proceed.
        if not session.can_ask_more:
            return AVDAssessment(
                status=AVDStatus.FORCED,
                confidence=0.0,
                question=None,
                summary="已达最大问诊轮次，将直接进入专家会诊。",
            )

        # Build multimodal content for the VLM.
        user_content: list[dict] = []
        for img in session.images:
            user_content.append({"type": "image", "image": img})

        # Include any descriptions and prior questions for context.
        context_parts: list[str] = []
        if session.descriptions:
            descs = "；".join(d for d in session.descriptions if d)
            if descs:
                context_parts.append(f"图片描述：{descs}")
        if session.user_context:
            context_parts.append(f"用户补充信息：{session.user_context}")
        if session.questions_asked:
            asked = "；".join(q.question for q in session.questions_asked)
            context_parts.append(f"之前已提问：{asked}")

        prompt_text = (
            f"当前共收到 {len(session.images)} 张图片。"
            + ("　" + "　".join(context_parts) if context_parts else "")
            + "\n请评估信息是否充分，并输出JSON。"
        )
        user_content.append({"type": "text", "text": prompt_text})

        # Call VLM.
        raw_response = await self.vlm.generate(
            system_prompt=self.ASSESSMENT_SYSTEM_PROMPT,
            user_content=user_content,
            images=list(session.images),
        )

        assessment = self._parse_assessment(
            raw_response,
            threshold=session.sufficiency_threshold,
        )

        # Track the question we just asked so we don't repeat it.
        if assessment.status == AVDStatus.QUESTIONING and assessment.question:
            session.questions_asked.append(assessment.question)

        return assessment

    async def run_session(
        self,
        session: AVDSession,
        ddp: DDPOrchestrator,
        on_assessment: Callable[[AVDAssessment], None] | None = None,
        on_message: Callable[[AgentMessage], None] | None = None,
    ) -> DebateResult | None:
        """Run a complete AVD → DDP pipeline step.

        Returns
        -------
        DebateResult
            When AVD decides the information is sufficient (or forced) and
            the DDP debate runs to completion.
        None
            When AVD needs more images — the caller should collect another
            image from the user and call ``run_session`` again.
        """
        assessment = await self.assess(session)

        if on_assessment:
            on_assessment(assessment)

        if assessment.status in (AVDStatus.SUFFICIENT, AVDStatus.FORCED):
            context = session.to_diagnosis_context()
            return await ddp.run_debate(context, on_message=on_message)

        # QUESTIONING — caller needs to gather another image.
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_assessment(
        self,
        response: str,
        *,
        threshold: float = 0.75,
    ) -> AVDAssessment:
        """Parse VLM JSON response into an :class:`AVDAssessment`.

        Handles malformed JSON gracefully — defaults to **SUFFICIENT** if
        parsing fails (better to attempt a diagnosis than to loop forever).
        """
        # Try to extract a JSON object from the response (the VLM may wrap
        # the JSON in markdown fences or add surrounding prose).
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("AVD: no JSON found in VLM response, defaulting to SUFFICIENT")
            return AVDAssessment(
                status=AVDStatus.SUFFICIENT,
                confidence=0.5,
                question=None,
                summary=response[:200] if response else "无法解析评估结果",
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("AVD: malformed JSON in VLM response, defaulting to SUFFICIENT")
            return AVDAssessment(
                status=AVDStatus.SUFFICIENT,
                confidence=0.5,
                question=None,
                summary=response[:200] if response else "无法解析评估结果",
            )

        # Extract fields with safe defaults.
        sufficient = bool(data.get("sufficient", True))
        confidence = float(data.get("confidence", 0.5))
        summary = str(data.get("current_assessment", ""))

        # Decide status.
        if sufficient or confidence >= threshold:
            return AVDAssessment(
                status=AVDStatus.SUFFICIENT,
                confidence=confidence,
                question=None,
                summary=summary or "信息充分，可以进行诊断。",
            )

        # Insufficient — build the follow-up question.
        q_data = data.get("if_insufficient", {})
        if not isinstance(q_data, dict):
            q_data = {}

        question = AVDQuestion(
            question=str(q_data.get("question", "请提供更多照片以辅助诊断")),
            reason=str(q_data.get("reason", "需要更多信息")),
            target_part=str(q_data.get("target_part", "其他部位")),
        )

        return AVDAssessment(
            status=AVDStatus.QUESTIONING,
            confidence=confidence,
            question=question,
            summary=summary or "信息不足，需要补充照片。",
        )
