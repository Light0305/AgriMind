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
你需要判断：当前的图片是否足以做出可靠诊断？

你必须输出一个JSON对象（不要markdown代码块），格式如下：
{
    "sufficient": false,
    "confidence": 0.0到1.0之间的数字,
    "current_assessment": "你对当前图片的分析（描述你看到了什么）",
    "if_insufficient": {
        "question": "你需要用户补充什么（具体描述，不要套用模板）",
        "reason": "为什么需要这个信息",
        "target_part": "需要拍摄的具体部位"
    }
}

当sufficient为true时，可以省略if_insufficient字段。

判断标准：
- 只有1张图片时，通常信息不够充分，confidence不应超过0.6
- 图片模糊、角度单一、无法看清病斑形态/颜色/分布时 → 信息不足
- 能清晰看到病斑特征（形状、颜色、大小、分布）且有多角度照片 → 信息充分
- 追问要具体：比如"请拍摄叶片背面的病斑特写"而不是"请提供更多照片"
- 不要总是建议拍植株整体——要根据当前缺失的信息来决定拍什么"""

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
        4. Apply heuristic: with only 1 image, raise threshold to 0.90.
        5. If the VLM says *sufficient* **and** confidence ≥ threshold →
           **SUFFICIENT**.
        6. Otherwise → **QUESTIONING** with a follow-up question.
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

        # Heuristic: with only 1 image, require much higher confidence.
        effective_threshold = session.sufficiency_threshold
        if len(session.images) <= 1:
            effective_threshold = max(effective_threshold, 0.90)

        assessment = self._parse_assessment(
            raw_response,
            threshold=effective_threshold,
            num_images=len(session.images),
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
        num_images: int = 1,
    ) -> AVDAssessment:
        """Parse VLM JSON response into an :class:`AVDAssessment`."""
        # Strip markdown code fences
        clean = response
        for fence in ("```json", "```"):
            clean = clean.replace(fence, "")
        clean = clean.strip()

        json_match = re.search(r"\{[\s\S]*\}", clean)
        if not json_match:
            logger.warning("AVD: no JSON found, asking follow-up")
            return AVDAssessment(
                status=AVDStatus.QUESTIONING,
                confidence=0.3,
                question=AVDQuestion(
                    question="请换个角度再拍一张，当前图片信息不够清晰",
                    reason="无法解析评估结果",
                    target_part="其他角度",
                ),
                summary=response[:200] if response else "无法解析评估结果",
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("AVD: malformed JSON, asking follow-up")
            return AVDAssessment(
                status=AVDStatus.QUESTIONING,
                confidence=0.3,
                question=AVDQuestion(
                    question="请补充拍摄病害部位的细节特写",
                    reason="评估结果格式异常，需要补充信息",
                    target_part="病斑特写",
                ),
                summary=response[:200] if response else "无法解析评估结果",
            )

        # Extract fields with safe defaults.
        sufficient = bool(data.get("sufficient", False))
        confidence = float(data.get("confidence", 0.3))
        summary = str(data.get("current_assessment", ""))

        # Clamp confidence: with only 1 image, cap at 0.7 to prevent
        # the model from blindly outputting high confidence.
        if num_images <= 1:
            confidence = min(confidence, 0.7)

        # Decide status: require BOTH sufficient=true AND confidence ≥ threshold.
        if sufficient and confidence >= threshold:
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

        # If the model said sufficient but confidence is low, or said
        # insufficient but didn't provide a question, generate a default.
        default_question = (
            "请从另一个角度拍摄病害部位的特写照片"
            if num_images <= 1
            else "请补充拍摄您认为最严重的病害部位"
        )

        question = AVDQuestion(
            question=str(q_data.get("question", default_question)),
            reason=str(q_data.get("reason", "需要更多视角以确认病害特征")),
            target_part=str(q_data.get("target_part", "病害部位特写")),
        )

        return AVDAssessment(
            status=AVDStatus.QUESTIONING,
            confidence=confidence,
            question=question,
            summary=summary or "信息不足，需要补充照片。",
        )
