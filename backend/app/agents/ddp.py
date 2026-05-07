"""DDP Orchestrator — runs the full 2-round diagnostic debate."""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import TYPE_CHECKING, Callable

from PIL import Image

from app.agents.arbiter import ArbiterAgent
from app.agents.challenger import ChallengerAgent
from app.agents.proposer import ProposerAgent
from app.schemas import (
    AgentMessage,
    Confidence,
    DebateResult,
    DiagnosisContext,
    RejectedDiagnosis,
)

if TYPE_CHECKING:
    from app.model.inference import VLMInference

logger = logging.getLogger(__name__)


class DDPOrchestrator:
    """Orchestrates the 2-round DDP debate protocol.

    Flow
    ----
    1. **Round 1 — Proposer** produces an initial diagnosis.
    2. **Round 1 — Challenger** reviews and challenges.
    3. **Round 2 — Proposer** responds to the challenge.
    4. **Round 2 — Arbiter** delivers the final ruling.

    Each step optionally invokes *on_message* so a WebSocket handler can
    stream intermediate results to the frontend.
    """

    def __init__(self, vlm: VLMInference) -> None:
        self.proposer = ProposerAgent(vlm)
        self.challenger = ChallengerAgent(vlm)
        self.arbiter = ArbiterAgent(vlm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_debate(
        self,
        context: DiagnosisContext,
        on_message: Callable[[AgentMessage], None] | None = None,
    ) -> DebateResult:
        """Run the full 2-round DDP debate.

        Args:
            context: Diagnosis input (images + optional text context).
            on_message: Optional callback fired after every agent turn
                        (useful for WebSocket streaming).

        Returns:
            Structured :class:`DebateResult`.
        """
        images = self._decode_images(context.images)

        # Optional extra context from the user
        extra = ""
        if context.user_context:
            extra = f"\n用户补充信息：{context.user_context}"
        if context.image_descriptions:
            extra += "\n图片描述：" + "；".join(context.image_descriptions)

        # ── Round 1: Proposer diagnoses ───────────────────────────────
        proposer_r1 = await self.proposer.run(
            images,
            f"请对这张作物图片进行诊断。{extra}",
            round_num=1,
        )
        if on_message:
            on_message(proposer_r1)

        # ── Round 1: Challenger challenges ────────────────────────────
        challenger_r1 = await self.challenger.run(
            images,
            f"初诊结果：\n{proposer_r1.content}\n\n请审查并提出质疑。",
            round_num=1,
        )
        if on_message:
            on_message(challenger_r1)

        # ── Round 2: Proposer responds to challenge ───────────────────
        proposer_r2 = await self.proposer.run(
            images,
            (
                f"你之前的诊断：\n{proposer_r1.content}\n\n"
                f"质疑：\n{challenger_r1.content}\n\n请回应。"
            ),
            round_num=2,
        )
        if on_message:
            on_message(proposer_r2)

        # ── Round 2: Arbiter arbitrates ───────────────────────────────
        debate_record = (
            f"初诊：\n{proposer_r1.content}\n\n"
            f"质疑：\n{challenger_r1.content}\n\n"
            f"回应：\n{proposer_r2.content}"
        )
        arbiter_result = await self.arbiter.run(
            images,
            f"辩论记录：\n{debate_record}\n\n请做最终裁定。",
            round_num=2,
        )
        if on_message:
            on_message(arbiter_result)

        # ── Assemble structured result ────────────────────────────────
        transcript = [proposer_r1, challenger_r1, proposer_r2, arbiter_result]
        return self._parse_result(arbiter_result.content, transcript)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_images(base64_images: list[str]) -> list[Image.Image]:
        """Decode a list of base64-encoded images into PIL Images."""
        result: list[Image.Image] = []
        for b64 in base64_images:
            # Strip optional data-URI prefix
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)
            result.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        return result

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(
        arbiter_text: str,
        transcript: list[AgentMessage],
    ) -> DebateResult:
        """Best-effort extraction of structured fields from arbiter prose.

        The arbiter is prompted to use ✅ / ❌ / ⚠️ markers, so we key on
        those.  Anything we can't extract falls back to safe defaults.
        """

        # --- Final diagnosis ---
        final_diagnosis = "未能确定诊断"
        diag_match = re.search(r"✅\s*最终诊断[：:]\s*(.+)", arbiter_text)
        if diag_match:
            final_diagnosis = diag_match.group(1).strip()
        else:
            # Fallback: grab the first line after ✅
            diag_match = re.search(r"✅\s*(.+)", arbiter_text)
            if diag_match:
                final_diagnosis = diag_match.group(1).strip()

        # --- Confidence ---
        confidence = Confidence.MEDIUM
        if re.search(r"置信度[：:]\s*高|高置信|confidence.*high", arbiter_text, re.I):
            confidence = Confidence.HIGH
        elif re.search(r"置信度[：:]\s*低|低置信|confidence.*low", arbiter_text, re.I):
            confidence = Confidence.LOW

        # --- Supporting evidence ---
        evidence: list[str] = []
        # Look for numbered evidence lines after 支持证据
        ev_block = re.search(r"支持证据[：:]([\s\S]*?)(?=❌|⚠️|\Z)", arbiter_text)
        if ev_block:
            for line in ev_block.group(1).strip().splitlines():
                line = re.sub(r"^\s*[\d\-•]+[.、)\s]*", "", line).strip()
                if line:
                    evidence.append(line)
        # Fallback: lines starting with a digit after ✅
        if not evidence:
            for line in arbiter_text.splitlines():
                m = re.match(r"\s*\d+[.、)]\s*(.+)", line)
                if m:
                    evidence.append(m.group(1).strip())

        # --- Rejected diagnoses ---
        rejected: list[RejectedDiagnosis] = []
        rej_block = re.search(r"❌([\s\S]*?)(?=⚠️|\Z)", arbiter_text)
        if rej_block:
            for line in rej_block.group(1).strip().splitlines():
                line = re.sub(r"^\s*[\d\-•]+[.、)\s]*", "", line).strip()
                # Skip header lines like "排除诊断：" or empty
                if not line or re.match(r"^排除诊断[：:]?\s*$", line):
                    continue
                # Try to split into name + reason at a colon / dash
                parts = re.split(r"[：:—\-–]\s*", line, maxsplit=1)
                name = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else "见辩论记录"
                if name:
                    rejected.append(RejectedDiagnosis(name=name, reason=reason))

        # --- Uncertainty notes ---
        uncertainty: list[str] = []
        unc_block = re.search(r"⚠️([\s\S]*)", arbiter_text)
        if unc_block:
            for line in unc_block.group(1).strip().splitlines():
                line = re.sub(r"^\s*[\d\-•]+[.、)\s]*", "", line).strip()
                if line:
                    uncertainty.append(line)

        return DebateResult(
            final_diagnosis=final_diagnosis,
            confidence=confidence,
            supporting_evidence=evidence,
            rejected_diagnoses=rejected,
            uncertainty_notes=uncertainty,
            debate_transcript=transcript,
        )
