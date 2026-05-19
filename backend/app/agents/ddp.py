"""DDP Orchestrator — runs the full 2-round diagnostic debate with enhancements.

Enhancements (v2):
  1. Memory-Augmented: injects similar historical cases into agent prompts
     so agents can reference past successful diagnoses.  (MeMAD-inspired)
  2. Selective Debate: if Proposer and Challenger reach consensus in Round 1,
     the second round is skipped, saving ~50% inference cost.  (SELENE-inspired)
  3. Guideline-Grounded Arbiter: the Arbiter receives RAG-retrieved plant-
     protection knowledge for authoritative, evidence-backed rulings.  (GLEAN-inspired)
"""

from __future__ import annotations

import base64
import io
import logging
import re
from difflib import SequenceMatcher
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
    RetrievedDocument,
    SimilarCase,
)

if TYPE_CHECKING:
    from app.model.inference import VLMInference
    from app.rag.retriever import KnowledgeRetriever
    from app.retrieval.similar_cases import SimilarCaseRetriever

logger = logging.getLogger(__name__)

# Threshold for early-consensus: if the Challenger's text contains any of
# these agreement patterns we consider it a consensus and skip Round 2.
_CONSENSUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"(同意|认同|赞成|确认|一致|无异议).{0,10}(初诊|诊断|判断|结论)"),
    re.compile(r"(初诊|诊断|判断|结论).{0,10}(正确|准确|合理|无误|可靠)"),
    re.compile(r"(没有|未发现|无明显).{0,5}(异议|分歧|不同|错误|问题|漏洞)"),
    re.compile(r"(我同意|我认为对|判断一致)"),
]

# Minimum similarity ratio (0-1) between Proposer and Challenger proposals
# that also counts as consensus.
_CONSENSUS_SIMILARITY_THRESHOLD = 0.4


def _check_consensus(proposer_text: str, challenger_text: str) -> bool:
    """Check whether the Challenger effectively agrees with the Proposer."""
    # Pattern-based check — Challenger explicitly signals agreement.
    for pat in _CONSENSUS_PATTERNS:
        if pat.search(challenger_text):
            logger.info("Selective debate: consensus detected (pattern match)")
            return True

    # Fallback: text-similarity heuristic — if the two responses overlap
    # substantially the Challenger is likely confirming rather than challenging.
    similarity = SequenceMatcher(None, proposer_text, challenger_text).ratio()
    if similarity > _CONSENSUS_SIMILARITY_THRESHOLD:
        logger.info(
            "Selective debate: consensus detected (similarity=%.2f)", similarity
        )
        return True

    return False


# ---------------------------------------------------------------------------
# DDP Orchestrator
# ---------------------------------------------------------------------------


class DDPOrchestrator:
    """Orchestrates the DDP debate protocol with memory, selectivity, and RAG.

    Flow (enhanced)
    ---------------
    1. **Retrieve similar cases** from the historical case library.
    2. **Round 1 — Proposer** produces an initial diagnosis (with case context).
    3. **Round 1 — Challenger** reviews and challenges (with case context).
    4. **Consensus check** — if Challenger agrees, skip to Arbiter.
    5. **Round 2 — Proposer** responds to the challenge (only if needed).
    6. **Retrieve knowledge** (treatment guidelines) for the candidate diagnosis.
    7. **Round 2 — Arbiter** delivers the final ruling (with guideline context).

    Each step optionally invokes *on_message* so a WebSocket handler can
    stream intermediate results to the frontend.
    """

    def __init__(
        self,
        vlm: VLMInference,
        *,
        case_retriever: SimilarCaseRetriever | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.proposer = ProposerAgent(vlm)
        self.challenger = ChallengerAgent(vlm)
        self.arbiter = ArbiterAgent(vlm)
        self.case_retriever = case_retriever
        self.knowledge_retriever = knowledge_retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_debate(
        self,
        context: DiagnosisContext,
        on_message: Callable[[AgentMessage], None] | None = None,
    ) -> DebateResult:
        """Run the full DDP debate (v2 — memory-augmented & selective).

        Args:
            context: Diagnosis input (images + optional text context).
            on_message: Optional callback fired after every agent turn
                        (useful for WebSocket streaming).

        Returns:
            Structured :class:`DebateResult`.
        """
        images = self._decode_images(context.images)

        # ── Build extra context (user input) ────────────────────────────
        extra = ""
        if context.user_context:
            extra = f"\n用户补充信息：{context.user_context}"
        if context.image_descriptions:
            extra += "\n图片描述：" + "；".join(context.image_descriptions)

        # ── Enhancement 1: Retrieve similar historical cases ─────────────
        similar_cases: list[SimilarCase] = []
        case_context = ""
        if self.case_retriever is not None and images:
            try:
                similar_cases = self.case_retriever.find_similar(
                    images[0], top_k=3
                )
                if similar_cases:
                    case_context = self._format_case_context(similar_cases)
            except Exception:
                logger.warning("Failed to retrieve similar cases", exc_info=True)

        # ── Round 1: Proposer diagnoses ─────────────────────────────────
        proposer_r1 = await self.proposer.run(
            images,
            f"请对这张作物图片进行诊断。{case_context}{extra}",
            round_num=1,
        )
        if on_message:
            on_message(proposer_r1)

        # ── Round 1: Challenger challenges ──────────────────────────────
        challenger_r1 = await self.challenger.run(
            images,
            (
                f"初诊结果：\n{proposer_r1.content}\n\n"
                f"{case_context}"
                f"请审查并提出质疑。"
            ),
            round_num=1,
        )
        if on_message:
            on_message(challenger_r1)

        # ── Enhancement 2: Selective debate — consensus check ────────────
        consensus = _check_consensus(proposer_r1.content, challenger_r1.content)
        skipped_r2 = False

        if consensus:
            # Skip Round 2 — both agents agree; go directly to Arbiter.
            proposer_r2 = proposer_r1  # carry forward for transcript
            skipped_r2 = True
            logger.info("Selective debate: Round 2 skipped (early consensus)")
        else:
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

        # ── Enhancement 3: Retrieve treatment guidelines ─────────────────
        treatment_context = ""
        if self.knowledge_retriever is not None:
            # Extract a short diagnosis label from the arbiter's input
            try:
                treatment_context = self.knowledge_retriever.get_treatment_context(
                    proposer_r1.content[:200]  # use proposer's diagnosis
                )
            except Exception:
                logger.warning(
                    "Failed to retrieve treatment knowledge", exc_info=True
                )

        # ── Arbiter arbitrates ──────────────────────────────────────────
        if skipped_r2:
            debate_record = (
                f"初诊：\n{proposer_r1.content}\n\n"
                f"审查：\n{challenger_r1.content}\n\n"
                f"[双方达成共识 — 无需第二轮辩论]"
            )
        else:
            debate_record = (
                f"初诊：\n{proposer_r1.content}\n\n"
                f"质疑：\n{challenger_r1.content}\n\n"
                f"回应：\n{proposer_r2.content}"
            )

        arbiter_prompt = (
            f"辩论记录：\n{debate_record}\n\n"
            f"{treatment_context}"
            f"请做最终裁定。"
        )
        arbiter_result = await self.arbiter.run(
            images, arbiter_prompt, round_num=2,
        )
        if on_message:
            on_message(arbiter_result)

        # ── Assemble structured result ───────────────────────────────────
        transcript = [proposer_r1, challenger_r1, proposer_r2, arbiter_result]
        result = self._parse_result(arbiter_result.content, transcript)
        result.similar_cases = similar_cases

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_case_context(cases: list[SimilarCase]) -> str:
        """Build a concise reference block from similar historical cases."""
        lines = ["\n【相似历史病例参考】"]
        for i, c in enumerate(cases, 1):
            lines.append(
                f"  {i}. {c.label} (相似度: {c.similarity:.0%}, 来源: {c.source})"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _decode_images(base64_images: list[str]) -> list[Image.Image]:
        """Decode a list of base64-encoded images into PIL Images."""
        result: list[Image.Image] = []
        for b64 in base64_images:
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
        """Best-effort extraction of structured fields from arbiter prose."""
        # --- Final diagnosis ---
        final_diagnosis = "未能确定诊断"
        diag_match = re.search(r"✅\s*最终诊断[：:]\s*(.+)", arbiter_text)
        if diag_match:
            final_diagnosis = diag_match.group(1).strip()
        else:
            diag_match = re.search(r"✅\s*(.+)", arbiter_text)
            if diag_match:
                final_diagnosis = diag_match.group(1).strip()

        # --- Confidence ---
        confidence = Confidence.MEDIUM
        if re.search(
            r"置信度[：:]\s*高|高置信|confidence.*high", arbiter_text, re.I
        ):
            confidence = Confidence.HIGH
        elif re.search(
            r"置信度[：:]\s*低|低置信|confidence.*low", arbiter_text, re.I
        ):
            confidence = Confidence.LOW

        # --- Supporting evidence ---
        evidence: list[str] = []
        ev_block = re.search(r"支持证据[：:]([\s\S]*?)(?=❌|⚠️|\Z)", arbiter_text)
        if ev_block:
            for line in ev_block.group(1).strip().splitlines():
                line = re.sub(r"^\s*[\d\-•]+[.、)\s]*", "", line).strip()
                if line:
                    evidence.append(line)
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
                if not line or re.match(r"^排除诊断[：:]?\s*$", line):
                    continue
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
