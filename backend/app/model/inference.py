"""Qwen2.5-VL inference wrapper shared by all DDP agents."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from app.config import Settings

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


class VLMInference:
    """Wraps Qwen2.5-VL for inference.  All agents share one instance."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._model: Qwen2_5_VLForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None

    def _load(self) -> None:
        """Lazy-load model & processor on first call (heavy, ~10 GB)."""
        if self._model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration

        logger.info("Loading model from %s …", self.config.model_path)

        quant_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if self.config.load_in_4bit
            else None
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            quantization_config=quant_config,
            device_map=self.config.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        """Run one VLM inference call in a background thread.

        Args:
            system_prompt: Agent role / persona definition.
            user_content: Mixed content items, e.g.
                ``[{"type": "text", "text": "..."}, {"type": "image", "image": img}]``
            images: PIL Image objects referenced by *user_content*.

        Returns:
            The generated text response (decoded).
        """
        self._load()
        return await asyncio.to_thread(
            self._generate_sync, system_prompt, user_content, images
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        assert self._model is not None and self._processor is not None

        # Build chat messages in the format Qwen2.5-VL expects.
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        # apply_chat_template returns token ids (or text depending on
        # processor version).  We use the processor's __call__ as well
        # to handle image pixel values.
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )

        # Slice off the input tokens to keep only the generated part.
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Settings) -> VLMInference:
        """Create an inference instance (model is lazy-loaded)."""
        return cls(config)
