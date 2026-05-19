"""VLM inference — supports local 4-bit Qwen2.5-VL and DashScope API."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

from app.config import Settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VLMInference:
    """Unified VLM inference — auto-selects local or API backend.

    Local mode: loads Qwen2.5-VL-7B with 4-bit quantization.
    API mode: calls DashScope Qwen-VL API (no GPU required).
    """

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    @property
    def _use_api(self) -> bool:
        return bool(self.config.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        if self._use_api:
            return await self._generate_api(system_prompt, user_content, images)
        return await self._generate_local(system_prompt, user_content, images)

    # ------------------------------------------------------------------
    # Local backend
    # ------------------------------------------------------------------

    async def _generate_local(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        import torch

        self._load_local()
        return await asyncio.to_thread(
            self._generate_local_sync, system_prompt, user_content, images
        )

    def _load_local(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration

        logger.info("Loading local model from %s …", self.config.model_path)

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
            self.config.model_path, trust_remote_code=True
        )
        logger.info("Local model loaded.")

    def _generate_local_sync(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        import torch

        assert self._model is not None and self._processor is not None

        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------
    # API backend (DashScope / OpenAI-compatible)
    # ------------------------------------------------------------------

    async def _generate_api(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        """Call DashScope multimodal API (OpenAI-compatible endpoint)."""
        return await asyncio.to_thread(
            self._generate_api_sync, system_prompt, user_content, images
        )

    def _generate_api_sync(
        self,
        system_prompt: str,
        user_content: list[dict],
        images: list[Image.Image],
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "API mode requires openai package. Install with: pip install openai"
            )

        api_key = self.config.api_key
        if not api_key:
            raise ValueError("API mode requires an API key (set AGRIMIND_API_KEY)")

        base_url = self.config.api_base_url or (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = self.config.api_model or "qwen-vl-plus"

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

        # Convert user_content to OpenAI multimodal format
        api_content: list[dict] = []

        # Add images first (OpenAI expects images before text for VL)
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            api_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        for item in user_content:
            if item.get("type") == "text":
                api_content.append({"type": "text", "text": str(item.get("text", ""))})
            # Skip image items — already added above from the images list

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": api_content},
        ]

        logger.info("Calling %s via API (model=%s) …", base_url, model)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        text = resp.choices[0].message.content or ""
        logger.info("API response: %d chars", len(text))
        return text.strip()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Settings) -> VLMInference:
        return cls(config)
