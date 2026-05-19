"""Application settings — loaded once, shared everywhere."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration consumed by VLMInference and FastAPI."""

    # ── Local model ──────────────────────────────────────────────────
    model_path: str = "models/qwen2.5-vl-7b"
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7
    device: str = "auto"

    # ── API mode (DashScope / OpenAI-compatible) ─────────────────────
    # If api_key is set, the system uses API mode instead of local model.
    api_key: str = ""
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_model: str = "qwen-vl-plus"

    model_config = {"env_prefix": "AGRIMIND_"}


settings = Settings()
