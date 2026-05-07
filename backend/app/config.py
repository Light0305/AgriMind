"""Application settings — loaded once, shared everywhere."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration consumed by VLMInference and FastAPI."""

    model_path: str = "/home/user/work/lch/robot/models/qwen2.5-vl-7b"
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7
    device: str = "auto"

    model_config = {"env_prefix": "AGRIMIND_"}


settings = Settings()
