"""Configuration models and persistence."""

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class BackendConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    type: str  # ollama | lmstudio | llamacpp | cloud
    host: str = "127.0.0.1"
    port: int = 11434
    model: str = ""
    enabled: bool = True
    disable_thinking: bool = False
    api_key: str = ""          # for cloud backends (Moonshot, OpenRouter, etc.)
    api_base: str = ""         # custom base URL for cloud (e.g. https://api.moonshot.cn/v1)

    @property
    def base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}"


class GlobalConfig(BaseModel):
    proxy_model_name: str = "proxy-model"
    timeout: int = 120
    disable_thinking_global: bool = False
    backends: list[BackendConfig] = Field(default_factory=list)


_config: GlobalConfig | None = None


def load_config() -> GlobalConfig:
    global _config
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text())
        _config = GlobalConfig(**raw)
    else:
        _config = GlobalConfig()
        save_config()
    return _config


def save_config() -> None:
    global _config
    if _config is None:
        _config = GlobalConfig()
    CONFIG_PATH.write_text(json.dumps(_config.model_dump(), indent=2))


def get_config() -> GlobalConfig:
    global _config
    if _config is None:
        return load_config()
    return _config


def should_filter_thinking(backend: BackendConfig) -> bool:
    """Whether to inject /no_think for this backend."""
    cfg = get_config()
    return cfg.disable_thinking_global or backend.disable_thinking
