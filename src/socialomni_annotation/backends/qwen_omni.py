from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from integrations.models.utils.omni_http_client import OmniHttpClient


class ModelBackend(Protocol):
    def generate(self, prompt: str, video_path: str | None = None) -> str:
        ...


@dataclass(frozen=True)
class QwenBackendConfig:
    backend: str = "mock"
    model: str = "qwen3-omni"
    server_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class MockQwenOmniBackend:
    def generate(self, prompt: str, video_path: str | None = None) -> str:
        if "raw_start_sec" in prompt and "第一局正式游戏开始" in prompt:
            return (
                "{"
                "\"game_id\":\"mock_game\","
                "\"player_id\":\"mock_player\","
                "\"raw_start_sec\":0.0,"
                "\"evidence\":\"mock sync offset\","
                "\"confidence\":0.1"
                "}"
            )
        clip_id = Path(video_path).stem if video_path else "mock_clip"
        parts = clip_id.split("_")
        game_id = parts[0] if len(parts) >= 4 else "mock_game"
        player_id = parts[1] if len(parts) >= 4 else "mock_player"
        start_sec = float(parts[-2]) if len(parts) >= 4 and parts[-2].isdigit() else 0.0
        end_sec = float(parts[-1]) if len(parts) >= 4 and parts[-1].isdigit() else 1.0
        return (
            "["
            "{"
            f"\"clip_id\":\"{clip_id}\","
            f"\"game_id\":\"{game_id}\","
            f"\"player_id\":\"{player_id}\","
            f"\"start_sec\":{start_sec},"
            f"\"end_sec\":{end_sec},"
            "\"event_type\":\"mock_observation\","
            "\"description\":\"Mock backend placeholder event.\","
            "\"visible_players\":[],"
            "\"mentioned_players\":[],"
            "\"location\":null,"
            "\"confidence\":0.1,"
            "\"evidence\":\"mock response\""
            "}"
            "]"
        )


class LocalQwenOmniServerBackend:
    def __init__(self, server_url: str) -> None:
        self.client = OmniHttpClient(server_url)

    def generate(self, prompt: str, video_path: str | None = None) -> str:
        if not video_path:
            raise ValueError("Local Qwen3-Omni server backend requires video_path")
        answer = self.client.call_api(video_path, prompt, use_video=True, use_audio=True)
        if answer is None:
            raise RuntimeError("Qwen3-Omni local server returned no answer")
        return answer


class OpenAICompatibleBackend:
    def __init__(self, config: QwenBackendConfig) -> None:
        from openai import OpenAI

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in environment variable {config.api_key_env}")
        self.model = config.model
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, prompt: str, video_path: str | None = None) -> str:
        content = prompt
        if video_path:
            content = f"{prompt}\n\n本地视频路径：{video_path}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def create_backend(config: QwenBackendConfig) -> ModelBackend:
    backend = config.backend.lower()
    if backend == "mock":
        return MockQwenOmniBackend()
    if backend in {"local", "local-server", "server"}:
        server_url = config.server_url or os.getenv("QWEN3_OMNI_SERVER_URL")
        if not server_url:
            server_url = "http://127.0.0.1:5090"
        return LocalQwenOmniServerBackend(server_url)
    if backend in {"openai", "openai-compatible"}:
        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
