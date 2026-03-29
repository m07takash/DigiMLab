"""各種AIモデルAPIへの統一クライアント"""
import env_loader  # system.env を自動ロード
from dataclasses import dataclass
from env_loader import expand


@dataclass
class ModelConfig:
    id: str
    name: str
    type: str          # openai_compatible | openai | anthropic
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    default_params: dict = None

    def __post_init__(self):
        self.api_key = expand(self.api_key)


class AIClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    async def chat(self, messages: list[dict], **kwargs) -> str:
        params = {**self.config.default_params, **kwargs}

        if self.config.type in ("openai", "openai_compatible"):
            return await self._openai_chat(messages, params)
        elif self.config.type == "anthropic":
            return await self._anthropic_chat(messages, params)
        else:
            raise ValueError(f"Unknown model type: {self.config.type}")

    async def _openai_chat(self, messages: list[dict], params: dict) -> str:
        from openai import AsyncOpenAI
        base_url = self.config.base_url or None
        client = AsyncOpenAI(
            api_key=self.config.api_key or "dummy",
            base_url=base_url,
        )
        model = self.config.model or "gpt-4o"
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content.strip()

    async def _anthropic_chat(self, messages: list[dict], params: dict) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]
        resp = await client.messages.create(
            model=self.config.model or "claude-sonnet-4-5",
            system=system,
            messages=user_messages,
            **params,
        )
        return resp.content[0].text.strip()
