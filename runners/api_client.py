"""各種AIモデルAPIへの統一クライアント"""
import asyncio
import env_loader  # system.env を自動ロード
from dataclasses import dataclass, field
from env_loader import expand


@dataclass
class RetryConfig:
    """リトライ設定"""
    max_retries: int = 3
    initial_delay_seconds: float = 2.0
    backoff_factor: float = 2.0
    retryable_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


@dataclass
class ModelConfig:
    id: str
    name: str
    type: str          # openai_compatible | openai | anthropic | custom
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    default_params: dict = None

    def __post_init__(self):
        self.api_key = expand(self.api_key)


def _extract_status_code(exc: Exception) -> int | None:
    """例外からHTTPステータスコードを抽出する"""
    if hasattr(exc, "status_code"):
        return exc.status_code
    if hasattr(exc, "status"):
        return exc.status
    return None


def _extract_retry_after(exc: Exception) -> float | None:
    """例外からRetry-Afterヘッダの値を抽出する"""
    headers = None
    if hasattr(exc, "response") and hasattr(exc.response, "headers"):
        headers = exc.response.headers
    elif hasattr(exc, "headers"):
        headers = exc.headers
    if headers:
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    return None


class AIClient:
    def __init__(self, config: ModelConfig, retry_config: RetryConfig | None = None):
        self.config = config
        self.retry = retry_config or RetryConfig()
        self.session_id: str | None = None  # テストセット単位で設定される

    async def chat(self, messages: list[dict], **kwargs) -> str:
        params = {**(self.config.default_params or {}), **kwargs}

        if self.config.type in ("openai", "openai_compatible"):
            call = self._openai_chat
        elif self.config.type == "anthropic":
            call = self._anthropic_chat
        elif self.config.type == "custom":
            call = self._custom_chat
        else:
            raise ValueError(f"Unknown model type: {self.config.type}")

        return await self._call_with_retry(call, messages, params)

    async def _call_with_retry(self, call, messages, params) -> str:
        last_exc = None
        for attempt in range(1 + self.retry.max_retries):
            try:
                return await call(messages, params)
            except Exception as exc:
                last_exc = exc
                status = _extract_status_code(exc)

                # リトライ対象外のエラーは即座に raise
                if status is not None and status not in self.retry.retryable_status_codes:
                    raise

                # 最終試行ならリトライしない
                if attempt >= self.retry.max_retries:
                    break

                # 待機時間を算出（Retry-After があればそちらを優先）
                retry_after = _extract_retry_after(exc)
                if retry_after:
                    delay = retry_after
                else:
                    delay = self.retry.initial_delay_seconds * (
                        self.retry.backoff_factor ** attempt
                    )

                status_msg = f" (HTTP {status})" if status else ""
                print(
                    f"  [Retry] {self.config.name}: "
                    f"attempt {attempt + 1}/{self.retry.max_retries} failed{status_msg}, "
                    f"retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"{self.config.name}: {self.retry.max_retries}回リトライしましたが失敗しました: {last_exc}"
        ) from last_exc

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

    async def _custom_chat(self, messages: list[dict], params: dict) -> str:
        """
        カスタムAPI（DigitalMATSUMOTO等）へのリクエスト。
        session_id はテストセット単位で固定される。
        default_params に service_info, user_info, agent_file, engine 等を指定する。
        """
        import aiohttp
        import json

        # messagesからuser_inputを抽出（systemは除外、userメッセージを結合）
        user_input = "\n".join(
            m["content"] for m in messages if m["role"] != "system"
        )
        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        if system_prompt:
            user_input = f"{system_prompt}\n\n{user_input}"

        # リクエストボディを構築
        body = {
            "service_info": params.get("service_info", {"SERVICE_ID": "DigiMLab", "SERVICE_DATA": {}}),
            "user_info": params.get("user_info", {"USER_ID": "anonymous", "USER_DATA": {}}),
            "session_id": self.session_id or "default_session",
            "user_input": user_input,
        }
        # 基本パラメータ
        for key in ("agent_file", "engine", "situation"):
            if key in params:
                body[key] = params[key]
        # boolパラメータ
        for key in ("stream_mode", "save_digest", "memory_use",
                    "magic_word_use", "meta_search", "rag_query_gene"):
            if key in params and params[key] is not None:
                body[key] = params[key]
        # web_search関連
        if params.get("web_search"):
            body["web_search"] = True
            if "web_search_engine" in params:
                body["web_search_engine"] = params["web_search_engine"]
        elif "web_search" in params:
            body["web_search"] = False

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.config.base_url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    error = type("HTTPError", (Exception,), {"status_code": resp.status})()
                    error.args = (f"HTTP {resp.status}: {text[:200]}",)
                    raise error
                data = await resp.json()
                return str(data.get("response", "")).strip()
