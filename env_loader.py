"""
system.env ファイルを読み込み、アプリ全体で使える環境変数として展開する。
他のモジュールより先にインポートすること。

  from env_loader import env

  api_key = env("OPENAI_API_KEY")
  model   = env("JUDGE_MODEL", default="gpt-4o")
"""
from __future__ import annotations

import os
import re
from pathlib import Path


_ENV_FILE = Path(__file__).parent / "system.env"
_loaded: dict[str, str] = {}


def _parse(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result


def load(env_file: str | Path = _ENV_FILE, *, override: bool = False) -> None:
    """
    system.env を読み込み os.environ へ反映する。
    override=True にすると既存の環境変数も上書きする。
    """
    global _loaded
    path = Path(env_file)
    if not path.exists():
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {path}\n"
            f"system.env.example をコピーして作成してください。"
        )
    _loaded = _parse(path)
    for k, v in _loaded.items():
        if override or k not in os.environ:
            os.environ[k] = v


def env(key: str, default: str | None = None) -> str:
    """
    system.env またはプロセス環境変数から値を取得する。
    見つからず default も None の場合は KeyError を送出する。
    """
    value = os.environ.get(key) or _loaded.get(key)
    if value is not None:
        return value
    if default is not None:
        return default
    raise KeyError(
        f"環境変数 '{key}' が設定されていません。system.env を確認してください。"
    )


def expand(value: str) -> str:
    """
    "${VAR_NAME}" 形式の文字列を system.env の値で展開する。
    models.yaml 内の値を処理するために使う。
    """
    return re.sub(r"\$\{(\w+)\}", lambda m: env(m.group(1), default=m.group(0)), value)


# モジュール読み込み時に自動ロード（ファイルがなければスキップ）
try:
    load()
except FileNotFoundError:
    pass
