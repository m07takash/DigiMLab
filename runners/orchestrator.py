"""テスト実行オーケストレーター"""
import asyncio
import importlib
import yaml
from datetime import datetime
from pathlib import Path

import env_loader  # system.env を自動ロード
from env_loader import expand
from runners.api_client import AIClient, ModelConfig
from evaluators.base import BaseEvaluator, EvalReport
from reporters.html_reporter import HTMLReporter


def _load_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        id=raw["id"],
        name=raw["name"],
        type=raw["type"],
        base_url=raw.get("base_url", ""),
        model=raw.get("model", ""),
        api_key=expand(raw.get("api_key", "")),
        default_params=raw.get("default_params", {}),
    )


def _load_evaluator(class_path: str, suite_config: dict, judge: AIClient | None) -> BaseEvaluator:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config=suite_config, judge_client=judge)


class Orchestrator:
    def __init__(self, models_config_path: str = "config/models.yaml", settings_path: str = "config/settings.yaml"):
        with open(models_config_path, encoding="utf-8") as f:
            self.models_cfg = yaml.safe_load(f)
        with open(settings_path, encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

    def _build_judge(self) -> AIClient | None:
        judge_cfg = self.models_cfg.get("judge")
        if not judge_cfg:
            return None
        cfg = ModelConfig(
            id="judge",
            name="Judge",
            type=judge_cfg.get("type", "openai"),
            model=expand(judge_cfg.get("model", "gpt-4o")),
            api_key=expand(judge_cfg.get("api_key", "")),
        )
        return AIClient(cfg)

    async def run_all(self, model_ids: list[str] | None = None) -> list[EvalReport]:
        judge = self._build_judge()
        concurrency = self.settings.get("concurrency", 3)
        all_reports: list[EvalReport] = []

        models = self.models_cfg.get("models", [])
        if model_ids:
            models = [m for m in models if m["id"] in model_ids]

        suites = {
            k: v for k, v in self.settings.get("test_suites", {}).items()
            if v.get("enabled", False)
        }

        for model_raw in models:
            model_cfg = _load_model_config(model_raw)
            client = AIClient(model_cfg)
            print(f"\n=== Model: {model_cfg.name} ===")

            for suite_name, suite_cfg in suites.items():
                print(f"  Running: {suite_name}")
                evaluator = _load_evaluator(suite_cfg["evaluator"], suite_cfg, judge)
                report = await evaluator.run(
                    client=client,
                    dataset_path=suite_cfg["dataset"],
                    sample_size=suite_cfg.get("sample_size"),
                    concurrency=concurrency,
                )
                all_reports.append(report)
                print(f"  Summary: {report.summary}")

        return all_reports

    def run(self, model_ids: list[str] | None = None) -> list[EvalReport]:
        reports = asyncio.run(self.run_all(model_ids))

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        reporter = HTMLReporter(self.settings["report"]["output_dir"])
        reporter.generate(reports, run_id)

        # JSON出力
        import json, dataclasses
        json_path = Path(self.settings["report"]["output_dir"]) / f"report_{run_id}.json"
        json_path.write_text(
            json.dumps([dataclasses.asdict(r) for r in reports], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[Report] JSON → {json_path}")
        return reports
