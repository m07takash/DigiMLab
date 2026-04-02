"""評価器の基底クラス"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import asyncio
import random
from runners.api_client import AIClient


@dataclass
class TestCase:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)
    expected: Any = None


@dataclass
class TestResult:
    case_id: str
    model_id: str
    response: str
    score: float | None = None
    score_detail: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    suite_name: str
    model_id: str
    model_name: str
    results: list[TestResult]
    summary: dict = field(default_factory=dict)


class BaseEvaluator(ABC):
    def __init__(self, config: dict, judge_client: AIClient | None = None):
        self.config = config
        self.judge = judge_client

    @abstractmethod
    def load_dataset(self, path: str) -> list[TestCase]:
        """テストセットを読み込む"""
        ...

    @abstractmethod
    async def score(self, case: TestCase, result: TestResult) -> TestResult:
        """スコアリング。resultにscore/score_detailを設定して返す"""
        ...

    def _sample_cases(
        self,
        cases: list[TestCase],
        sample_size: int | None,
        sampling_mode: str = "head",
        random_seed: int | None = None,
    ) -> list[TestCase]:
        """sample_size に応じてテストケースを抽出する"""
        if not sample_size or sample_size >= len(cases):
            return cases

        if sampling_mode == "random":
            rng = random.Random(random_seed)
            return rng.sample(cases, sample_size)
        else:
            # "head" — 先頭N件
            return cases[:sample_size]

    async def run(
        self,
        client: AIClient,
        dataset_path: str,
        sample_size: int | None = None,
        concurrency: int = 3,
        request_delay_seconds: float = 0.0,
        sampling_mode: str = "head",
        random_seed: int | None = None,
    ) -> EvalReport:
        cases = self.load_dataset(dataset_path)
        cases = self._sample_cases(cases, sample_size, sampling_mode, random_seed)

        semaphore = asyncio.Semaphore(concurrency)
        results: list[TestResult] = []

        async def run_one(index: int, case: TestCase) -> TestResult:
            async with semaphore:
                # リクエスト間隔を空ける（初回以外）
                if request_delay_seconds > 0 and index > 0:
                    await asyncio.sleep(request_delay_seconds)

                system_prompt = self.config.get("system_prompt", "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": case.prompt})
                try:
                    response = await client.chat(messages)
                    result = TestResult(
                        case_id=case.id,
                        model_id=client.config.id,
                        response=response,
                    )
                    result = await self.score(case, result)
                except Exception as e:
                    result = TestResult(
                        case_id=case.id,
                        model_id=client.config.id,
                        response="",
                        error=str(e),
                    )
                return result

        tasks = [run_one(i, c) for i, c in enumerate(cases)]
        results = await asyncio.gather(*tasks)

        summary = self.compute_summary(list(results))
        return EvalReport(
            suite_name=self.config.get("description", ""),
            model_id=client.config.id,
            model_name=client.config.name,
            results=list(results),
            summary=summary,
        )

    def compute_summary(self, results: list[TestResult]) -> dict:
        valid = [r for r in results if r.score is not None]
        if not valid:
            return {"error": "No valid results"}
        avg = sum(r.score for r in valid) / len(valid)
        return {"total": len(results), "valid": len(valid), "avg_score": round(avg, 4)}
